#!/usr/bin/env python3
"""Validate and record a task state transition. See WORKFLOW.md for the
full transition table this enforces.

Standard library only.

Example:
    python .agent-system/scripts/advance_task.py --task-id fix-histogram-binning --to ready_to_run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    TaskError,
    eprint,
    iteration_dir_name,
    list_iterations,
    load_json,
    load_state,
    load_task,
    now_iso,
    save_state,
    task_dir,
)

ALL_STATES = {
    "created", "planning", "documenting", "ready_for_implementation",
    "awaiting_implementation_approval", "implementing",
    "ready_to_run", "running", "ready_for_review", "reviewing",
    "changes_requested", "awaiting_human_input", "blocked", "approved", "closed",
}

TRANSITIONS: dict[str, set[str]] = {
    "created": {"planning"},
    "planning": {"documenting"},
    "documenting": {"ready_for_implementation"},
    "ready_for_implementation": {"awaiting_implementation_approval", "implementing"},
    "awaiting_implementation_approval": {"implementing"},
    "implementing": {"ready_to_run"},
    "ready_to_run": {"running"},
    "running": {"ready_for_review"},
    "ready_for_review": {"reviewing"},
    "reviewing": {"approved", "changes_requested", "awaiting_human_input", "blocked"},
    # Two loop-back edges: to 'documenting' when the document is at fault (the
    # common case), to 'implementing' when the document was right and only the
    # implementation was wrong. See agents/reviewer.md § "Routing".
    "changes_requested": {"documenting", "implementing", "awaiting_human_input"},
    "approved": {"closed"},
    "closed": set(),
}

# The loop-back edges that open a new iteration.
NEW_ITERATION_TARGETS = {"documenting", "implementing"}

# From these two states, a human has already made the resumption decision;
# allow moving to any valid state rather than modeling every possible
# resume path in TRANSITIONS.
RESUME_FROM = {"awaiting_human_input", "blocked"}

TERMINAL_OR_PAUSED = {"awaiting_human_input", "blocked", "approved", "closed"}


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-id", required=True)
    p.add_argument("--to", required=True, choices=sorted(ALL_STATES))
    p.add_argument("--note", default="", help="Optional free-text note recorded with the transition.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Bypass the approval/finding guards (still enforces the transition table). Use sparingly.",
    )
    return p.parse_args(argv)


def check_approval_guard(task: dict, target: str, from_state: str) -> str | None:
    approvals = task.get("approvals", {})
    if target == "implementing" and from_state in {"ready_for_implementation", "awaiting_implementation_approval"}:
        if approvals.get("implementation_required") and not approvals.get("implementation_granted"):
            return (
                "task.json approvals.implementation_required is true but "
                "implementation_granted is false — get human approval of the selection "
                "document and set it before implementing."
            )
    if target == "running":
        if approvals.get("execution_required") and not approvals.get("execution_granted"):
            return (
                "task.json approvals.execution_required is true but "
                "execution_granted is false — get human approval before running commands."
            )
    return None


def check_approved_guard(tid: str, current_iteration: int) -> str | None:
    if current_iteration < 1:
        return "cannot approve before any iteration has run."
    fb_path = task_dir(tid) / "iterations" / iteration_dir_name(current_iteration) / "feedback.json"
    try:
        feedback = load_json(fb_path)
    except TaskError:
        return f"no feedback.json for iteration {current_iteration} — the Reviewer must write one first."
    open_blockers = [
        f["finding_id"]
        for f in feedback.get("findings", [])
        if f.get("severity") in {"blocker", "major"} and f.get("status") == "open"
    ]
    if open_blockers:
        return f"cannot approve: open blocker/major findings remain: {', '.join(open_blockers)}"
    return None


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    tid = args.task_id

    try:
        task = load_task(tid)
        state = load_state(tid)
    except TaskError as exc:
        eprint(f"error: {exc}")
        return 1

    current_status = state["status"]
    current_iteration = state["current_iteration"]
    target = args.to

    allowed = TRANSITIONS.get(current_status, set())
    if current_status in RESUME_FROM:
        allowed = ALL_STATES - {current_status}
    if target not in allowed:
        eprint(
            f"error: invalid transition {current_status!r} -> {target!r} for task {tid!r}.\n"
            f"allowed next states from {current_status!r}: {sorted(allowed) or '(none — terminal state)'}"
        )
        return 1

    is_new_iteration = current_status == "changes_requested" and target in NEW_ITERATION_TARGETS
    is_first_iteration_start = current_status == "created" and target == "planning"

    max_iterations = task.get("max_iterations", 3)
    if is_new_iteration and (current_iteration + 1) > max_iterations:
        eprint(
            f"error: iteration limit reached ({current_iteration}/{max_iterations}). "
            "Advance to 'awaiting_human_input' instead, or raise max_iterations in task.json "
            "with human approval."
        )
        return 1

    if not args.force:
        guard_msg = check_approval_guard(task, target, current_status)
        if guard_msg:
            eprint(f"error: {guard_msg}\n(use --force to bypass — not recommended)")
            return 1
        if target == "approved":
            guard_msg = check_approved_guard(tid, current_iteration)
            if guard_msg:
                eprint(f"error: {guard_msg}\n(use --force to bypass — not recommended)")
                return 1

    new_iteration = current_iteration
    if is_first_iteration_start:
        new_iteration = 1
    elif is_new_iteration:
        new_iteration = current_iteration + 1
        new_dir = task_dir(tid) / "iterations" / iteration_dir_name(new_iteration)
        if new_dir.exists():
            eprint(f"error: refusing to reuse existing iteration directory: {new_dir}")
            return 1
        existing = list_iterations(tid)
        if existing and new_iteration != existing[-1] + 1:
            eprint(f"error: iteration numbering would not be sequential (existing: {existing}).")
            return 1
        new_dir.mkdir(parents=True)

    state["status"] = target
    state["current_iteration"] = new_iteration
    state["updated_at"] = now_iso()
    state.setdefault("transitions_log", []).append(
        {"from": current_status, "to": target, "iteration": new_iteration, "at": now_iso(), "note": args.note}
    )
    save_state(tid, state)

    print(f"{tid}: {current_status} -> {target} (iteration {new_iteration})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
