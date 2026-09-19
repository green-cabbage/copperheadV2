#!/usr/bin/env python3
"""Summarize a task across all its iterations.

Read-only by default (prints to stdout), per POLICY.md's "read-only
inspection before modification" default — pass --write to actually create
<task-dir>/final-summary.md.

Standard library only.

Example:
    python .agent-system/scripts/summarize_task.py --task-id fix-histogram-binning
    python .agent-system/scripts/summarize_task.py --task-id fix-histogram-binning --write
    python .agent-system/scripts/summarize_task.py --task-dir .agent-system/examples/hello-world-task
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import TaskError, eprint, load_json, task_dir  # noqa: E402

ITERATION_DIR_NAME_RE = re.compile(r"^(\d{3})$")


def list_iterations_at(tdir: Path) -> list[int]:
    iterations_root = tdir / "iterations"
    if not iterations_root.is_dir():
        return []
    found = []
    for child in iterations_root.iterdir():
        if child.is_dir():
            m = ITERATION_DIR_NAME_RE.match(child.name)
            if m:
                found.append(int(m.group(1)))
    return sorted(found)


def build_summary(label: str, tdir: Path) -> str:
    task = load_json(tdir / "task.json")
    state = load_json(tdir / "current-state.json")
    iterations = list_iterations_at(tdir)

    lines: list[str] = []
    lines.append(f"# Final summary — {label}")
    lines.append("")
    lines.append(f"**Final state:** {state['status']}")
    lines.append(f"**Iterations used:** {state['current_iteration']} / {task['max_iterations']}")
    lines.append("")
    lines.append("## Selection document")
    lines.append("")
    lines.append("| Iteration | Requirements | Verify-tagged | Not enforced | Open questions |")
    lines.append("|---|---|---|---|---|")
    any_doc = False
    latest_report: dict | None = None
    for n in iterations:
        dr_path = tdir / "iterations" / f"{n:03d}" / "doc-report.json"
        if not dr_path.exists():
            continue
        try:
            dr = load_json(dr_path)
        except TaskError:
            lines.append(f"| {n:03d} | (unreadable doc-report.json) | | | |")
            continue
        any_doc = True
        latest_report = dr
        reqs = [r for r in dr.get("requirements", []) if isinstance(r, dict)]
        verify = sum(1 for r in reqs if r.get("classification") == "verify")
        unenforced = sum(1 for r in reqs if r.get("enforcement") in {"stored_only", "absent"})
        lines.append(
            f"| {n:03d} | {len(reqs)} | {verify} | {unenforced} | {len(dr.get('open_questions', []))} |"
        )
    if not any_doc:
        lines.append("| — | (no doc-report.json recorded in any iteration) | | | |")
    lines.append("")

    if latest_report is not None:
        questions = latest_report.get("open_questions", [])
        if questions:
            lines.append(
                "Open questions from the latest selection document — these are physics "
                "questions the document could not settle, and they outlive the task:"
            )
            lines.append("")
            for q in questions:
                lines.append(f"- {q}")
            lines.append("")
        misrouted = latest_report.get("misrouted_findings", [])
        if misrouted:
            lines.append(
                "Findings the Documentation Generator returned as not its defect "
                "(see `agents/reviewer.md` § \"Routing\"):"
            )
            lines.append("")
            for m in misrouted:
                lines.append(f"- {m.get('finding_id', '?')}: {m.get('reason', '')}")
            lines.append("")

    lines.append("## Changes made")
    lines.append("")
    lines.append(
        "See each iteration's `generator-report.md` § \"Files changed\" for the file-level detail "
        "(free-form narrative, not machine-readable). Commands *proposed* to validate those changes:"
    )
    lines.append("")
    proposed: dict[str, str] = {}
    for n in iterations:
        idir = tdir / "iterations" / f"{n:03d}"
        cmds_path = idir / "proposed-commands.json"
        if cmds_path.exists():
            try:
                for cmd in load_json(cmds_path):
                    proposed.setdefault(cmd.get("command", ""), f"iteration {n:03d}")
            except TaskError:
                pass
    if not proposed:
        lines.append("- (no proposed-commands.json recorded in any iteration)")
    else:
        for cmd, where in proposed.items():
            lines.append(f"- `{cmd}` — {where}")
    lines.append("")

    lines.append("## Commands run")
    lines.append("")
    any_commands = False
    for n in iterations:
        rr_path = tdir / "iterations" / f"{n:03d}" / "run-report.json"
        if not rr_path.exists():
            continue
        try:
            rr = load_json(rr_path)
        except TaskError:
            continue
        for cmd in rr.get("commands", []):
            any_commands = True
            lines.append(
                f"- iteration {n:03d}: `{cmd.get('command')}` "
                f"→ exit={cmd.get('exit_code')} ({cmd.get('classification')})"
            )
    if not any_commands:
        lines.append("- (no run-report.json recorded in any iteration)")
    lines.append("")

    lines.append("## Reviewer decisions by iteration")
    lines.append("")
    lines.append("| Iteration | Decision | Notes |")
    lines.append("|---|---|---|")
    any_decision = False
    # Findings must be carried forward every iteration (see WORKFLOW.md), so
    # the same finding_id can appear with different statuses across
    # iterations — only the latest status per ID is what actually matters.
    latest_finding_status: dict[str, tuple[str, str]] = {}  # finding_id -> (status, description)
    for n in iterations:
        fb_path = tdir / "iterations" / f"{n:03d}" / "feedback.json"
        if not fb_path.exists():
            lines.append(f"| {n:03d} | (no feedback.json yet) | |")
            continue
        try:
            fb = load_json(fb_path)
        except TaskError:
            lines.append(f"| {n:03d} | (unreadable feedback.json) | |")
            continue
        any_decision = True
        summary = (fb.get("summary") or "").strip().replace("\n", " ")
        lines.append(f"| {n:03d} | {fb.get('decision')} | {summary} |")
        for finding in fb.get("findings", []):
            latest_finding_status[finding.get("finding_id")] = (
                finding.get("status"),
                finding.get("description", ""),
            )
    if not any_decision:
        lines.append("| — | (no iterations reviewed yet) | |")
    lines.append("")

    lines.append("## Findings left open, rejected, or deferred")
    lines.append("")
    unresolved = {
        fid: (status, desc)
        for fid, (status, desc) in latest_finding_status.items()
        if status != "fixed"
    }
    if unresolved:
        for fid, (status, desc) in unresolved.items():
            lines.append(f"- {fid} — {status} — {desc}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Unresolved limitations")
    lines.append("")
    any_limitation = False
    for n in iterations:
        rr_path = tdir / "iterations" / f"{n:03d}" / "run-report.json"
        if not rr_path.exists():
            continue
        try:
            rr = load_json(rr_path)
        except TaskError:
            continue
        for lim in rr.get("limitations", []):
            any_limitation = True
            lines.append(f"- iteration {n:03d}: {lim}")
    if not any_limitation:
        lines.append("- (none recorded)")
    lines.append("")

    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-id", help="Summarize tasks/<task-id>/.")
    group.add_argument(
        "--task-dir",
        type=Path,
        help="Summarize an explicit task-shaped directory outside tasks/ (e.g. examples/hello-world-task).",
    )
    p.add_argument("--write", action="store_true", help="Write <task-dir>/final-summary.md instead of only printing.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    tdir = args.task_dir if args.task_dir else task_dir(args.task_id)
    label = args.task_dir.name if args.task_dir else args.task_id

    if not tdir.is_dir():
        eprint(f"error: no such task directory: {tdir}")
        return 1

    try:
        summary = build_summary(label, tdir)
    except TaskError as exc:
        eprint(f"error: {exc}")
        return 1

    if args.write:
        out_path = tdir / "final-summary.md"
        out_path.write_text(summary, encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
