#!/usr/bin/env python3
"""Create a new .agent-system task directory.

Standard library only. See .agent-system/README.md for the overall workflow.

Example:
    python .agent-system/scripts/create_task.py \\
        --task-id fix-histogram-binning \\
        --title "Fix histogram binning" \\
        --request-file request.txt
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    TaskError,
    eprint,
    git_metadata,
    is_safe_task_id,
    now_iso,
    save_json,
    save_state,
    save_task,
    task_dir,
)

TASK_TEMPLATE = {
    "schema_version": "1.0",
    "mode": "implementation",
    "status": "created",
    "current_iteration": 0,
    "max_iterations": 3,
    "scope": {"include": [], "exclude": []},
    "constraints": [],
    "acceptance_criteria": [],
    "risk": {"level": "low", "categories": []},
    "approvals": {
        "implementation_required": True,
        "implementation_granted": False,
        "execution_required": False,
        "execution_granted": False,
        "production_required": True,
        "production_granted": False,
    },
    "validation": {"allowed_commands": [], "forbidden_commands": [], "required_checks": []},
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-id", required=True)
    p.add_argument("--title", required=True)
    req = p.add_mutually_exclusive_group(required=True)
    req.add_argument("--request-file", type=Path, help="Path to a text file with the user request.")
    req.add_argument("--request", help="The user request, given inline.")
    p.add_argument("--mode", default="implementation", choices=["implementation", "review", "investigation"])
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--risk-level", default="low", choices=["low", "medium", "high"])
    p.add_argument("--scope-include", action="append", default=[], help="Repeatable.")
    p.add_argument("--scope-exclude", action="append", default=[], help="Repeatable.")
    p.add_argument("--acceptance-criteria", action="append", default=[], help="Repeatable.")
    p.add_argument("--created-by", default="human", help="e.g. human, coding-agent, automation.")
    p.add_argument(
        "--implementation-preapproved",
        action="store_true",
        help="Set approvals.implementation_granted=true immediately (skips the approval gate).",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if not is_safe_task_id(args.task_id):
        eprint(
            f"error: {args.task_id!r} is not a safe task id "
            "(use lowercase letters, digits, hyphens only; no path separators)."
        )
        return 1

    tdir = task_dir(args.task_id)
    if tdir.exists():
        eprint(f"error: task directory already exists, refusing to overwrite: {tdir}")
        return 1

    if args.request_file is not None:
        try:
            request_text = args.request_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            eprint(f"error: could not read --request-file: {exc}")
            return 1
    else:
        request_text = args.request.strip()

    if not request_text:
        eprint("error: request text is empty.")
        return 1

    task = dict(TASK_TEMPLATE)
    task["task_id"] = args.task_id
    task["title"] = args.title
    task["request"] = request_text
    task["mode"] = args.mode
    task["max_iterations"] = args.max_iterations
    task["scope"] = {"include": args.scope_include, "exclude": args.scope_exclude}
    task["acceptance_criteria"] = args.acceptance_criteria
    task["risk"] = {"level": args.risk_level, "categories": []}
    task["approvals"] = dict(TASK_TEMPLATE["approvals"])
    if args.implementation_preapproved:
        task["approvals"]["implementation_granted"] = True

    git_meta = git_metadata()
    task["metadata"] = {
        "created_at": now_iso(),
        "created_by": args.created_by,
        **git_meta,
    }

    try:
        tdir.mkdir(parents=True)
        (tdir / "iterations" / "001").mkdir(parents=True)
        save_task(args.task_id, task)
        save_state(
            args.task_id,
            {
                "task_id": args.task_id,
                "current_iteration": 0,
                "status": "created",
                "updated_at": now_iso(),
                "transitions_log": [],
            },
        )
    except OSError as exc:
        eprint(f"error: failed to create task directory: {exc}")
        shutil.rmtree(tdir, ignore_errors=True)
        return 1
    except TaskError as exc:
        eprint(f"error: {exc}")
        shutil.rmtree(tdir, ignore_errors=True)
        return 1

    print(str(tdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
