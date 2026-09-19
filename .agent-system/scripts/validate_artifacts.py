#!/usr/bin/env python3
"""Structurally validate task artifacts against schemas/*.schema.json, plus
a handful of workflow-specific checks (iteration numbering, immutability of
prior iterations, finding-ID continuity).

This is a lightweight, hand-rolled structural checker (required keys,
types, enums, simple patterns/minimums) — not a full JSON Schema draft-07
implementation — so it has no dependency beyond the standard library.

Standard library only.

Example:
    python .agent-system/scripts/validate_artifacts.py --task-id fix-histogram-binning
    python .agent-system/scripts/validate_artifacts.py --all
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    TASKS_DIR,
    TaskError,
    eprint,
    load_json,
    load_schema,
    task_dir,
)

ITERATION_DIR_NAME_RE = re.compile(r"^(\d{3})$")


def list_iterations_at(tdir: Path) -> list[int]:
    """Like _common.list_iterations, but works from an explicit directory
    rather than requiring a safe task-id under TASKS_DIR — needed so
    --task-dir can point at examples/ as well as tasks/."""
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

JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}

FINDING_ID_RE = re.compile(r"^REV-(\d{3,})$")


def check_type(value, expected, path: str, errors: list[str]) -> bool:
    types = expected if isinstance(expected, list) else [expected]
    # bool is a subclass of int in Python; only accept it where "boolean" is listed.
    if isinstance(value, bool):
        ok = "boolean" in types
    else:
        py_types = tuple(JSON_TYPE_MAP[t] for t in types if t in JSON_TYPE_MAP)
        ok = isinstance(value, py_types) if py_types else False
    if not ok:
        errors.append(f"{path}: expected type {types}, got {type(value).__name__}")
    return ok


def validate(instance, schema: dict, path: str, errors: list[str]) -> None:
    if "type" in schema:
        if not check_type(instance, schema["type"], path, errors):
            return
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: value {instance!r} not in allowed set {schema['enum']}")
    if "pattern" in schema and isinstance(instance, str):
        if not re.match(schema["pattern"], instance):
            errors.append(f"{path}: {instance!r} does not match pattern {schema['pattern']!r}")
    if "minimum" in schema and isinstance(instance, (int, float)):
        if instance < schema["minimum"]:
            errors.append(f"{path}: {instance} is below minimum {schema['minimum']}")

    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required key {key!r}")
        props = schema.get("properties", {})
        for key, subschema in props.items():
            if key in instance:
                validate(instance[key], subschema, f"{path}.{key}", errors)

    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            validate(item, schema["items"], f"{path}[{i}]", errors)


def validate_file(path: Path, schema_name: str, errors: list[str]) -> dict | None:
    try:
        data = load_json(path)
    except TaskError as exc:
        errors.append(str(exc))
        return None
    schema = load_schema(schema_name)
    validate(data, schema, str(path), errors)
    return data


def check_finding_continuity(tdir: Path, iterations: list[int], errors: list[str]) -> None:
    seen: dict[str, str] = {}  # finding_id -> status as of the last iteration seen
    max_number_seen = 0
    for n in iterations:
        fb_path = tdir / "iterations" / f"{n:03d}" / "feedback.json"
        if not fb_path.exists():
            continue
        try:
            feedback = load_json(fb_path)
        except TaskError as exc:
            errors.append(str(exc))
            continue
        current_ids = set()
        for finding in feedback.get("findings", []):
            fid = finding.get("finding_id", "")
            current_ids.add(fid)
            m = FINDING_ID_RE.match(fid)
            if not m:
                errors.append(f"iteration {n:03d}: malformed finding_id {fid!r}")
                continue
            num = int(m.group(1))
            if fid not in seen and num <= max_number_seen:
                errors.append(
                    f"iteration {n:03d}: finding_id {fid!r} reuses/precedes an already-used number "
                    f"(max seen so far: REV-{max_number_seen:03d}) — IDs must be introduced in "
                    "increasing order and never reused for a different issue."
                )
            max_number_seen = max(max_number_seen, num)
            seen[fid] = finding.get("status", "")
        # Findings are never deleted, per POLICY.md/reviewer.md — even a
        # "fixed" finding must still appear (with that status) in every
        # later iteration's feedback.json, not just up to the iteration
        # that fixed it.
        for fid in set(seen) - current_ids:
            errors.append(
                f"iteration {n:03d}: finding {fid!r} present in an earlier iteration but missing "
                "here — findings must be carried forward (with an updated status), never dropped."
            )


def validate_task(task_id: str, tdir: Path | None = None) -> list[str]:
    """Validate the task directory for task_id. If tdir is given explicitly
    (via --task-dir), it is used as-is instead of resolving task_id under
    TASKS_DIR — this lets the same checks run against examples/ directories
    that mirror the task-directory shape without living under tasks/."""
    errors: list[str] = []
    if tdir is None:
        tdir = task_dir(task_id)
    if not tdir.is_dir():
        return [f"no such task directory: {tdir}"]

    validate_file(tdir / "task.json", "task", errors)

    try:
        state = load_json(tdir / "current-state.json")
    except TaskError as exc:
        errors.append(str(exc))
        state = None

    iterations = list_iterations_at(tdir)
    if iterations != list(range(1, len(iterations) + 1)) and iterations:
        errors.append(f"iteration directories are not a contiguous sequence starting at 001: {iterations}")

    if state is not None and state.get("current_iteration", 0) > 0:
        if state["current_iteration"] not in iterations and iterations:
            errors.append(
                f"current-state.json says iteration {state['current_iteration']} but no such "
                f"iterations/ directory exists (found: {iterations})"
            )

    for n in iterations:
        idir = tdir / "iterations" / f"{n:03d}"
        run_report = idir / "run-report.json"
        review_report = idir / "review-report.json"
        feedback = idir / "feedback.json"
        if run_report.exists():
            validate_file(run_report, "run-report", errors)
        if review_report.exists():
            validate_file(review_report, "review-report", errors)
        if feedback.exists():
            validate_file(feedback, "feedback", errors)

    check_finding_continuity(tdir, iterations, errors)

    return errors


def all_task_ids() -> list[str]:
    if not TASKS_DIR.is_dir():
        return []
    return sorted(p.name for p in TASKS_DIR.iterdir() if p.is_dir() and (p / "task.json").exists())


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-id", help="Validate tasks/<task-id>/.")
    group.add_argument("--all", action="store_true", help="Validate every task under tasks/.")
    group.add_argument(
        "--task-dir",
        type=Path,
        help="Validate an explicit task-shaped directory outside tasks/ (e.g. examples/hello-world-task).",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.task_dir:
        jobs = [(args.task_dir.name, args.task_dir)]
    elif args.all:
        jobs = [(tid, None) for tid in all_task_ids()]
    else:
        jobs = [(args.task_id, None)]

    total_errors = 0
    for label, explicit_dir in jobs:
        errors = validate_task(label, tdir=explicit_dir)
        if errors:
            eprint(f"FAIL {label}: {len(errors)} error(s)")
            for e in errors:
                eprint(f"  - {e}")
        else:
            print(f"OK {label}")
        total_errors += len(errors)

    if total_errors:
        eprint(f"\n{total_errors} error(s) across {len(jobs)} task(s).")
        return 1
    print(f"\nOK: {len(jobs)} task(s) validated, 0 errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
