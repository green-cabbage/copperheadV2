"""Shared helpers for .agent-system/scripts/*.py.

Standard library only, by design (see SYSTEM.md / README.md) — these
utilities must run under a plain `python3` without the repository's Pixi
environment being active, so a task can be inspected/advanced even when no
analysis dependencies are installed.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SYSTEM_DIR = Path(__file__).resolve().parent.parent
TASKS_DIR = SYSTEM_DIR / "tasks"
SCHEMAS_DIR = SYSTEM_DIR / "schemas"
REPO_ROOT = SYSTEM_DIR.parent

TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$|^[a-z0-9]$")

ITERATION_DIR_RE = re.compile(r"^(\d{3})$")


class TaskError(Exception):
    """Raised for user-facing task/state errors (invalid id, bad transition, ...)."""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_safe_task_id(task_id: str) -> bool:
    """Reject anything that isn't a plain lowercase-kebab identifier.

    In particular rejects path separators, '..', leading dots, spaces, and
    empty strings so task_id can never be used to escape TASKS_DIR.
    """
    if not task_id or "/" in task_id or "\\" in task_id or ".." in task_id:
        return False
    return bool(TASK_ID_RE.match(task_id))


def task_dir(task_id: str) -> Path:
    if not is_safe_task_id(task_id):
        raise TaskError(f"unsafe or malformed task id: {task_id!r}")
    return TASKS_DIR / task_id


def iteration_dir_name(n: int) -> str:
    if n < 1:
        raise TaskError(f"iteration numbers start at 1, got {n}")
    return f"{n:03d}"


def iteration_dir(task_id: str, n: int) -> Path:
    return task_dir(task_id) / "iterations" / iteration_dir_name(n)


def list_iterations(task_id: str) -> list[int]:
    """Return the sorted list of existing iteration numbers for a task."""
    iterations_root = task_dir(task_id) / "iterations"
    if not iterations_root.is_dir():
        return []
    found = []
    for child in iterations_root.iterdir():
        if child.is_dir():
            m = ITERATION_DIR_RE.match(child.name)
            if m:
                found.append(int(m.group(1)))
    return sorted(found)


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise TaskError(f"missing file: {path}")
    except json.JSONDecodeError as exc:
        raise TaskError(f"malformed JSON in {path}: {exc}")


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
        fh.write("\n")


def load_task(task_id: str) -> dict:
    return load_json(task_dir(task_id) / "task.json")


def save_task(task_id: str, data: dict) -> None:
    save_json(task_dir(task_id) / "task.json", data)


def load_state(task_id: str) -> dict:
    return load_json(task_dir(task_id) / "current-state.json")


def save_state(task_id: str, data: dict) -> None:
    save_json(task_dir(task_id) / "current-state.json", data)


def load_schema(name: str) -> dict:
    return load_json(SCHEMAS_DIR / f"{name}.schema.json")


def git(*args: str, cwd: Path | None = None) -> str | None:
    """Best-effort git call; returns None instead of raising on any failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd or REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_metadata() -> dict:
    return {
        "repository": REPO_ROOT.name,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "starting_commit": git("rev-parse", "HEAD"),
    }


def git_environment() -> dict:
    dirty = git("status", "--porcelain")
    commit = git("rev-parse", "HEAD")
    return {
        "operating_system": sys.platform,
        "python_version": sys.version.split()[0],
        "working_directory": str(REPO_ROOT),
        "git_commit": commit,
        "dirty_worktree": None if commit is None else bool(dirty),
    }


def eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)
