# Skill: run-validation

Used by the Code Runner role during `running`.

## Steps

1. Run `inspect-task.md` first.
2. Read `iterations/<NNN>/proposed-commands.json`. Do not run anything not
   listed there; do not substitute a different command.
3. Cross-check each command's category against `POLICY.md` § "Commands
   requiring explicit approval". If a proposed command falls in one of
   those categories and the corresponding `approvals.*_granted` is not
   `true` in `task.json`, skip it, classify it `blocked` in the run
   report, and explain why — do not run it anyway.
4. Execute the remaining commands in the order given, using the
   repository's Pixi environment (`pixi run -e <environment> ...`) unless
   the command is explicitly a stdlib-only `.agent-system/scripts/*.py`
   utility meant to run under plain `python3`.
5. For each command, record exactly what `agents/code-runner.md` §
   "Required output" specifies — including a `classification` — as you go,
   not reconstructed from memory afterward.
6. If a command times out or the environment is unavailable, classify it
   accordingly and continue with the remaining commands rather than
   aborting the whole run, unless a later command genuinely depends on the
   failed one's output.
7. Write `iterations/<NNN>/run-report.json`.
8. Update `current-state.json`: status → `ready_for_review`.
