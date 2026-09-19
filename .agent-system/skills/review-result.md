# Skill: review-result

Used by the Reviewer role during `reviewing`.

## Steps

1. Run `inspect-task.md` first.
2. Read the actual diff for this iteration (compare against the prior
   iteration or the task's `metadata.starting_commit`, not just the
   `generator-report.md` summary).
3. Read `iterations/<NNN>/run-report.json` in full, including the raw
   captured stdout/stderr summaries per command, not only the top-level
   `summary` counts.
4. Open the actual generated artifacts referenced in the run report where
   practical (a result file, a plot, generated data) rather than trusting
   the report's description of them.
5. Check each item in `task.json.acceptance_criteria` individually against
   what you observed — not against what the Generator claims.
6. Carry forward every prior iteration's findings; update each `open` one
   to its new status per `agents/reviewer.md` § "Finding format". Add new
   findings with fresh, never-reused IDs.
7. Decide: `approved` only if no `blocker`/`major` finding is currently
   `open`. Otherwise pick the decision that matches
   `agents/reviewer.md` § "Decision".
8. Write `iterations/<NNN>/review-report.md` (narrative reasoning) and
   `iterations/<NNN>/feedback.json` (structured, matching
   `templates/feedback.json`).
9. Update `current-state.json` with the decision. If `approved`, also
   trigger `close-task.md`. If `changes_requested`, leave status there —
   `advance_task.py --to implementing` is what actually opens the next
   iteration.
