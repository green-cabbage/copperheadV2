# Skill: review-result

Used by the Reviewer role during `reviewing`.

## Steps

1. Run `inspect-task.md` first.
2. Read `iterations/<NNN>/selection-doc.md` and `doc-report.json` first, before
   the generated code. You need to know what the Code Generator was told before
   you can judge what it produced — and unlike it, you are not blinded, so you
   can check `doc-report.json` `sources[]` against the real repository to see
   whether the document extracted the physics correctly in the first place.
3. Read the actual diff for this iteration (compare against the prior
   iteration or the task's `metadata.starting_commit`, not just the
   `generator-report.md` summary). Read the `generator-report.md`
   *Documentation gaps* section carefully — the Code Generator recording "the
   document didn't say" is direct evidence about the document's sufficiency,
   and it is the signal this loop exists to produce.
4. Read `iterations/<NNN>/run-report.json` in full, including the raw
   captured stdout/stderr summaries per command, not only the top-level
   `summary` counts.
5. Open the actual generated artifacts referenced in the run report where
   practical (a result file, a plot, generated data) rather than trusting
   the report's description of them.
6. Check each item in `task.json.acceptance_criteria` individually against
   what you observed — not against what the Generator claims.
7. For every defect, decide whether the document or the implementation is at
   fault, and set `target_role` accordingly
   (`agents/reviewer.md` § "Routing"). Where you route to `code-generator`,
   cite the `SEL-NNN` id proving the document was right — without that
   citation it is a documentation finding.
8. Re-check any finding the Documentation Generator listed in
   `doc-report.json` `misrouted_findings[]`: confirm the claim against the
   document, then either re-route it to `code-generator` or send it back with
   the deficient passage quoted. A finding that has bounced twice goes to
   `human_review_required` instead.
9. Carry forward every prior iteration's findings; update each `open` one
   to its new status per `agents/reviewer.md` § "Finding format". Add new
   findings with fresh, never-reused IDs.
10. Decide: `approved` only if no `blocker`/`major` finding is currently
    `open`. Otherwise pick the decision that matches
    `agents/reviewer.md` § "Decision".
11. Write `iterations/<NNN>/review-report.md` (narrative reasoning) and
    `iterations/<NNN>/feedback.json` (structured, matching
    `templates/feedback.json`). State the routing tally in the narrative: how
    many open findings point at the document, how many at the code, and which
    loop-back edge that implies.
12. Update `current-state.json` with the decision. If `approved`, also
    trigger `close-task.md`. If `changes_requested`, leave status there —
    `advance_task.py --to documenting` (or `--to implementing`, when every
    open finding is routed to the Code Generator) is what actually opens the
    next iteration. On the `implementing` edge, copy the current iteration's
    `selection-doc.md` and `doc-report.json` into the new iteration directory
    unchanged, so each iteration stays self-contained.
