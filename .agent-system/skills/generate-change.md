# Skill: generate-change

Used by the Code Generator role during `implementing`.

## Steps

1. Run `inspect-task.md` first.
2. Read the relevant source files directly — don't rely on memory of the
   repo or on the task request's description of the code.
3. Draft the smallest implementation plan that satisfies
   `task.json.acceptance_criteria`, honoring `scope.include`/
   `scope.exclude` and `constraints`.
4. If this is iteration 1 and `approvals.implementation_required` is true
   and not yet granted: write the plan into
   `iterations/001/generator-report.md`, set status to
   `awaiting_implementation_approval`, and stop — do not write code yet.
5. Once authorized (or if approval isn't required), implement the change.
   Prefer editing existing files/conventions over introducing new patterns.
6. If this iteration is responding to `feedback.json` from the prior
   iteration, address every `open` `blocker`/`major` finding; for each,
   record in this iteration's `generator-report.md` which finding it
   addresses and how (or why it's being rejected/deferred instead — see
   `agents/reviewer.md` § "Finding format" for the allowed statuses).
7. Write `iterations/<NNN>/generator-report.md` and
   `iterations/<NNN>/proposed-commands.json` per
   `agents/code-generator.md` § "Required outputs".
8. Update `current-state.json`: status → `ready_to_run`.
