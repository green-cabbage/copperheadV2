# WORKFLOW.md — task state machine

## States

| State | Meaning |
|---|---|
| `created` | Task directory and `task.json` exist; nothing else has happened. |
| `planning` | Code Generator is inspecting the repo and drafting an implementation plan. |
| `awaiting_implementation_approval` | Plan written; waiting on human approval to implement (only if `approvals.implementation_required`). |
| `implementing` | Code Generator is writing/modifying code. |
| `ready_to_run` | Generator's report for this iteration is complete; commands proposed. |
| `running` | Code Runner is executing approved validation commands. |
| `ready_for_review` | Run report for this iteration is complete. |
| `reviewing` | Reviewer is evaluating code + run report. |
| `changes_requested` | Reviewer found blocking/major issues; loops back to `implementing`. |
| `awaiting_human_input` | Ambiguity, repeated failure to converge, or a decision only a human can make. |
| `blocked` | Execution cannot proceed (missing environment, forbidden command needed, etc.). |
| `approved` | Reviewer approved the result. |
| `closed` | `final-summary.md` written; task complete. |

## Transitions

```
created
    → planning
    → awaiting_implementation_approval   (only if implementation approval required)
    → implementing
    → ready_to_run
    → running
    → ready_for_review
    → reviewing
        → approved
        → changes_requested
        → awaiting_human_input
        → blocked

changes_requested
    → implementing               (increments current_iteration)
    → awaiting_human_input       (only when current_iteration == max_iterations; see "Iteration limits")

approved
    → closed
```

`awaiting_human_input` and `blocked` are terminal until a human acts; there
is no automatic transition out of them. Once the human responds (approves,
clarifies, or authorizes a specific forbidden-category command), the task
resumes from the state it was in when it paused.

## Guard conditions

- `implementing → ready_to_run` requires the Generator's report for the
  current iteration to exist and be well-formed (validated by
  `validate_artifacts.py`).
- `running → ready_for_review` requires a run report for the current
  iteration to exist and be well-formed, and every command in it to have a
  recorded exit code or explicit timeout/blocked classification.
- `reviewing → approved` requires every finding in the current iteration's
  feedback with severity `blocker` or `major` to have status `fixed`,
  `rejected with justification`, or `deferred with approval` in a *later*
  iteration's response — i.e. you cannot approve while unresolved blockers
  are still open in the latest feedback.
- `changes_requested → implementing` increments `current_iteration` by
  exactly 1 and creates the next `iterations/<NNN>/` directory. It never
  reuses or overwrites a prior iteration directory.
- Any transition is rejected if `current_iteration >= max_iterations` and
  the target state is not `awaiting_human_input`, `blocked`, `approved`, or
  `closed` — see "Iteration limits" below.

`.agent-system/scripts/advance_task.py` enforces this table in code; it
rejects any transition not listed above.

## Iteration limits

Default `max_iterations: 3` (overridable per-task in `task.json`). When
`current_iteration` reaches `max_iterations` and the Reviewer has not
approved:

1. Stop automatically — do not start another `implementing` phase.
2. Preserve all existing iteration artifacts unchanged.
3. Write a summary of unresolved findings (into that iteration's
   `feedback.json` `summary` field if not already present, or as a note
   from `summarize_task.py`).
4. Set `status: awaiting_human_input`.
5. Do not continue editing until a human raises `max_iterations` or
   otherwise directs the next step.

## Early stop by the Reviewer

The Reviewer may set `decision: human_review_required` (independent of
iteration count) when:

- requirements are ambiguous;
- repeated changes are not resolving the issue (e.g. the same finding
  recurs across iterations);
- the environment cannot actually validate the implementation;
- a physics, security, legal, or production-impact decision requires a
  human;
- the requested behavior conflicts with `POLICY.md`.

## Immutability

Files under `iterations/<NNN>/` are never edited after that iteration
closes (i.e. once the next iteration's directory is created). Corrections
happen in a *new* iteration, referencing the finding ID from the old one.
`current-state.json` is the only file in a task directory that is
routinely overwritten — it just points at the current iteration and
status; it is not the record of history.
