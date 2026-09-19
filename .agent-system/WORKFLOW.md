# WORKFLOW.md — task state machine

## States

| State | Meaning |
|---|---|
| `created` | Task directory and `task.json` exist; nothing else has happened. |
| `planning` | Documentation Generator is scoping the request and identifying what to read. |
| `documenting` | Documentation Generator is reading the framework and writing `selection-doc.md` + `doc-report.json`. |
| `ready_for_implementation` | Selection document complete and validated; the Code Generator has something to implement. |
| `awaiting_implementation_approval` | Document written; waiting on human approval before code is written from it (only if `approvals.implementation_required`). |
| `implementing` | Code Generator is writing code from the document, blind to the repository. |
| `ready_to_run` | Generator's report for this iteration is complete; commands proposed. |
| `running` | Code Runner is executing approved validation commands. |
| `ready_for_review` | Run report for this iteration is complete. |
| `reviewing` | Reviewer is evaluating code + run report. |
| `changes_requested` | Reviewer found blocking/major issues; loops back to `documenting`, or to `implementing` when every open finding is routed to the Code Generator. |
| `awaiting_human_input` | Ambiguity, repeated failure to converge, or a decision only a human can make. |
| `blocked` | Execution cannot proceed (missing environment, forbidden command needed, etc.). |
| `approved` | Reviewer approved the result. |
| `closed` | `final-summary.md` written; task complete. |

## Transitions

```
created
    → planning
    → documenting
    → ready_for_implementation
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
    → documenting                (increments current_iteration; the usual route)
    → implementing               (increments current_iteration; code-only findings)
    → awaiting_human_input       (only when current_iteration == max_iterations; see "Iteration limits")

approved
    → closed
```

## Which loop-back edge

`changes_requested` has two exits, and the Reviewer's routing decides which:

- **→ `documenting`** when any open `blocker`/`major` finding carries
  `target_role: documentation-generator`. The document is corrected first, and
  the Code Generator then re-implements from the corrected document. This is
  the default and the common case.
- **→ `implementing`** when every open finding carries
  `target_role: code-generator` — the document was right, only the code was
  wrong. The current iteration's document carries forward unchanged; copy it
  into the new iteration directory rather than regenerating it, so the
  iteration remains self-contained.

Both edges increment `current_iteration` and create the next iteration
directory. Taking the `implementing` edge when a documentation defect is open
wastes the iteration: the Code Generator cannot see the document's error and
will reproduce it.

`awaiting_human_input` and `blocked` are terminal until a human acts; there
is no automatic transition out of them. Once the human responds (approves,
clarifies, or authorizes a specific forbidden-category command), the task
resumes from the state it was in when it paused.

## Guard conditions

- `documenting → ready_for_implementation` requires `selection-doc.md` and
  `doc-report.json` to exist for the current iteration and to be well-formed
  (validated by `validate_artifacts.py`, which also enforces the pseudocode
  budget and checks that every `SEL-NNN` id is both indexed and cited).
- `ready_for_implementation → implementing` requires
  `approvals.implementation_granted` when `approvals.implementation_required`
  is set. The gate is on the *document*, not on the code.
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
- `changes_requested → documenting` and `changes_requested → implementing`
  each increment `current_iteration` by exactly 1 and create the next
  `iterations/<NNN>/` directory. Neither reuses or overwrites a prior
  iteration directory.
- Any transition is rejected if `current_iteration >= max_iterations` and
  the target state is not `awaiting_human_input`, `blocked`, `approved`, or
  `closed` — see "Iteration limits" below.

`.agent-system/scripts/advance_task.py` enforces this table in code; it
rejects any transition not listed above.

## Iteration limits

Default `max_iterations: 3` (overridable per-task in `task.json`). When
`current_iteration` reaches `max_iterations` and the Reviewer has not
approved:

1. Stop automatically — do not start another `documenting` or `implementing`
   phase.
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
- a finding has bounced twice between the Documentation Generator and the
  Code Generator without either accepting it (see `agents/reviewer.md`
  § "Routing") — two roles disagreeing twice about whose defect it is, is not
  something either can settle;
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
