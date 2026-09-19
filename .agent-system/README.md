# .agent-system

A minimal, vendor-agnostic workflow for running scoped development tasks in
this repository with an auditable, resumable paper trail. Works with any
repository-aware coding assistant (Claude Code, Codex, Cursor, Gemini CLI,
Copilot, ...) — nothing here depends on a specific vendor's subagent API,
slash commands, hooks, or memory feature.

**Start at [`SYSTEM.md`](SYSTEM.md)** — it is the canonical entry point and
defines roles, workflow, artifact locations, task states, and completion
criteria. `POLICY.md` and `WORKFLOW.md` are the two documents it delegates
to for safety rules and the detailed state machine, respectively. This
README is orientation; those three files are the source of truth.

## The four roles

1. **Documentation Generator** (`agents/documentation-generator.md`) — reads
   the code framework and writes down its selection as *physics*: thresholds,
   working points, ordering, and whether each requirement is actually enforced
   — stated independently of any code package (numpy, awkward, coffea,
   HiggsDNA) or language (C++, Python).
2. **Code Generator** (`agents/code-generator.md`) — implements that document.
   **Blinded to this repository**: the document and public external sources
   are all it gets.
3. **Code Runner** (`agents/code-runner.md`) — executes approved validation
   commands, records exact evidence, never edits source or judges
   correctness.
4. **Reviewer** (`agents/reviewer.md`) — independently evaluates the result
   against the request, the document, and the run evidence, files structured
   findings, and routes each to the role that can fix it.

## The feedback loop

```
User request → Documentation Generator → Code Generator → Code Runner → Reviewer
                        ↑                      ↑                            │
                        └── document defect ───┴──── code defect ───────────┘
```

Feedback returns to the **Documentation Generator** by default: the document is
the specification the Code Generator worked from, so a defect in the result is
usually a defect in the document. It goes back to the Code Generator only when
the document stated the requirement correctly and the implementation still got
it wrong (`agents/reviewer.md` § "Routing").

The loop stops when the Reviewer approves, `max_iterations` is reached,
execution is blocked, human input is required, or the request turns out to
be unsafe/too ambiguous to proceed. See `WORKFLOW.md` for the full state
machine.

## Why blind the Code Generator?

Because then the document is the only thing carrying the analysis into the
implementation. Anything the document fails to say shows up as a defect in the
generated code, instead of being quietly repaired by a generator that could go
read the original. The loop measures the documentation, which is the reason to
run it.

## Creating a task

```bash
python .agent-system/scripts/create_task.py \
  --task-id my-task \
  --title "Short title" \
  --request-file request.txt
```

This creates `.agent-system/tasks/<task-id>/` with `task.json`,
`current-state.json`, and `iterations/001/`. It prints the created path.

## Running one iteration

There is no single "run" command — the roles are prompts for a coding
assistant, not an autonomous script. A typical iteration:

1. Assistant reads `SYSTEM.md`, `POLICY.md`, the task's `task.json`, and
   (if present) the latest `feedback.json`.
2. Acting as **Documentation Generator** (see
   `agents/documentation-generator.md` + `skills/generate-documentation.md`):
   reads the framework, writes `selection-doc.md` and `doc-report.json`.
3. `python .agent-system/scripts/advance_task.py --task-id <id> --to ready_for_implementation`
4. Human approves the document, if `approvals.implementation_required`; then
   `advance_task.py --task-id <id> --to implementing`.
5. Acting as **Code Generator** (see `agents/code-generator.md` +
   `skills/generate-change.md`), reading *only* the document: writes
   `generated_doc/`, `generator-report.md`, and `proposed-commands.json`.
6. `python .agent-system/scripts/advance_task.py --task-id <id> --to ready_to_run`
7. Acting as **Code Runner** (see `agents/code-runner.md` +
   `skills/run-validation.md`): executes the approved commands, writes
   `run-report.json`.
8. `python .agent-system/scripts/advance_task.py --task-id <id> --to ready_for_review`
9. Acting as **Reviewer** (see `agents/reviewer.md` + `skills/review-result.md`):
   writes `review-report.md` and `feedback.json`, each finding carrying a
   `target_role`.
10. `python .agent-system/scripts/advance_task.py --task-id <id> --to <approved|changes_requested|awaiting_human_input|blocked>`
11. If `changes_requested`, advance to `documenting` (step 2) or, when every
    open finding is routed to the Code Generator, straight to `implementing`
    (step 5). If `approved`, run `summarize_task.py` and advance to `closed`.

Validate artifacts at any point with:

```bash
python .agent-system/scripts/validate_artifacts.py --task-id <id>
```

## Switching between assistants mid-task

Because all durable state lives in `tasks/<task-id>/` as plain
JSON/Markdown, any assistant can pick a task back up: point it at
`SYSTEM.md` and the task directory, and it has everything it needs — no
vendor-specific memory or session state required. Start with Claude Code,
continue with Codex, finish review with a third tool: the artifacts don't
care.

## Resuming / stopping

- **Resume**: `current-state.json` in the task directory records the
  current iteration and status; an assistant reads it and continues from
  there.
- **Stop**: nothing auto-continues. The loop only advances when a role
  explicitly writes the next artifact and `advance_task.py` records the
  transition. To halt a task, simply don't take the next step — or set
  `status: blocked`/`awaiting_human_input` explicitly.

## Why old iterations are immutable

Each `iterations/<NNN>/` is a historical record: what was proposed, what
ran, what the Reviewer found. Overwriting it would destroy the audit trail
and make "did this regress since iteration 1" unanswerable. Corrections
happen in a new iteration; findings keep a stable ID (`REV-00N`) across
iterations so their fixed/rejected/deferred status is traceable.

## Approvals

`task.json` carries an `approvals` block
(`implementation_required/granted`, `execution_required/granted`,
`production_required/granted`). Roles check these before acting; the
state machine also blocks the relevant transition until the corresponding
`_granted` flag is `true`. See `POLICY.md` for the list of actions that
always need explicit approval regardless of what a task file says.

## Extending later

- New role: add `agents/<role>.md` + relevant `skills/*.md`, update
  `SYSTEM.md` role list and `WORKFLOW.md` if it changes state transitions.
- New artifact type: add a template + JSON Schema pair under
  `templates/`/`schemas/`, and teach `validate_artifacts.py` about it.
- New vendor: add a thin adapter (see the pattern in the root `CLAUDE.md`
  and `AGENTS.md`) — it should do nothing but point at `SYSTEM.md`.

Deliberately not included in v1: a vector database, web dashboard, message
queue, container orchestration, external database, or model API
integration. This is a filesystem-based system by design; add
infrastructure only when a real task proves the filesystem approach
insufficient.

## Documentation structure reference

`templates/doc-structure-reference/` is a verbatim copy of an upstream
`.claude/skills/` tree, kept as the structural model for what the
Documentation Generator produces: a `SKILL.md` entry point plus one
`references/<object>.md` per physics object, each with a numbered source table,
classification tags, per-cut tables, a review checklist, and an evidence
summary. Its `PROVENANCE.md` records where it came from and — importantly —
which parts to imitate and which not to: those files cite file paths and line
numbers, and the Documentation Generator's output must not.

## Worked example

`examples/hello-world-task/` predates the Documentation Generator and shows the
earlier three-role loop (Generator → Runner → Reviewer). Its mechanics —
iteration immutability, stable finding IDs, feedback carried forward — are
unchanged and still worth reading; just note that its findings all route to
`code-generator`, and that a four-role task begins with a `documenting` phase
it does not show.

It walks through a full two-iteration loop
(request → generator → runner → reviewer finds a bug → generator fixes it
→ runner reruns → reviewer approves → final summary) on a trivial
text-processing function, with stable finding IDs preserved across
iterations. Read it before creating your first real task.
