# SYSTEM.md — canonical entry point

This file is the authoritative definition of the agent workflow in this
repository. It is vendor-agnostic: it must work with any repository-aware
coding assistant (Claude Code, OpenAI Codex, Cursor, Gemini CLI, GitHub
Copilot, and similar future tools). It does not reference proprietary
commands, subagent APIs, slash commands, hooks, or model-specific memory
features.

If you are a coding assistant reading this: read this file first, then
`POLICY.md`, then `WORKFLOW.md`, then the relevant files under `agents/` and
`skills/`.

## The three roles

1. **Code Generator** — understands the request, inspects the repository,
   reads prior feedback, proposes and implements a minimal change. See
   `agents/code-generator.md`.
2. **Code Runner** — executes approved validation commands and records exact,
   unembellished evidence. See `agents/code-runner.md`.
3. **Reviewer** — independently evaluates the change against the request and
   the run evidence, and produces structured feedback. See
   `agents/reviewer.md`.

## The workflow

```
User request
    ↓
Code Generator
    ↓
Code Runner
    ↓
Reviewer
    ↓
Feedback artifact
    ↓
Code Generator  (loop)
```

The loop continues until one of:

- the Reviewer decision is `approved`;
- `current_iteration` reaches `max_iterations`;
- execution is `blocked`;
- the task state becomes `awaiting_human_input`;
- an unrecoverable error occurs;
- the request is judged unsafe or too ambiguous to proceed.

Full state machine and transition rules: `WORKFLOW.md`.

## Instruction priority

When instructions conflict, resolve in this order — and if a conflict is
found, **report it rather than silently choosing**:

1. Current user instruction
2. Current repository code and configuration
3. `.agent-system/POLICY.md`
4. `.agent-system/SYSTEM.md` (this file)
5. Relevant role definition (`agents/*.md`)
6. Relevant skill (`skills/*.md`)
7. Current task artifacts (`tasks/<task-id>/...`)
8. Historical task records

## Artifact locations

| What | Where |
|---|---|
| Role definitions | `.agent-system/agents/` |
| Step-by-step skills | `.agent-system/skills/` |
| Canonical templates (JSON) + illustrative examples (YAML) | `.agent-system/templates/` |
| JSON Schemas for the above | `.agent-system/schemas/` |
| Live task directories | `.agent-system/tasks/<task-id>/` |
| Utility scripts | `.agent-system/scripts/` |
| Worked example of the full loop | `.agent-system/examples/hello-world-task/` |

**Canonical machine-readable format is JSON**, not YAML. See
`POLICY.md` § "Why JSON is canonical" for the reason. `templates/` also
ships `*.example.yaml` files purely as human-readable illustrations of the
same schema — they are documentation, not inputs to any script.

## Task states

`created → planning → awaiting_implementation_approval → implementing →
ready_to_run → running → ready_for_review → reviewing → {approved |
changes_requested | awaiting_human_input | blocked}`

`changes_requested → implementing` (loop back)
`approved → closed`

Full transition table and guard conditions: `WORKFLOW.md`.

## Iteration limits

Default `max_iterations: 3`, overridable per-task in `task.json`. When the
limit is reached, the system stops automatically, preserves all artifacts,
summarizes unresolved findings, and sets state to `awaiting_human_input`. It
does not keep editing past the limit.

## Safety and approval rules

See `POLICY.md` for the full list of actions requiring explicit approval and
the list of forbidden behaviors. In summary: read-only inspection before
modification, minimal changes, no destructive actions, no production
execution, no silent self-approval, no fabricated results.

## Completion criteria

A task is `closed` only when the Reviewer decision is `approved` and
`final-summary.md` has been written for that task. Reaching
`awaiting_human_input` or `blocked` is a valid stopping point but not
completion — a human must act before the loop resumes.

## Running without separate model instances

Some assistants cannot spawn genuinely separate model processes. In that
case, a single assistant instance performs the three roles **sequentially**,
but must still:

- clearly announce the active role at the start of its turn (see
  `POLICY.md` § "Interaction protocol");
- write to the separate, role-specific artifacts described above rather
  than blending them into one file;
- never let the Code Generator approve its own implementation — the
  Reviewer pass must be a genuinely fresh read of the code and run
  evidence, not a restatement of the Generator's claims;
- preserve logical separation between roles even though the underlying
  process is the same.
