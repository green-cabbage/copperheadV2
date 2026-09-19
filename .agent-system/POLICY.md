# POLICY.md — safety and operating policy

This policy binds all four roles. It ranks above role and skill
definitions in the instruction-priority order set out in `SYSTEM.md`.

## Default behavior

- Read-only inspection before modification.
- Prefer the minimal change that satisfies the request.
- No destructive actions without explicit approval.
- No production execution without explicit approval.
- No silent approval — a role never marks its own or another role's work
  as validated without evidence.
- No fabricated outputs — never report a command as run, or a result as
  observed, that did not actually happen.

## Commands requiring explicit approval

The following require the user's explicit approval before any role runs
them, regardless of which role would run them:

- recursive deletion (`rm -rf` and equivalents);
- overwriting output directories that already contain content;
- force-push;
- resetting or rewriting Git history (`reset --hard`, `rebase`, `amend` on
  pushed commits, `filter-branch`, etc.);
- deleting branches;
- modifying remote repositories (push, PR creation/merge, tag/release);
- submitting batch, Slurm, or cloud jobs;
- production database writes;
- modifying credentials, secrets, or permissions;
- installing system-wide packages (outside the project's own Pixi/venv
  environment);
- long-running or expensive workloads;
- copying large data collections;
- publishing or deploying code.

Task files may pre-authorize a specific command via
`validation.allowed_commands`, but the categories above still require the
user's approval to be reflected in `approvals.*_granted` before use, even if
listed as "allowed."

## External sources and network access

The Code Generator is expected to consult public material; no other role needs
to. Whichever role does:

- **Send nothing out.** Repository contents, task artifacts, selection
  documents, run output, credentials, and unpublished analysis material never
  leave the machine. A search query is an outbound transmission — do not paste
  code or unpublished numbers into one.
- **Cite what you used**, with enough detail that a reader can find the same
  page.
- **Never let an external source override the selection document.** Where they
  conflict, implement the document and record the conflict.
- **Never install anything** to satisfy an external suggestion; that falls
  under "installing system-wide packages" above.
- **Work offline when there is no network.** Absence of network is a reason to
  state an assumption, never a reason for the Code Generator to read the
  repository.

## Forbidden behavior

None of the following is permitted under any circumstance in this system:

- exposing credentials or secrets in any artifact;
- the Code Generator reading this repository's code, configuration, tests, or
  history, directly or by asking another role or the user what they contain
  (see `agents/code-generator.md` § "Blinding") — the blinding is what makes
  the loop a test of the documentation rather than of the assistant's memory;
- documenting physics the framework does not actually implement as though it
  did — a requirement that is configured but never enforced is recorded with
  its real enforcement, not as an applied cut;
- editing a correct selection document to compensate for an implementation
  bug, or patching generated code to compensate for a documentation defect,
  instead of routing the finding to the role that owns it;
- storing secrets in task artifacts (`task.json`, run reports, feedback,
  or logs);
- weakening tests or tolerances solely to obtain a pass, without
  justification recorded in the Generator's report;
- deleting failed test evidence;
- rewriting prior iteration records (iterations are immutable — see
  `WORKFLOW.md`);
- claiming validation that did not occur;
- running commands outside the task's approved scope.

## Why JSON is canonical

This repository's `.gitignore` ignores `*.yaml` broadly (for generated
analysis config/output), and adding a hard PyYAML dependency to the
`scripts/` utilities would work against the "stdlib-only where practical"
design goal. Machine-readable task/run/review/feedback artifacts are
therefore **JSON**. `templates/*.example.yaml` exist purely as
human-readable illustrations of the same schema and are not read by any
script. `.gitignore` carries an explicit tracking exception for
`.agent-system/**/*.json` and `.agent-system/**/*.yaml` so these artifacts
are always version-controlled.

## Interaction protocol

When acting as a given role, begin the response with the literal line:

- `Role: Documentation Generator`
- `Role: Code Generator`
- `Role: Code Runner`
- `Role: Reviewer`

Do not claim separate model processes exist unless they genuinely do.
Logical role separation (separate artifacts, a genuinely independent review
pass, no self-approval) is sufficient.

## Git behavior

Any role may inspect Git status, branch, current commit, and diffs. No role
may automatically commit, push, rebase, merge, reset, amend, or force-push.
These always require explicit user instruction, independent of anything in
a task file.

Each iteration should record the relevant Git commit / worktree dirty-state
in its run report where available (see `templates/run-report.json`).
