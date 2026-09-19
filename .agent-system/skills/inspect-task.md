# Skill: inspect-task

Used at the start of any role's turn, before doing anything else.

## Steps

1. Read `.agent-system/SYSTEM.md`, `POLICY.md` (skip if already loaded this
   session).
2. Read `.agent-system/tasks/<task-id>/task.json` — request, scope,
   constraints, acceptance criteria, approvals, `max_iterations`.
3. Read `.agent-system/tasks/<task-id>/current-state.json` — current
   iteration number and status.
4. If `current_iteration > 1`, read the **latest** iteration's
   `feedback.json` (if it exists yet) and skim earlier iterations' finding
   IDs so you know what's already been raised.
5. Confirm the current status matches what your role is meant to act on
   (e.g. Code Generator acts on `implementing`/`changes_requested`, not
   `reviewing`). If it doesn't, stop and report the mismatch instead of
   proceeding.
6. Check `approvals.*` against what you're about to do; if something you
   need isn't granted, stop and request it rather than proceeding without
   it.
