# Skill: close-task

Used once the Reviewer's decision for the latest iteration is `approved`,
or once a human directs closure from `awaiting_human_input`/`blocked`.

## Steps

1. Confirm no `blocker`/`major` finding across any iteration is currently
   `open` (skip this check if closing from a human-directed stop rather
   than an approval).
2. Run `python .agent-system/scripts/summarize_task.py --task-id <id>` —
   it reads all iterations and produces the content for
   `final-summary.md`: changes made, commands run, reviewer decisions per
   iteration, any findings left `rejected_with_justification` or
   `deferred_with_approval` (these are not "unresolved" but should still be
   visible), and the final state.
3. Write/confirm `tasks/<task-id>/final-summary.md` from that output.
4. `python .agent-system/scripts/advance_task.py --task-id <id> --to closed`.
5. Do not delete or modify any `iterations/<NNN>/` content as part of
   closing — closure only adds `final-summary.md` and updates
   `current-state.json`.
