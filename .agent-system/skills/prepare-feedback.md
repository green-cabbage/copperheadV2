# Skill: prepare-feedback

Sub-procedure of `review-result.md`, covering `feedback.json` specifically.
Also read `agents/reviewer.md` § "Finding format" — this skill is the
mechanical checklist for it, that document is the rule.

## Steps

1. Start from the prior iteration's `feedback.json` if one exists (do not
   start from a blank template — you must carry findings forward).
2. For each finding already present:
   - If it was addressed this iteration, update `status` per the actual
     evidence (verify — don't take the generator-report's claim at face
     value).
   - If it wasn't addressed and is still valid, leave `status: open`.
   - Never delete it, never change its `finding_id`.
3. Add new findings discovered this iteration with fresh IDs, continuing
   the existing numbering (don't restart at `REV-001` if `REV-004` already
   exists elsewhere in the task).
4. Fill `required_actions` (task-level, distinct from per-finding
   `required_action`) with anything that spans multiple findings or blocks
   the whole task.
5. Fill `unanswered_questions` for anything you need clarified before you
   can finish reviewing — this alone can justify
   `decision: human_review_required` if the questions are load-bearing.
6. Set `human_approval_required: true` if any open finding needs a
   decision only a human can make (see `WORKFLOW.md` § "Early stop by the
   Reviewer").
7. Set `confidence` honestly — `low` if you couldn't fully verify the
   result (e.g. couldn't run the artifact yourself, evidence was thin),
   not just when you found bugs.
8. Validate the file against `schemas/feedback.schema.json` before
   finishing (`validate_artifacts.py` does this automatically, but check
   the diff yourself too — a schema-valid file can still say the wrong
   thing).
