# Skill: generate-documentation

Used by the Documentation Generator role during `documenting`.

## Steps

1. Run `inspect-task.md` first.
2. Read `templates/doc-structure-reference/` — at minimum its `PROVENANCE.md`
   and `cms-object-guidelines/SKILL.md`, plus one `references/*.md` for the
   object you are about to document. You are copying its *structure*, not its
   content style; it cites code, your output must not.
3. Read the actual source files named or implied by the request. Read them
   directly — the whole point of this role is that nothing downstream can.
4. For each selection requirement you find, establish four things before
   writing it down:
   - the **quantity** being cut on, in physics terms;
   - the **threshold or working point**, with units;
   - the **era applicability** — all eras, or specific ones;
   - the **enforcement** — is it applied, computed into a flag but never
     filtered on, merely stored as an output column, or absent entirely?
     Check by following what actually consumes the value, not by trusting that
     a configured threshold is a threshold in force.
5. Assign each requirement a stable `SEL-NNN` id, continuing the numbering from
   prior iterations (don't restart at `SEL-001` if `SEL-014` already exists in
   this task).
6. Write `iterations/<NNN>/selection-doc.md` from `templates/selection-doc.md`.
   Before moving on, re-read it for leakage: any file path, line number,
   function or variable name, config key, package name, or line of Python/C++
   is a defect — strip it and put the provenance in `doc-report.json` instead.
7. Check every pseudocode block against the 20-line budget
   (`agents/documentation-generator.md` § "Pseudocode budget").
8. Write `iterations/<NNN>/doc-report.json` per `templates/doc-report.json`:
   the `sources[]` table, one `requirements[]` entry per `SEL-NNN` id in the
   document, `open_questions[]`, and — if this iteration is answering feedback
   — `responses[]` and `misrouted_findings[]`.
9. If this iteration is responding to `feedback.json`, address every finding
   with `target_role: documentation-generator` and `status: open`. For each,
   either correct the document and record it in `responses[]`, or — if the
   document was already right and the defect is downstream — record it in
   `misrouted_findings[]` and leave the document alone. Do not edit a correct
   document to compensate for a Code Generator bug.
10. Run `python .agent-system/scripts/validate_artifacts.py --task-id <id>` and
    fix anything it reports.
11. Update `current-state.json`: status → `ready_for_implementation`.

## Self-check before handing off

- Could a physicist who has never seen this repository implement the selection
  from this document alone, in a language of their choosing?
- Does every requirement state its enforcement, not just its threshold?
- Is every ordering constraint explicit?
- Is everything uncertain tagged `[Verify]` rather than asserted?
- Does every `SEL-NNN` in the document appear in `doc-report.json`, and vice
  versa?
