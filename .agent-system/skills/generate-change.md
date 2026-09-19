# Skill: generate-change

Used by the Code Generator role during `implementing`.

Read `agents/code-generator.md` § "Blinding" before step 1. This skill assumes
it: nowhere below are you to open a repository file.

## Steps

1. Run `inspect-task.md` first — but read only `task.json` and
   `current-state.json` from the task directory, not `doc-report.json`.
2. Read `iterations/<NNN>/selection-doc.md` in full, including any
   `selection-doc-references/*.md` it lists. This is your entire picture of the
   physics; read it as a specification, not as background.
3. Build the requirement checklist: every `SEL-NNN` id in the document, with its
   stated enforcement (`applied`, `computed, not applied`, `stored only`,
   `absent`). You will be answering to this list in the report, and the
   Reviewer checks it item by item.
4. Consult external sources where an implementation detail is genuinely open —
   an algorithm, a library API, a public physics reference. Note each for
   citation. Never let one override the document
   (`agents/code-generator.md` § "External sources").
5. Implement into `iterations/<NNN>/generated_doc/`, one module per documented
   section, named after that section.
6. Record every ambiguity, silence, or contradiction you hit in the document as
   you hit it — not reconstructed at the end. These become the report's
   *Documentation gaps*, and they are the loop's main product.
7. If this iteration is responding to `feedback.json`, address every finding
   with `target_role: code-generator` and `status: open`. For each, record in
   the report which finding it addresses and how — or why it is being
   rejected, deferred, or is really a documentation defect (see
   `agents/reviewer.md` § "Finding format" for the allowed statuses).
8. Write `iterations/<NNN>/generator-report.md` and
   `iterations/<NNN>/proposed-commands.json` per
   `agents/code-generator.md` § "Required outputs". Propose commands that
   exercise the acceptance criteria, not merely commands that prove the code
   imports.
9. Update `current-state.json`: status → `ready_to_run`.

## Note on the approval gate

The `approvals.implementation_required` gate now sits between
`ready_for_implementation` and `implementing` — it approves the *selection
document* before code is written from it, which is the cheaper place to catch a
wrong specification. By the time this skill runs, that gate has already been
passed or was not required.
