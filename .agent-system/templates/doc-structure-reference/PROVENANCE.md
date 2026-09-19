# Provenance

Verbatim copy of `.claude/skills/` from an upstream repository, kept here as the
**structural reference** for documentation produced by the Documentation
Generator (`agents/documentation-generator.md`).

| | |
|---|---|
| Source | `https://github.com/ram1123/copperheadV2/tree/Week_June10/.claude/skills` |
| Branch | `Week_June10` |
| Commit | `88d51bf0dfef82c65b6d8c28ee43d4b86e9e1d45` (2026-09-11) |
| Copied | 2026-09-19 |

## What to take from it, and what not to

Take the **structure**: a `SKILL.md` entry point that says when the document
applies, what context must be established before using it, and how findings are
classified; plus one `references/<object>.md` per physics object, each carrying
a numbered-source table, classification tags, per-cut tables, a review
checklist, an evidence summary, and a "last verified" date.

Do **not** take the *content style* wholesale. These files are written for an
agent that can read the repository: they cite `src/copperhead_processor.py`
line numbers, `configs/parameters/*.yaml` keys, and framework specifics. The
Documentation Generator's output must be free of those — see
`agents/documentation-generator.md` § "Package- and language-agnostic" for what
that means in practice, and § "Provenance without leakage" for where the code
pointers go instead.

These files are reference material. Nothing reads them programmatically; no
script parses them; they are not inputs to any task.
