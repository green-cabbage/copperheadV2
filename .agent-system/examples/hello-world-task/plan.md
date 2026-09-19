# Plan — hello-world-task

Add `normalize_whitespace(text)` to
`.agent-system/examples/hello-world-task/src/text_utils.py`: collapse any
run of whitespace (spaces, tabs, newlines) into a single space, strip
leading/trailing whitespace. Cover it with a unit test that actually
exercises tabs/newlines/multi-space input, not just a single easy case.

Trivial, isolated, standard-library-only — no reason to touch anything
outside this example directory.

Status as of iteration 2: implemented with `re.sub(r"\s+", " ", text).strip()`,
approved and closed. See `final-summary.md` for the full history.
