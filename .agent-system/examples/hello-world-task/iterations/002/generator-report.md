Role: Code Generator

# Implementation plan — iteration 2

## Request

Same as iteration 1. Responding to `feedback.json` REV-001 (blocker).

## Plan

Replace the `str.replace` approach with `re.sub(r"\s+", " ", text).strip()`
— `\s` matches any whitespace character (space, tab, newline, ...) and
`+` collapses any run of them uniformly, regardless of length or which
whitespace characters are mixed in. Expand the test to actually cover
tabs, newlines, multi-space runs, and the empty-string edge case.

## Files changed

- `src/text_utils.py` — implementation swapped to the regex approach.
- `tests/test_text_utils.py` — added 3 new test cases (tabs+newlines,
  leading/trailing whitespace, empty string), keeping the original
  double-space case.

## Assumptions

- None beyond what's already verified: re-ran the exact failing case from
  REV-001's evidence against the new implementation before writing this
  report (see "Expected behavior").

## Commands recommended for validation

See `proposed-commands.json`.

## Expected behavior

`normalize_whitespace('  a\t\tb\n\nc   d  ')` → `'a b c d'` (previously
`'a\t\tb\n\nc  d'`).

## Risks

- None identified for a pure stdlib regex substitution on a single
  function.

## Unresolved questions

- None.

## Feedback addressed

| Finding ID | This iteration's response |
|---|---|
| REV-001 | fixed — swapped to `re.sub(r"\s+", " ", text).strip()`; added `test_tabs_and_newlines_collapse`, `test_leading_and_trailing_whitespace_stripped`, `test_empty_string`. |
