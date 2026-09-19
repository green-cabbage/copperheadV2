Role: Code Generator

# Implementation plan — iteration 1

## Request

Add `normalize_whitespace(text)`: collapse whitespace runs to a single
space, strip the ends. Include a unit test.

## Plan

Implement with `str.strip()` + `str.replace("  ", " ")` — simple, no new
dependency.

## Files changed

- `src/text_utils.py` — new file, `normalize_whitespace` function.
- `tests/test_text_utils.py` — new file, one test.

## Assumptions

- Assumed (not verified against the acceptance criteria at implementation
  time) that repeated `.replace("  ", " ")` behavior on a single call would
  be "good enough" for typical inputs. This turned out to be wrong — see
  iteration 2.

## Commands recommended for validation

See `proposed-commands.json`.

## Expected behavior

`normalize_whitespace("a  b")` → `"a b"`.

## Risks

- Did not check tab/newline handling before proposing this — the
  acceptance criteria explicitly mention tabs and newlines, and this
  implementation was not tested against them prior to review.

## Unresolved questions

- None raised at this iteration.
