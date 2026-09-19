Role: Reviewer

# Review report — iteration 1

## What I checked independently

The run report shows `1 passed`, but the test it ran only checks
`normalize_whitespace("a  b") == "a b"` — a single double-space case. That
does not exercise two of the three acceptance criteria (tabs, newlines).
Rather than trust the passing test, I read `src/text_utils.py` directly
and ran the actual acceptance-criteria case myself:

```
>>> normalize_whitespace('  a\t\tb\n\nc   d  ')
'a\t\tb\n\nc  d'
```

Expected per the acceptance criteria: `'a b c d'`.

## Findings

- The implementation is `text.strip().replace("  ", " ")`. `str.replace`
  only matches the literal two-space substring — it does not touch tabs
  (`\t`) or newlines (`\n`) at all, and it only removes one two-space
  occurrence per match, so three-or-more consecutive spaces are not fully
  collapsed (`"   "` → `"  "`, not `" "`).
- The test written for this iteration is too narrow to have caught this —
  it happens to pass against the buggy implementation. A passing test is
  not sufficient evidence of correctness; the acceptance criteria are the
  bar, and this implementation does not meet two of the three.

## Acceptance criteria checklist

See `review-report.json`.

## Decision

`changes_requested` — see `feedback.json` REV-001 (blocker).
