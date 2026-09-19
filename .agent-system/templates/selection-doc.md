# <Analysis / selection name> — Physics Selection

<!--
TEMPLATE. Produced by the Documentation Generator; read by the Code Generator,
which cannot see this repository. Structure follows
templates/doc-structure-reference/ (see its PROVENANCE.md). Rules that bind this
document: agents/documentation-generator.md.

Two-tier layout, like the reference:
  - this file is the entry point (the reference's SKILL.md analogue);
  - for a large selection, per-object sections may move to
    selection-doc-references/<object>.md and be listed in § "Sections" below.
    Small selections stay in this one file.

Delete every comment block before writing the real document.

NO repository paths, line numbers, function names, config keys, package names
(numpy / awkward / coffea / HiggsDNA / uproot / correctionlib / CMSSW), or
language syntax anywhere in this file. Cite sources by id only (S1, C3) —
they resolve through doc-report.json.
-->

| | |
|---|---|
| Covers | <which objects and selection stages> |
| Eras | <eras this document is valid for> |
| Data / simulation | <both, or which> |
| NanoAOD campaign | <campaign and version> |
| Iteration | <NNN> |

**Deliberately out of scope:** <what a reader must not expect to find here>

---

## Required context

Establish before using this document; each of these can change the answer:

1. run period (Run 2 / Run 3 / Phase-2);
2. exact era;
3. data or simulation;
4. NanoAOD campaign and version;
5. intended working point.

<!-- If a requirement below depends on one of these and the value is unknown,
     the requirement is tagged [Verify], not guessed. -->

## Classification tags

- **[Official]** — a POG or CMS-wide recommendation.
- **[Analysis-specific]** — a choice this analysis made; not an official number.
- **[Verify]** — could not be established from the source material; must be
  confirmed by a physicist before it is relied on.

## Sections

<!-- List the sections present. For the two-tier layout, link the per-object
     files here. Omit objects the analysis does not use — say so explicitly
     rather than leaving a silent gap. -->

- Electrons
- Muons
- Jets
- b tagging
- Missing transverse momentum
- Trigger
- Event selection
- Corrections and ordering

---

## <Object or stage name>

<!-- One section per object or selection stage. Repeat this block. -->

<Short prose statement of what this object is selected for in this analysis.>

| ID | Requirement | Value | Era | Class | Enforcement |
|---|---|---|---|---|---|
| SEL-001 | <quantity, in words, then the NanoAOD branch if the branch *is* the physics> | <threshold with units> | <eras, or "all"> | [Analysis-specific] | applied |
| SEL-002 | <...> | <...> | <...> | [Verify] | stored only |

<!--
Enforcement is load-bearing, not decoration. One of:
  applied              — the framework cuts on it
  computed, not applied — evaluated and recorded as a flag, but nothing filters on it
  stored only          — the input quantity is written out; no threshold is ever evaluated
  absent               — specified by a source, not implemented at all
A Code Generator reading "applied" for something the framework merely stores will
produce a selection the analysis does not actually perform.
-->

**Notes.** <Anything the table cannot carry: conditional applicability, known
disagreements between a source and the implementation, era-specific exceptions.>

---

## Corrections and ordering

<!-- Ordering is physics and is the first thing lost in a reimplementation.
     State it explicitly; never leave it implied by section order. -->

Corrections apply in this order:

1. <correction> — <what it changes, and on data, simulation, or both>
2. <correction> — <...>

**Ordering constraints that must hold:**

- <X must happen before Y, and why the result differs otherwise.>
- <Quantity Q is evaluated after correction C, so a cut on the uncorrected
  value is a different selection.>

<!--
Pseudocode is allowed here and anywhere else a procedure is genuinely clearer
than prose. Hard limit: 20 lines per block, checked mechanically by
scripts/validate_artifacts.py. No language syntax, no package calls, no variable
names copied from the source. The surrounding text must still state the rule —
a block is an illustration, never the only place a requirement appears.

    for each muon:
        if beam-spot-constrained momentum is available and its fit quality is acceptable:
            use the beam-spot-constrained momentum
        otherwise:
            use the default momentum
        apply the momentum scale correction
-->

---

## Event selection

<Requirements that apply to the event rather than to a single object: object
multiplicities, charge combinations, mass windows, vetoes. Same table format.>

---

## Open questions

<!-- Mirror doc-report.json open_questions[]. Phrase each so a physicist could
     answer it without reading code. -->

1. <question>

## Source ids

Cited by id throughout; the id table with locations lives in `doc-report.json`
`sources[]`. This document does not name repository paths — see
`agents/documentation-generator.md` § "Provenance without leakage".

## Last verified

- Source review: <date>
- Official recommendations: <date, or "pending">
