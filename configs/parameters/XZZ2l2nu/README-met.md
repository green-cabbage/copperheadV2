# XZZ PuppiMET XY correction

The processor applies `met.yaml` to XZZ2l2nu, NanoAODv15, for every Run 2 year that has a payload
configured; the correction is gated on the payload being present, not on a year literal. Each year
points at its own official JME POG `met.json.gz`. The published routine
(`XYMETCorrection_withUL17andUL18andUL16.h`) covers UL16, UL17 and UL18.

The 2017 reference skim applies the same correction and we reproduce it to 1e-5. The 2018 reference
MC files created before 2026-05-10 do not apply it, while 2018 reference data does; that is stale
reference output, not a reason to drop the correction for 2018. See task iteration 034,
`reference-met-xy-survey.json`.

The validation below was performed on 2017.
The input payload is verified against its configured SHA-256 before loading.
Raw PuppiMET is retained alongside the corrected values.

The published UL XY routine translates the MET x/y components using vertex
count and era. It caps the vertex count at 100 and has no upper MET-pT limit.
The pinned JSON places the same formula inside a `[0, 6500)` input bin.
`src/corrections/xzz_met_xy.py` extends only that bin's upper edge to infinity
in memory, after checking the expected schema. It retains the original file,
checksum, formula, coefficients, run selection and phi bounds. It rejects
nonfinite inputs and outputs. MET is neither clamped nor used to introduce an
additional event veto.

Task iterations 028–029 validate the domain extension against the published
C++ implementation for MC and all five 2017 data eras, including values at and
above 6500 GeV. In-range outputs are bitwise unchanged. The triggering boosted
DY event is run 1, luminosity block 612, event 2096105: raw corrected-input
PuppiMET is 19001.66015625 GeV. The reference includes this same input outlier.
Its corrected float32 MET differs from our float64 result by about 0.000824
GeV; this remains a mismatch under the required absolute 1e-4 comparison.

Published routine:
https://lathomas.web.cern.ch/METStuff/XYCorrections/XYMETCorrection_withUL17andUL18andUL16.h
