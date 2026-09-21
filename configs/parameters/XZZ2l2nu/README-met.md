# XZZ PuppiMET XY correction

For XZZ2l2nu NanoAODv15, `met.yaml` enables the MET correction for each Run 2 year with a configured official JME POG correction file (`met.json.gz`), once it is turned on in the switches.yaml.

The [published UL16–18 routine](https://lathomas.web.cern.ch/METStuff/XYCorrections/XYMETCorrection_withUL17andUL18andUL16.h) translates MET x/y by vertex count and era, caps vertices at 100 and imposes no upper MET-pT limit. After schema checks, `src/corrections/xzz_met_xy.py` extends the pinned JSON's `[0, 6500)` bin to infinity in memory. File/checksum, formula, coefficients, run selection and phi bounds stay unchanged. Nonfinite inputs/outputs are rejected; MET is neither clamped nor an added event veto.

- **NPV > 100:** the MET XY correction uses NPV = 100. The original vertex count is unchanged, and this cap does not reject the event.
- **Prescription checked:** the published UL16–18 correction routine linked above explicitly uses `if(npv>100) npv=100;`. We verified the cap against that prescription, rather than inferring it solely from reference output.


Evidence from task xzz-2l2nu-007:

- Iterations028–029 validated against published C++ for 2017 MC and all five data eras, including MET ≥ 6500 GeV; in-range results are bitwise unchanged.
- Boosted-DY event `run=1, luminosityBlock=612, event=2096105` has input PuppiMET 19001.66015625 GeV in both implementations. Reference float32 and our float64 corrected MET differ by ~0.000824 GeV, exceeding the absolute 1e-4 comparison tolerance.
- Iteration034 `reference-met-xy-survey.json`: the 2017 reference correction agrees to 1e-5; 2018 reference data applies it, but reference MC produced before 2026-05-10 omits it. Keep the correction for 2018 despite stale reference MC.
