# XZZ UnifiedParT b-tag event weights

Shared Stage1 enables central medium-working-point SFs for XZZ2l2nu 2018 NanoAODv15 MC with `switches.do_xzz_btag_sf`. HMuMu and data skip the correction. The legacy `do_btag_wgt` must be false when this switch is enabled.

`btag.yaml` selects `nearest` (the nominal convention used in the full validation) or `supported`. Both factors use the exact veto candidate jets, including untagged jets. A tagged jet contributes SF; an untagged jet contributes `(1-SF*eff)/(1-eff)`; multiply over jets. Empty events have factor one.

The checked-in `btag_efficiency_2018.json` contains the frozen candidate MC efficiencies and explicit assignments for the 26 validated background samples. New samples must receive a justified assignment in this file; unrecognized samples fail. The relative file path resolves against the repository root, including on Gateway workers.

The pinned BTV payload is defined in `src/corrections/xzz_btag_sf.py`. `supported` uses unity outside its domain; `nearest` clips only SF lookup coordinates to supported boundaries. Efficiencies always use the original jet pT. These are explicit user-approved conventions, not official extrapolation recommendations. The efficiency map retains its candidate status, single eta bin and DY/WJets pooling assumptions.

Output: `wgt_nominal` and all existing full weight variations include one central factor. `wgt_nominal_without_btag` stores the baseline; `wgt_nominal_supported` stores the supported convention. `btag_sf_nearest`, `btag_sf_supported` and `btag_sf_outside_coverage` record diagnostics. With individual weights enabled, `separate_wgt_xzz_btag` is saved. Supported/nearest are alternate conventions, not up/down nuisance variations. Calibration and efficiency-map systematic uncertainties are not newly propagated by this integration.

Full validation in task xzz-2l2nu-007 iteration017 showed a mixed effect, not global Data/MC closure. Iteration018 compares shared Stage1 to all26 saved MC pilots and checks HMuMu/data isolation and full event-weight variation propagation.
