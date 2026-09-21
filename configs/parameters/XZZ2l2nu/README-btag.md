# XZZ UnifiedParT b-tag event weights

`switches.do_xzz_btag_sf` enables central medium-working-point SFs in shared Stage1 for XZZ2l2nu 2018 NanoAODv15 MC. Data skips them; legacy `do_btag_wgt` must be false (for the time being, needs to be refactored).

`btag.yaml` selects how to handle jets outside the b-tag calibration’s valid range with two modes:
- **`nearest`:** use the scale factor at the nearest valid calibration boundary.
- **`supported`:** use a scale factor of 1 outside the valid range, applying no correction there.
`nearest` is the configured default.

Multiply over the exact veto candidate jets: SF for tagged jets, `(1-SF*eff)/(1-eff)` for untagged jets, unity for empty events.

`btag_efficiency_2018.json` freezes candidate efficiencies and assignments for 26 validated backgrounds. New samples require justified assignments; unknown samples fail. Paths resolve from the repository root, including on Gateway workers. The map retains a single eta bin and DY/WJets pooling assumptions.

`src/corrections/xzz_btag_sf.py` pins the BTV payload. Outside coverage, `supported` uses unity; `nearest` clips SF lookup coordinates. Efficiencies use original jet pT. These user-approved conventions are not official extrapolation recommendations or up/down nuisances; no new calibration or efficiency-map systematics are propagated.

Outputs:

- `wgt_nominal` and existing full weight variations include one central factor.
- `wgt_nominal_without_btag` saves the baseline; `wgt_nominal_supported` saves the supported convention.
- Diagnostics: `btag_sf_nearest`, `btag_sf_supported`, `btag_sf_outside_coverage`; optionally `separate_wgt_xzz_btag` with individual weights enabled.

Evidence: task xzz-2l2nu-007 iteration017 found mixed Data/MC effects, without global closure. Iteration018 compares all 26 saved MC pilots and checks that data is unaffected and weight variations propagate correctly.
