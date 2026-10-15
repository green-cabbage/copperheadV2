# Reference observable diagnostic

- `reference.yaml` → `reference_observables` → year → `enabled: true` enables the observable diagnostic. Default is `false`; setting it back restores normal definitions.
- This is separate from `reference_bug_compatibility`, which reproduces the reference code bugs. The synchronization pilot enables both.

| Quantity | Diagnostic definition |
|---|---|
| Analysis MET | `mc_met`/`data_met`: `raw` uses input PuppiMET; `configured` keeps the configured XY correction. Verified2018 pilot setup: MC raw, data configured. |
| Transverse mass | Save ROOT `Mt` of Z + massless MET directly in `MT_ZZ`. |
| Loose/medium b-tag count | Count selected jets with DeepJet > configured threshold. Verified reference2017/2018 configs use0.0494/0.2770. |

- ROOT `Mt` is `sqrt(m_Z² + pT_Z² + MET² + 2 E_Z MET)`, with `E_Z = sqrt(m_Z² + (pT_Z cosh(eta_Z))²)`.
- DeepJet counts use the same selected jets as `njets`; in this pilot, pT >30 GeV and |eta| <4.7. Existing close-pair bug mode still zeros b-tag counts for ΔR <0.3.
- Normal MET corrections, normal transverse-mass definition and UParT counts remain the defaults. This option changes reported b-tag counts, not b-tag scale factors. Combined reference weights remain deferred.
- Only the2017/2018 reference threshold settings have been verified here. Other-year entries are disabled and require verified thresholds before enabling. No year conditions are hard-coded in processing.
- User accepts dilepton-mass agreement as complete; remaining measured mass differences are retained as accepted evidence.
