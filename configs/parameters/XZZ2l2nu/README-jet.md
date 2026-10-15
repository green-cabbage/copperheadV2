# XZZ jets

- Select jets with pT > 30 GeV and |eta| < 4.7. NanoAODv15 uses reconstructed jet ID; the configured `RUN2ULPUPPI_REFSYNC` recipe remains unchanged.
- Clean within ΔR ≤ 0.5 of the configured loose leptons. Use their four momenta before FSR recovery, including Rochester-corrected muon pT.
- Cleaning electrons use pT > 25 GeV, |eta| < 2.5 and the exact excluded gap `1.4442–1.566`. `jet_cleaning_electron_gap` sets these boundaries identically for all UL years. Electron event selection is separate.
- `loose_no_id` omits electron ID only from the cleaning list. Event selection retains Boolean WP90.

Iterations 041–042 fixed post-FSR cleaning and rounded gap boundaries. The jet counts and leading two jets' pT/eta/phi now match the independently reconstructed reference model with corrected inputs in all 194,722 shared 2018 pilot events. Other UL years have the same configuration but were not rerun in this study.

The saved reference skims still differ: their `SetJets` call swaps `chHEF`/`neEmEF` and charged/neutral multiplicities. Our jet ID keeps the correct fields. A tested reference patch and raw event examples are in the [jet report](../../../.agent-system/tasks/xzz-2l2nu-007-control-plot-sync-2026-09/iterations/042-jet-cleaning-gap_2018/evidence-report.md).

B-tag counts use UParT and a separate jet selection; the reference uses DeepJet. These definitions are not interchangeable. The reference jet-ID recipe also omits official multiplicity requirements; that previously adopted distinction is retained, not newly introduced here.

## Optional reference input swap

- `jet_id_reference_input_swap: false` (default): use the correct energy fractions and particle counts for jet ID.
- Set the year entry to `true` in `jet.yaml` to reproduce the reference SetJets argument swaps. Set it back to `false` to restore the normal method.
- This XZZ-only diagnostic swaps charged-hadron and neutral electromagnetic energy fractions, plus charged/neutral particle counts, only inside reconstructed Run 2 jet ID. Original jet fields and momenta remain unchanged. Stored jet-ID flags and Run 3 are unsupported and raise an error if enabled.
- In the barrel, the intended cuts include charged-hadron fraction > 0 and neutral electromagnetic fraction < 0.9. Swapped inputs instead require neutral electromagnetic fraction > 0 and charged-hadron fraction < 0.9. Other cuts still apply.
- Example `1:109383:907542494`, raw jet 1: charged-hadron fraction 0.9458, neutral electromagnetic fraction 0.05435. It passes with correct inputs and fails with swapped inputs. See [raw event evidence](../../../.agent-system/tasks/xzz-2l2nu-007-control-plot-sync-2026-09/iterations/042-jet-cleaning-gap_2018/reference-wiring-examples.csv).
- Multiplicity swaps have no effect in the active 2017/2018 `RUN2ULPUPPI_REFSYNC` recipe because it omits the separate charged/neutral count cuts.
- Changed jet acceptance can change jet counts, leading jets, b-tag counts and jet-dependent weights. This switch does not reproduce unrelated reference differences.

## Combined reference-bug diagnostic

Set the year entry `reference_bug_compatibility` to `true` in `jet.yaml` to enable the following together. It is XZZ Run2 only and defaults to `false`. To restore normal processing, set both this switch and `jet_id_reference_input_swap` to `false`.

| Reference bug | Diagnostic behavior |
|---|---|
| Swapped jet inputs | Uses the swapped jet-ID inputs described above. |
| Electron ID/charge | Bypasses electron ID; cleans against loose electrons without ID. Saved ee charges become WP90 true → -1, false → +1. Raw electron records remain intact. |
| Same-sign pairs | Allows same-sign ee and muon pairs. |
| Close lepton pair | Sets b-tag counts to zero for ee/mm ΔR < 0.3; still counts jets. |
| Leading-jet update | Reproduces the two independent slot updates, including duplicate slots for unsorted inputs. |
| Unset MET direction | Uses -999 rad for the jet–MET angle and 999 for events without selected jets. Saved MET remains unchanged. |
| Ignored mass window | No extra override: Stage1 already does not filter on the mass-window setting. Its saved mass-window flag remains meaningful. |

- This reproduces the listed defects within our existing framework. Lepton multiplicity/pair construction, e-mu control-region charge handling, b-tag definition, FSR and weights retain their separate procedures. It is not a complete nanoAOD_skim replacement.
- Electron ID is bypassed only in diagnostic output; the resulting weights do not define a validated physics selection. Normal defaults retain electron ID and opposite-sign cuts.

- Validation and event examples: [combined diagnostic report](../../../.agent-system/tasks/xzz-2l2nu-007-control-plot-sync-2026-09/iterations/045-reference-close-pair_2018/evidence-report.md).
