# XZZ2017 FSR recovery

`fsr.yaml` configures electron and muon photon cuts independently. Both initially use the supplied HZZ slide's photon pT>2GeV, |eta|<2.4 and relIso03<0.8. The user explicitly authorizes recovery for both flavors; equality of the current cuts is not an assumption that future prescriptions must be identical. The linked NanoAOD `fsrPhotonIdx` supplies association; no additional association-quality threshold is inferred from the reference.

Electron Boolean WP90, original pT/eta acceptance and impact-parameter cuts are evaluated before dressing. Electron isolation ID is not recomputed. Saved electron p4 includes the accepted photon; raw coordinates and recovery flags are retained. Jet cross-cleaning uses the original lepton directions, consistently with existing muon cleaning. Muon isolation recovery retains its existing separate procedure.

Recovered muon mass accompanies recovered pT/eta/phi. Electron recovery uses double-precision four-vector addition and preserves the signed mass-squared convention of the serialized NanoAOD electron. Selected-lepton pair arithmetic uses double precision to avoid loss from float32 subtraction. These new changes are confined to XZZ2017; other analyses/years retain their baseline behavior.

Evidence: task iterations025 (muon four-vector conservation and precision candidate),026 (electron integration validation). Full reference agreement is not assumed; additional reference photon cuts or selection differences need concrete event evidence.

Muon and photon coordinates are also converted to float64 **before** trigonometric and energy arithmetic when the XZZ flavor configuration is active. This prevents float32 cancellation from producing non-finite recovered muon masses. The iteration 027 candidate was checked on both failing boosted-DY events with an independent stable invariant formula, three paired XZZ samples and four paired HMuMu samples; iteration 028 integrates the identical candidate. Photon association and selection cuts are unchanged.
