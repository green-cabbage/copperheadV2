# XZZ FSR recovery

`fsr.yaml` configures electron and muon cuts independently. Both use:

- photon pT > 2 GeV
- |eta| < 2.4
- relIso03 < 0.8
- lepton–photon ΔR < 0.5
- photon `dROverEt2 < 0.012`

This is consistent for both muons and electrons. Both flavors use NanoAOD `fsrPhotonIdx` for recovery.

Electron WP90 ID, pT/eta acceptance and impact-parameter cuts are evaluated before FSR recovery; isolation ID is not recomputed. Saved electron four momenta include accepted photons; original coordinates and whether recovery occurred are also saved. Jet cleaning uses original lepton four momenta.

For XZZ Run 2:
- **Muon selection:** with FSR enabled, uses corrected `pfRelIso03_all < 0.25`. As in HMuMu, isolation is `(original isolation × muon pT before recovery − photon pT) / recovered muon pT`, applied to XZZ's cone 0.3. Muons without an accepted photon keep their original isolation.
- **Electron selection:** does **not** use FSR-updated isolation. It uses the original Boolean WP90 ID, which is not recomputed after recovery.


- **Muons:** adding the photon changes the combined muon–photon mass as well as pT, eta and phi. We save that recovered mass instead of keeping the original muon mass, which should be fine because this is due to natural combining of muon four momenta and photon four momenta.
- **Electrons:** NanoAOD can encode a slightly negative mass-squared as a negative mass value. The recovery code preserves that sign when calculating the electron’s energy. This is a numerical storage convention, not a physically negative electron mass.

Four-vector addition and selected-pair arithmetic use float64; muon/photon coordinates are converted before trigonometric and energy calculations when the XZZ flavor configuration is active, preventing non-finite masses from float32 cancellation.

Evidence from task xzz-2l2nu-007:

- Iterations025–026 validated muon conservation/precision and electron integration in XZZ2017; those changes initially left other analyses/years at baseline.
- Iteration027 checked the early float64 conversion on two failing boosted-DY events using an independent stable invariant formula, plus three paired XZZ and four paired HMuMu samples; iteration028 integrated it without changing association or cuts.

Iteration039 added the two photon-quality cuts from `nanoAOD_skim`, reducing 2018 DY mass mismatches from 548 to 26 in 167,400 shared pilot events (relative tolerance 0.1%). Data/top pilots and HMuMu regression were also checked.

Iteration040's optional `association: reference_scan` tests the reference's nearby-photon search and leaves 10 DY mass differences. It is not enabled in `fsr.yaml`: the scan can assign one photon to multiple leptons. The remaining ten cases involve seven reference pairs with a failed electron WP90 and three muon-smearing differences. Evidence: `.agent-system/tasks/xzz-2l2nu-007-control-plot-sync-2026-09/iterations/040-fsr-association_2018/`.
