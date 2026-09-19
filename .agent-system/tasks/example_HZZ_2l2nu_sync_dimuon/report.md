# XZZ2l2nu Stage-1 Selection Audit

Does `-A XZZ2l2nu` match the TWiki?

A line-by-line comparison of the object and event selection actually executed by
`run_stage1.py` against the *X → ZZ → 2l2ν analysis synchronisation* TWiki.
Thirty-one checks, each with the verbatim requirement and the code that answers it.

| | |
|---|---|
| Branch | `dev_HZZ_2l2nu_Aug28_2026` |
| Analysis flag | `-A XZZ2l2nu` |
| Era evaluated | 2018, NanoAODv9 |
| TWiki revision | r5, 2025-12-09 |
| Verdict | 18 mismatches, 2 blocking |

Source: *ScalarSearchXZZ2l2nu*, CMS TWiki, revision r5 of 2025-12-09. Page numbers refer
to the PDF export in `ref_materials/HZZ_2l2nu_stage1/`. Line numbers refer to branch
`dev_HZZ_2l2nu_Aug28_2026` and will drift as the file changes.

> This file is the copy of record. It was drafted as a Claude artifact, which has
> since been deleted; nothing else holds this content.

---

## Bottom line

**No.** With `-A XZZ2l2nu` the framework runs the H→µµ selection path with a handful of
renamed config values. Four of thirty-one requirements match outright.

The root cause is structural: `configs/parameters/XZZ2l2nu/xzz2l2nu.yaml` — the file that
looks like it holds the 2l2ν cuts — is dead. A repo-wide search for `zz2l2nu` in Python
matches only a docstring and an argparse choice. Every threshold it declares (muon impact
parameters, Z-candidate window, Z pT, MET, jet–MET Δφ, VBF cuts) is loaded into the config
dict and then never read.

Config resolved by actually calling
`getParametersForYr("./configs/parameters/", "2018", analysis="XZZ2l2nu")`, which merges
`common/*.yaml` with `XZZ2l2nu/*.yaml`. Selection code read from
`src/copperhead_processor.py` and `src/corrections/jet.py`.

### Tally

| Status | Count | Meaning |
|---|---|---|
| Mismatch | 18 | Looser, tighter, or absent |
| Match | 4 | Implemented as specified |
| Extra cut | 4 | Not on the TWiki at all |
| Deferred | 4 | Stored, not enforced |
| By design | 1 | TWiki is stale |

---

## 1. The comparison matrix

### Electrons

| # | Requirement | TWiki specifies | Code does | Status |
|---|---|---|---|---|
| 1 | Electron pT (`electron_pt_cut`) | pT > 25 GeV — *Object Selection › Electrons, p.2*: "pT > 25 GeV" | `electron_pt_cut = 20.0`<br>`XZZ2l2nu/electron.yaml:1` → `copperhead_processor.py:1098` | **Looser** |
| 2 | Electron η (`electron_eta_cut`) | \|η\| < 2.1 — *Object Selection › Electrons, p.2*: "\|η\| < 2.1" | `electron_eta_cut = 2.5`<br>`XZZ2l2nu/electron.yaml:14` → `copperhead_processor.py:1099` | **Looser** |
| 3 | Electron ID (`electron_id_v9 / _v12`) | `Electron_mvaFall17V2Iso_WP90` — *Object Selection › Electrons, p.2*: "Identification: Electron_mvaFall17V2Iso_WP90" | `mvaFall17V2Iso_WP90` for v9; `mvaIso_WP90` for v12/v15 (the equivalent WP)<br>`XZZ2l2nu/electron.yaml:27,41` → `copperhead_processor.py:1100` | Match |
| 4 | Electron isolation | `Electron_pfRelIso03_all`, no threshold given — *Object Selection › Electrons, p.2*: "Isolation: Electron_pfRelIso03_all" | No explicit cut. Isolation enters only implicitly through the Iso WP90 ID; the variable is not written to the output either.<br>`copperhead_processor.py:1097–1102` | Ambiguous — TWiki names a branch but no working point |
| 5 | ECAL gap veto | Nothing. The page has no gap requirement. — *Not mentioned* | Electrons with `1.44 < \|η\| < 1.57` are rejected<br>`copperhead_processor.py:1096, 1101` | **Extra cut** — H→µµ carryover, AN-19-124 |

### Muons

| # | Requirement | TWiki specifies | Code does | Status |
|---|---|---|---|---|
| 6 | Muon pT (`muon_pt_cut`) | pT > 25 GeV — *Object Selection › Muons, p.2*: "pT > 25 GeV" | `muon_pt_cut = 20.0`, and it is applied to `pt_raw` — the pre-Rochester momentum — while the muon written out is post-Rochester and post-FSR.<br>`XZZ2l2nu/muon.yaml:1` → `copperhead_processor.py:903` | **Looser**, and the wrong pT variable |
| 7 | Muon η (`muon_eta_cut`) | \|η\| < 2.4 — *Object Selection › Muons, p.2*: "\|η\| < 2.4" | `muon_eta_cut = 2.4`, applied on `eta_raw`<br>`XZZ2l2nu/muon.yaml:14` → `copperhead_processor.py:904` | Match |
| 8 | Muon ID (`muon_id`) | `Muon_tightId` — *Object Selection › Muons, p.2*: "Identification: Muon_tightId" | `muon_id = mediumId`<br>`XZZ2l2nu/muon.yaml:40` → `copperhead_processor.py:905` | **Looser** |
| 9 | Muon \|dxy\| (`zz2l2nu_muon_dxy_cut`) | \|dxy\| < 0.02 cm — *Object Selection › Muons, p.2*: "Impact parameter requirements: \|dxy\| < 0.02 cm" | No cut anywhere. `zz2l2nu_muon_dxy_cut: 0.045` is declared but never read; `dxy` is only written to the output columns.<br>`xzz2l2nu.yaml:50`; output at `copperhead_processor.py:2028` | **Missing** |
| 10 | Muon \|dz\| (`zz2l2nu_muon_dz_cut`) | \|dz\| < 0.1 cm — *Object Selection › Muons, p.2*: "\|dz\| < 0.1 cm" | No cut anywhere. `zz2l2nu_muon_dz_cut: 0.2` declared, never read.<br>`xzz2l2nu.yaml:64` | **Missing** |
| 11 | Muon isolation (`muon_iso_cut`) | `Muon_pfRelIso04_all`, no threshold given — *Object Selection › Muons, p.2*: "Isolation: Muon_pfRelIso04_all" | Correct branch: `pfRelIso04_all < 0.25`. Note FSR recovery overwrites that branch with `iso_fsr` before the cut is applied.<br>`XZZ2l2nu/muon.yaml:27` → `copperhead_processor.py:935` | Match — threshold unspecified upstream |
| 12 | Global or tracker muon | Nothing. — *Not mentioned* | `isGlobal \| isTracker` required<br>`copperhead_processor.py:906` | **Extra cut** — AN-19-124 Table 3.5 |

### Jets

| # | Requirement | TWiki specifies | Code does | Status |
|---|---|---|---|---|
| 13 | Jet pT (`jet_pt_cut`) | pT > 30 GeV — *Object Selection › Jets, p.2*: "Jet pT > 30 GeV" | `jet_pt_cut = 25.0`<br>`XZZ2l2nu/jet.yaml:1` → `copperhead_processor.py:2824` | **Looser** |
| 14 | Jet η (`jet_eta_cut`) | \|η\| < 4.7 — *Object Selection › Jets, p.2*: "\|η\| < 4.7" | `jet_eta_cut = 4.7`<br>`XZZ2l2nu/jet.yaml:14` → `copperhead_processor.py:2935` | Match |
| 15 | Jet ID (`jet_id`) | `jetId ≥ 1` (loose WP) — *Object Selection › Jets, p.2*: "Jet ID: jetId ≥ 1 (loose WP)" | `jet_id = tight`, which resolves to `jetId >= 2`<br>`XZZ2l2nu/jet.yaml:53` → `corrections/jet.py:277` | **Tighter** — loose WP does not exist in UL NanoAOD, so the TWiki is likely stale here too |
| 16 | Jet pileup ID (`jet_puid` / `do_jet_PUID_cut`) | `puId ≥ 7` (tight WP) — *Object Selection › Jets, p.2*: "Pileup ID: puId ≥ 7 (tight WP)" | `do_jet_PUID_cut = false` for this analysis, so no pileup ID cut is applied at all. The `jet_puid: loose` setting is inert.<br>`XZZ2l2nu/switches.yaml` → `copperhead_processor.py:2802–2806` | **Missing** |
| 17 | Jet–lepton cleaning (`min_dr_mu_jet`) | ΔR(lepton, jet) > 0.4 for **both** electrons and muons — *Object Selection › Jets, p.2*: "ΔR(lepton, jet) > 0.4 for both electrons and muons" | Jets are cleaned against `mu1` and `mu2` only. Electrons never enter the cleaning, so the `ee` channel keeps jets that overlap its own leptons.<br>`copperhead_processor.py:2724–2745` | **Half done** — blocking, see fix 2 |

### Event selection

| # | Requirement | TWiki specifies | Code does | Status |
|---|---|---|---|---|
| 18 | Leading lepton | pT(L1) > 25 GeV, \|η_L1\| < 2.4 — *Event Selection, p.2*: "Leading lepton: pT (L1) > 25 GeV and \|ηL1\| < 2.4" | Not applied. Inherited from the looser object cuts in rows 1, 2, 6 and 7. In `mm` a hidden `pt_roch ≥ 26` arrives via trigger matching instead.<br>`copperhead_processor.py:1006` | **Missing** |
| 19 | Subleading lepton | pT(L2) > 25 GeV, \|η_L2\| < 2.4 — *Event Selection, p.2*: "Subleading lepton: pT(L2) > 25 GeV and \|ηL2\| < 2.4" | Not applied. | **Missing** |
| 20 | Z mass window (`zz2l2nu_z_mass_low / _high`) | \|m_Z1 − 91\| < 15 GeV, i.e. 76–106 — *Event Selection, p.2*: "Z candidate: pT(Z1) > 55 GeV and \|mZ1 − 91 GeV\| < 15 GeV" | Computed as the flag `pass_z_mass_window` and written out, but never used to filter. The config keys are unread; the bounds are hard-coded at the call site.<br>`copperhead_processor.py:1330, 1944` | Flag only — defensible if stage 2 cuts on it |
| 21 | Z pT (`zz2l2nu_z_pt_cut`) | pT(Z1) > 55 GeV — *Event Selection, p.2*: "Z candidate: pT(Z1) > 55 GeV" | Neither applied nor flagged. Only the raw `dilepton_pt` column is stored. `zz2l2nu_z_pt_cut: 55.0` is unread.<br>`copperhead_processor.py:1946` | **No flag** |
| 22 | Missing ET (`zz2l2nu_met_cut`) | MET > 100 GeV — *Event Selection, p.2*: "Missing transverse energy: MET > 100 GeV" | Neither applied nor flagged; only `PuppiMET_pt` is stored. `zz2l2nu_met_cut: 100.0` is unread. The TWiki does not say which MET flavour — the code assumes PuppiMET.<br>`copperhead_processor.py:1934` | **No flag**, and MET flavour is an assumption |

### Trigger

| # | Requirement | TWiki specifies | Code does | Status |
|---|---|---|---|---|
| 23 | HLT paths, 2018 (`hlt`) | About 30 paths across SingleMuon, DoubleMuon, MuonEG and EGamma — *Event Selection › Trigger (2018), pp.1–2*: "The following HLT paths are used for the 2018 data-taking period." | 14 paths configured, 6 of which overlap. Full breakdown in section 2.<br>`XZZ2l2nu/trigger.yaml` → `copperhead_processor.py:804–813` | **Divergent** |
| 24 | Trigger-object matching (`do_trigger_match`) | Nothing beyond the HLT bit firing. — *Not mentioned* | `do_trigger_match = true` requires one of the two leading *muons* to match an IsoMu trigger object (`filterBits & 8`) with `pt_roch ≥ 26`. In `ee` events `nmuons == 0`, so both matches are false and every such event is dropped.<br>`copperhead_processor.py:992–1017` | **Blocking** — removes 100% of the ee channel |

### Corrections applied on top

| # | Requirement | TWiki specifies | Code does | Status |
|---|---|---|---|---|
| 25 | Rochester correction (`do_roccor`) | Nothing. — *Not mentioned* | `do_roccor = true`; muon pT is rescaled before it reaches the output.<br>`copperhead_processor.py:887–897` | **Extra** — shifts dumped sync values |
| 26 | FSR recovery (`do_fsr`) | Nothing. — *Not mentioned* | `do_fsr = true`; overwrites muon pT, η, φ and replaces `pfRelIso04_all` with `iso_fsr` before the isolation cut.<br>`copperhead_processor.py:927–931, 1041–1045` | **Extra** — shifts dumped sync values |
| 27 | HEM veto, 2018 (`do_HemVeto`) | Nothing. — *Not mentioned* | `do_HemVeto = true`; the HEM-affected events are filtered out.<br>`copperhead_processor.py:1124–1131` | **By design** — deliberate; the TWiki is out of date on HEM |

### Dump format

| # | Requirement | TWiki specifies | Code does | Status |
|---|---|---|---|---|
| 28 | Lepton masses (`el1_mass` / `el2_mass`) | `massL1` and `massL2` in the candidate string — *Dump Files Format, p.2*: "…{massL1:.2f}…{massL2:.2f}…" | Electron mass columns are not written. Muon masses are recoverable from the stored kinematics; electron masses are not.<br>`copperhead_processor.py:1952–1961` | **Missing** |
| 29 | isELE field (`channel`) | `1.` when the event is in the electron channel — *Dump Files Format, p.3*: "1. for isELE if it is a electron channel" | Stored as `channel` with 0 = mm, 1 = em, 2 = ee — equivalent information, different encoding.<br>`copperhead_processor.py:1941` | Derivable |
| 30 | Absent-jet sentinel | Jet pT written as `-999.` when no jet is present — *Dump Files Format, p.3*: "-999. for jet pT if jet not present" | Jet columns are padded with `None`, landing in the parquet as nulls rather than the sentinel.<br>`copperhead_processor.py:1456` | Derivable — convert at dump time |

### Root cause

| # | Requirement | TWiki specifies | Code does | Status |
|---|---|---|---|---|
| 31 | The zz2l2nu config block (`xzz2l2nu.yaml`) | — (framework-side) | Fifteen `zz2l2nu_*` keys are defined and merged into the config dict; not one is read by any Python file. `grep -rni zz2l2nu --include=*.py` matches a docstring and an argparse choice, nothing else.<br>`configs/parameters/XZZ2l2nu/xzz2l2nu.yaml` | **Dead config** — drives rows 9, 10, 16, 20, 21, 22 |

---

## 2. Row 23 in detail — the 2018 HLT menu

**In both (6)**

`IsoMu24` · `IsoMu27` · `Ele23_Ele12_CaloIdL_TrackIdL_IsoVL` · `Ele32_WPTight_Gsf` ·
`Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ` · `Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ`

**TWiki only (24)**

`IsoMu30` · `Mu50` · `Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8` ·
`Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8` · `Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass3p8` ·
`Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass8` · `Mu12_TrkIsoVVL_Ele23_…_DZ` ·
`Mu12_TrkIsoVVL_Ele23_…` · `Mu23_TrkIsoVVL_Ele12_…` · `Mu8_TrkIsoVVL_Ele23_…` ·
`Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ` · `DiEle27_WPTightCaloOnly_L1DoubleEG` ·
`DoubleEle33_CaloIdL_MW` · `DoubleEle25_CaloIdL_MW` · `DoubleEle27_CaloIdL_MW` ·
`DoublePhoton70` · `Ele115_CaloIdVT_GsfTrkIdT` · `Ele27_WPTight_Gsf` ·
`Ele35_WPTight_Gsf` · `Ele38_WPTight_Gsf` · `Ele40_WPTight_Gsf` ·
`Ele32_WPTight_Gsf_L1DoubleEG` · `Photon200`

The four non-DZ MuonEG variants are counted individually above.

**Code only (8)**

`IsoMu20` · `Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ` · `Photon50_R9Id90_HE10_IsoM` ·
`Photon75_R9Id90_HE10_IsoM` · `Photon90_R9Id90_HE10_IsoM` · `Photon120_R9Id90_HE10_IsoM` ·
`Photon165_R9Id90_HE10_IsoM` · `Photon300_NoHE`

The DoubleMu path present here is the plain `_DZ` variant; the TWiki asks for the
`Mass3p8` and `Mass8` versions.

---

## 3. Fix order

These three are ordered by dependency, not by severity alone — each one gates the
usefulness of testing the next.

1. **Gate the muon trigger matching on channel.** Row 24. Until this is conditional on
   `is_mm` (or on `config["analysis"]`), the `ee` channel produces zero events and nothing
   downstream of it can be validated. It also imposes an unadvertised 26 GeV leading-muon
   threshold in `mm`.

2. **Clean jets against electrons.** Row 17. Even once `ee` events survive, their jet
   collection is wrong: jets overlapping the two selected electrons are kept, which
   propagates into `min_delta_phi_jet_MET`, the jet counts and the VBF variables.

3. **Wire up the zz2l2nu config keys.** Row 31. The dead block is why rows 9, 10, 16, 20,
   21 and 22 are *silently* absent rather than loudly wrong. Reading these keys in the
   processor turns six invisible omissions into six explicit, reviewable cuts.

---

**No code was modified.** This report is a read-only audit.
