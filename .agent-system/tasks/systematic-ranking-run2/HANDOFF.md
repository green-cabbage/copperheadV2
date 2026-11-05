# Systematic ranking — how the deltas were produced

Run2 VBF H->mumu, Asimov. Everything below is what was actually run; `run_ranking.sh`
automates steps 1-5 and `summarize_ranking.py` does step 6.

**Method.** sigma_total from the fully floating fit, sigma_frozen with one group's
nuisances frozen, and the group's contribution is

    delta_g = sqrt(sigma_total^2 - sigma_frozen_g^2)

Groups are passed straight to `--freezeParameters`. They are never written into the
datacard, so no card or workspace needs regenerating.

---

## 0. Inputs

    CARD_DIR=/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_BtagZptFix_sept29_2026/stage3_datacards_20bin_noFixMass/score_Run2_NanoV15_BtagZptFix_sept29_2026
    W=$CARD_DIR/HMuMu_13TeV_Run2.root        # pre-existing, read-only
    cd $(mktemp -d)                          # combine writes to CWD -- stay out of CARD_DIR

## 1. Environment

`pixi run -e combine` is broken on the AF (`PIXI_CACHE_DIR` resolves to
`/work/users//.pixi-cache`, empty username -> EACCES). Point at the env directly:

    E=/work/users/yun79/sideHustle2/copperheadV2/.pixi/envs/combine
    export PATH="$E/bin:$PATH"
    export LD_LIBRARY_PATH="$E/lib:${LD_LIBRARY_PATH:-}"
    export ROOTSYS="$E"

Cling will print `'assert.h' file not found` when the combine dictionary loads.
Cosmetic — ignore it.

## 2. Group -> comma-separated member lists

    python3 -c "
    import json
    d = json.load(open('nuisance-groups.json'))
    for g, ms in d['primary_groups'].items():
        print(g, ','.join(ms))
    " > groups.txt

11 lines, 103 names total (95 constrained nuisances = the card's kmax, + 8 DY
rateParams). On 21bin_btagZpt the same 11 groups hold 121 names; the difference is
entirely `pdf_alpha_s` (18 vs 36) -- see the second-production section.

### The grouping map

11 groups, 103 names: 95 constrained nuisances (= the card's `kmax`) + 8 DY
rateParams. Every one appears in exactly one group; `shape` unless marked.
Full per-nuisance detail (which processes and year-bins each touches) is in
`nuisance-groups.json`.

```
jes (27)
    Absolute Absolute_2016 Absolute_2016APV Absolute_2017 Absolute_2018
    BBEC1 BBEC1_2016 BBEC1_2016APV BBEC1_2017 BBEC1_2018 EC2 EC2_2016
    EC2_2016APV EC2_2017 EC2_2018 FlavorQCD HF HF_2016 HF_2016APV HF_2017
    HF_2018 RelativeBal RelativeSample_2016 RelativeSample_2016APV
    RelativeSample_2017 RelativeSample_2018 Total

btag (10)
    btag_bc_2016postVFP btag_bc_2016preVFP btag_bc_2017 btag_bc_2018
    btag_bc_correlated btag_light_2016postVFP btag_light_2016preVFP
    btag_light_2017 btag_light_2018 btag_light_correlated

muon (12)
    muID2016 muID2017 muID2018 muIso2016 muIso2017 muIso2018 muTrig2016
    muTrig2017 muTrig2018 mu_roccor2016 mu_roccor2017 mu_roccor2018

pileup (3)
    pu_wgt2016 pu_wgt2017 pu_wgt2018

prefiring (3)
    l1prefiring_wgt2016 l1prefiring_wgt2017 l1prefiring_wgt2018

lumi (3)  [lnN]
    lumi2016 lumi2017 lumi2018

zpt (1)
    zpt

bkg_DY_norm (8)  [rateParam, unconstrained]
    XSecAndNorm2016preVFPDY01J XSecAndNorm2016preVFPDY2J
    XSecAndNorm2016postVFPDY01J XSecAndNorm2016postVFPDY2J
    XSecAndNorm2017DY01J XSecAndNorm2017DY2J XSecAndNorm2018DY01J
    XSecAndNorm2018DY2J

pdf_alpha_s (18)
    pdf_alpha_s_unc_DY_matched01J2016 pdf_alpha_s_unc_DY_matched01J2017
    pdf_alpha_s_unc_DY_matched01J2018 pdf_alpha_s_unc_DY_matched2J2016
    pdf_alpha_s_unc_DY_matched2J2017 pdf_alpha_s_unc_DY_matched2J2018
    pdf_alpha_s_unc_EWK2016 pdf_alpha_s_unc_EWK2017
    pdf_alpha_s_unc_EWK2018 pdf_alpha_s_unc_TT+ST2016
    pdf_alpha_s_unc_TT+ST2017 pdf_alpha_s_unc_TT+ST2018
    pdf_alpha_s_unc_VBF2016 pdf_alpha_s_unc_VBF2017
    pdf_alpha_s_unc_VBF2018 pdf_alpha_s_unc_ggH2016
    pdf_alpha_s_unc_ggH2017 pdf_alpha_s_unc_ggH2018

LHEFac_norm (8)
    LHEFac_DY_matched01J LHEFac_DY_matched2J LHEFac_EWK LHEFac_TT+ST
    LHERen_DY_matched01J LHERen_DY_matched2J LHERen_EWK LHERen_TT+ST

STX_qqH (10)
    THU_VBF_JET01 THU_VBF_Mjj1000 THU_VBF_Mjj120 THU_VBF_Mjj1500
    THU_VBF_Mjj350 THU_VBF_Mjj60 THU_VBF_Mjj700 THU_VBF_PTH200
    THU_VBF_PTH25 THU_VBF_Yield
```

Year suffixes on the JES sources: `_2016APV` = 2016preVFP, `_2016` = 2016postVFP.
Note `lumi`, `pdf_unc_*` and `alpha_s_unc_*` carry a bare `2016` that covers **both**
VFP eras (one parameter shared by the 2016preVFP and 2016postVFP bins), whereas
`btag_*` splits them — so the three year groups are not the same partition
everywhere.

Coarse roll-ups, if a summary table is wanted instead of the full 11 rows. They
contain whole groups, so never mix the two levels in one table:

    experimental (58) = jes + btag + muon + pileup + prefiring + lumi
    theory       (36) = pdf_alpha_s + LHEFac_norm + STX_qqH
    modelling     (9) = zpt + bkg_DY_norm

(`theory` is 54 on 21bin_btagZpt, where `pdf_alpha_s` has 36 members not 18.)

## 3. Common options

    COMMON=(-M MultiDimFit "$W" -m 125 --algo singles \
            --setParameterRanges r=-5,5 -t -1 --expectSignal 1)

`-t -1 --expectSignal 1` -> Asimov, because the analysis is blinded. On Asimov the
nuisances sit at nominal at the best fit, so freezing at prefit *is* freezing at
postfit and no `--saveWorkspace`/`--snapshotName` two-stage is needed.

## 4. Nominal fit -> sigma_total

    combine "${COMMON[@]}" --freezeParameters MH -n .nominal

    r = 1.000000  -0.687714/+0.765740   sym 0.726727

## 5. Reference fit -> the stat floor

Freeze the union of the groups' own members. **Not** `allConstrainedNuisances` —
see gotcha 3.

    ALL=$(cut -d' ' -f2 groups.txt | paste -sd,)
    combine "${COMMON[@]}" --freezeParameters "MH,$ALL" -n .freeze_union

    sym 0.678699   =>  total systematic = sqrt(0.726727^2 - 0.678699^2) = 0.259808

## 6. One fit per group -> sigma_frozen

~3.5 s each, so just fan them out:

    while read -r g members; do
      ( combine "${COMMON[@]}" --freezeParameters "MH,$members" -n ".freeze_$g" \
          > "fit_$g.log" 2>&1 ) &
    done < groups.txt
    wait

## 7. Read the numbers at full precision

The log prints 3 decimals, which cannot separate the small groups — four of them
differ from sigma_total in the 4th decimal or beyond. Read the tree instead. With
`--algo singles` the `limit` tree has 3 entries: best fit, -1sigma, +1sigma.

    python3 - <<'PY'
    import ROOT, math
    ROOT.gROOT.SetBatch(True)
    def sig(tag):
        f = ROOT.TFile.Open(f"higgsCombine.{tag}.MultiDimFit.mH125.root")
        t = f.Get("limit"); r = []
        for i in range(t.GetEntries()):
            t.GetEntry(i); r.append(t.r)
        f.Close()
        return r[0] - r[1], r[2] - r[0]          # (lower error, upper error)

    q = lambda a, b: math.sqrt(max(a*a - b*b, 0.0))
    lo, hi = sig("nominal"); tot = (lo + hi) / 2
    for g in [l.split()[0] for l in open("groups.txt")]:
        fl, fh = sig(f"freeze_{g}")
        print(f"{g:<14} {q(tot, (fl+fh)/2):.4f}   dn {q(lo,fl):.4f}  up {q(hi,fh):.4f}")
    PY

Sort descending and that is the ranking. Quote absolute deltas, not percentages
(gotcha 5).

---

## MultiDimFit results

For `stage3_datacards_20bin_noFixMass` (kmax 95). Asimov, so `r = 1.0000` in every
fit by construction — all the information is in the uncertainties. Read from the
`limit` tree at full precision, shown rounded. The 21bin_btagZpt equivalent is in
`results/21bin_btagZpt/raw_fits.json`.

|  fit | r | -1sigma | +1sigma | sym | **delta sym** |
|---|---|---|---|---|---|
| nominal (all floating) | 1.0000 | -0.6877 | +0.7657 | 0.72673 | — |
| all 11 groups frozen | 1.0000 | -0.6437 | +0.7137 | 0.67870 | **0.04803** |
| freeze `jes` | 1.0000 | -0.6688 | +0.7473 | 0.70805 | **0.01868** |
| freeze `bkg_DY_norm` | 1.0000 | -0.6761 | +0.7569 | 0.71651 | **0.01022** |
| freeze `LHEFac_norm` | 1.0000 | -0.6839 | +0.7633 | 0.72357 | **~0.00315** |
| freeze `pdf_alpha_s` | 1.0000 | -0.6847 | +0.7634 | 0.72408 | **0.00265** |
| freeze `muon` | 1.0000 | -0.6859 | +0.7648 | 0.72537 | **~0.00135** |
| freeze `pileup` | 1.0000 | -0.6868 | +0.7647 | 0.72577 | **0.00096** |
| freeze `STX_qqH` | 1.0000 | -0.6877 | +0.7647 | 0.72619 | **~0.00054** |
| freeze `lumi` | 1.0000 | -0.6876 | +0.7651 | 0.72633 | **~0.00040** |
| freeze `btag` | 1.0000 | -0.6874 | +0.7654 | 0.72639 | **0.00034** |
| freeze `prefiring` | 1.0000 | -0.6876 | +0.7655 | 0.72657 | **~0.00015** |
| freeze `zpt` | 1.0000 | -0.6877 | +0.7657 | 0.72673 | **~0.00000** |

`sym` = (|-1sigma| + |+1sigma|)/2.

**How `delta sym` is computed.**

    sym_X       = ( |-1sigma_X| + |+1sigma_X| ) / 2
    delta_sym_g = abs( sym_nominal - sym_g )

with `sym_nominal` = 0.72673. Worked example, `jes`:

    sym_jes   = (0.6688 + 0.7473) / 2 = 0.70805
    delta_sym = abs(0.72673 - 0.70805)  = 0.01868

That is how much sigma(r) shrinks when the group is frozen, and it is what sets
the ranking. Note it is a shrinkage, **not** a contribution in quadrature: these
deltas do not combine with each other and are not commensurate with the total
systematic. If you need a number that is, `ranking.json` carries `delta_quad` =
sqrt(sym_nominal^2 - sym_g^2) per group (jes 0.1637, total systematic 0.2598).
Both definitions are monotone in `sym_g`, so **the ranking order is identical
either way** -- only the scale differs.

`~` marks the six rows -- `LHEFac_norm`, `muon`, `STX_qqH`, `lumi`, `prefiring`,
`zpt` -- where the down-side and up-side results disagree by more than 30%, so
`sym` is averaging two estimates that do not agree (gotcha 7); the per-side values
are in `ranking.json` as `delta_down` / `delta_up`.

Combine reports `-1sigma` as a negative number, so square or subtract the absolute
value.

Note how little separates the bottom rows from nominal: `zpt` is identical to 4
decimals on both sides, and `prefiring` shrinks sigma by 1.5e-4. This is why step 7
reads the tree instead of the log, whose 3 decimals would show the bottom five rows
as zero. `zpt` in fact comes out 3.3e-07 *wider* than nominal -- fit noise, and the
reason the quadrature floor exists.

---

## Gotchas, all of them hit during this work

1. **`--setParameterRanges r=0.5,2.5` clips.** It returns a lower error of exactly
   -0.500 — the distance to the boundary, not a measurement. `r=-2,4` and `r=-5,5`
   both give -0.704 and agree to 6 decimals.

2. **`--algo singles` rather than `--algo grid --points 20`, but not because the
   grid is wrong.** Measured both ways on 21bin_btagZpt (grid: 20 points over `r=-2,4`, sigma from
   the 2*deltaNLL=1 crossing by linear interpolation):

   | | grid sym | singles sym | grid delta | singles delta |
   |---|---|---|---|---|
   | nominal | 0.72975 | 0.73692 | — | — |
   | `jes` | 0.71613 | 0.72402 | 0.01362 | 0.01290 |
   | `pileup` | 0.72877 | 0.73605 | 0.00098 | 0.00087 |
   | `zpt` | 0.72974 | 0.73691 | 0.00001 | 0.00001 |

   The grid's **absolute** sigma is biased ~1% low (0.72975 vs 0.73692), but that
   bias is common to every fit and largely cancels in the difference, so the
   grid's deltas land within 5-13% and would give the same ordering. Singles was
   chosen because it is 20x cheaper (1 fit vs 20 per configuration), reads the
   error from the algorithm built to compute it with no interpolation step, and
   gets the absolute sigma right -- which matters if you quote sigma_total
   alongside the ranking. If you only need the ordering, the grid is fine.

3. **`allConstrainedNuisances` also freezes the `prop_bin*` autoMCStats
   parameters.** On 21bin_btagZpt it gives 0.675677 where the group union gives
   0.694184; on 20bin_noFixMass, 0.66382 against 0.67870. Using it
   inflates the total systematic to 0.333 and drops closure to 0.63 — a
   plausible-looking wrong answer. Always freeze your own group members.

4. **`--robustFit 1` and `--cminDefaultMinimizerStrategy 2` fail on this
   workspace**, emitting "No valid high-error found" and returning range-clipped
   values (sigma 2.35, 3.37). The default minimiser converges cleanly and agrees
   with a 100x tighter tolerance to 6 decimals. Don't mistake those for estimator
   variants.

5. **The deltas do not sum to the total, and must not be shown as percentages.**
   0.235 quadrature sum vs 0.260 direct = 82% of the variance on this card (0.210
   vs 0.247, 72%, on 21bin_btagZpt). Freezing one group
   lets the other ten re-profile and absorb part of its effect, so each delta is
   marginal, not standalone. Verified not to be a symmetrization artifact: 81.9%
   down-side, 63.9% up-side.

6. **Unique `-n` per fit.** Reusing one `-n` silently overwrites earlier outputs.

7. **Over half the table is unresolved.** Compare the down-side and up-side
   deltas for each group. On 20bin_noFixMass, `jes`, `bkg_DY_norm`, `pdf_alpha_s`,
   `pileup` and `btag` agree to better than 30%; the other six -- `LHEFac_norm`,
   `muon`, `STX_qqH`, `lumi`, `prefiring`, `zpt` -- do not. Their rank among
   themselves is noise; quote them as bounds. The flagged set is not stable between
   productions (`bkg_DY_norm` and `LHEFac_norm` swap), which is itself a reason not
   to read precision into the small rows.

---

## One caveat on the result itself

**MC statistical was excluded** on request, but it is not small: 0.223 against a
total systematic of 0.247 on 21bin_btagZpt, and 0.141 against 0.260 on
20bin_noFixMass. Neither measurement is systematics-dominated.

## Compared with `stage3_datacards_21bin_btagZpt`

Everything above is 20bin_noFixMass. The same method was first run on
21bin_btagZpt; same 11 groups, and both are kept under `results/`:

    ./run_ranking.sh <outdir> /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_BtagZptFix_sept29_2026/stage3_datacards_21bin_btagZpt/score_Run2_NanoV15_BtagZptFix_sept29_2026/HMuMu_13TeV_Run2.txt

| | 21bin_btagZpt | 20bin_noFixMass |
|---|---|---|
| kmax | 113 | **95** |
| nuisances + rateParams | 121 | 103 |
| sigma_total | 0.736924 | 0.726727 |
| sigma, all groups frozen | 0.694184 | 0.678699 |
| total systematic (quad) | 0.247317 | 0.259808 |
| closure | 0.8499 | 0.9060 |

**The inventory changed between the two.** Upstream merged `pdf_unc_*` and
`alpha_s_unc_*` into single `pdf_alpha_s_unc_<proc><year>` nuisances: 36 -> 18,
which is exactly the 113 -> 95 difference. Nothing else moved. `derive_nuisance_groups.py` caught it by
failing the kmax check rather than quietly grouping 95 of 95 and leaving the 18 new
names ungrouped; it now carries rules for both spellings, and both cards pass.

### Ranking, side by side

| group | N 21bin | N 20bin | delta sym 21bin | delta sym 20bin | quad 21bin | quad 20bin |
|---|---|---|---|---|---|---|
| `jes` | 27 | 27 | 0.01290 | 0.01868 | 0.1373 | 0.1637 |
| `bkg_DY_norm` | 8 | 8 | 0.00778 | 0.01022 | 0.1068 | 0.1214 |
| `LHEFac_norm` | 8 | 8 | 0.00416 | 0.00315 | 0.0782 | 0.0676 |
| `pdf_alpha_s` | 36 | 18 | 0.00272 | 0.00265 | 0.0633 | 0.0620 |
| `muon` | 12 | 12 | 0.00014 | 0.00135 | 0.0142 | 0.0443 |
| `pileup` | 3 | 3 | 0.00087 | 0.00096 | 0.0358 | 0.0373 |
| `STX_qqH` | 10 | 10 | 0.00048 | 0.00054 | 0.0267 | 0.0280 |
| `lumi` | 3 | 3 | 0.00044 | 0.00040 | 0.0254 | 0.0241 |
| `btag` | 10 | 10 | 0.00038 | 0.00034 | 0.0238 | 0.0222 |
| `prefiring` | 3 | 3 | 0.00026 | 0.00015 | 0.0197 | 0.0149 |
| `zpt` | 1 | 1 | 0.00001 | 0.00000 | 0.0042 | 0.0000 |

Ordering is nearly the same. The one real mover is **`muon`, 10th -> 5th**
(delta sym 0.00014 -> 0.00135, quad 0.0142 -> 0.0443). `jes` and `bkg_DY_norm` both
grow. `pdf_alpha_s` is essentially unchanged (quad 0.0633 -> 0.0620) despite halving
its nuisance count -- what you would expect if the upstream merge preserved the
total PDF+alpha_s variation rather than dropping part of it.

`zpt` is the first row to reach the quadrature floor: its frozen fit comes out
3.3e-07 *wider* than nominal, which is fit noise, not a negative contribution.
`summarize_ranking.py` now warns on stderr when that happens instead of silently
printing 0.0000.

### Cross-check against `significance.txt`

That log reports Run2 prefit 1.51320, stat-only 1.73001. Its "Stat Only" freezes
`allConstrainedNuisances`, so the comparable fit is that one, not the group union:

| fit | sigma |
|---|---|
| nominal | 0.72673 |
| all 11 groups frozen | 0.67870 |
| `allConstrainedNuisances` | 0.66382 |

ratio nominal/allConstrained = **1.0948** vs their significance ratio **1.1433** --
consistent at the ~4% level, not exact. Significance is sqrt(2*deltaNLL) from r=0
while these sigmas are the 68% profile interval at the minimum, and the errors are
asymmetric (-0.688/+0.766), so the two are not reciprocal. A sanity check, not a
closure test.

That fit also gives this card's MC-stat contribution:
sqrt(0.67870^2 - 0.66382^2) = **0.141**, against 0.260 for all systematics.

---

## ggH channel — `Run2_BtagZptFix_sept29_2026_ggh_passB`

A different beast from the VBF cards above: a **parametric** fit, 5 BDT categories,
background from a RooMultiPdf envelope. The parameters that matter are not in the
datacard at all -- they are free RooRealVars in the workspace -- so grouping the
card alone would miss the entire result.

### Inputs

    CARD=/work/users/yun79/sideHustle2/copperheadV2/output/bdt_Run2_NanoV15_May24_2026_all/stage3/all/Run2_BtagZptFix_sept29_2026_ggh_passB/datacards/datacard_comb_sig_all_ggh.txt
    W=${CARD%.txt}.root          # pre-existing, read-only; fits read this
    cd $(mktemp -d)              # combine writes to CWD

Environment: same as step 1 above (point PATH/LD_LIBRARY_PATH/ROOTSYS at
`.pixi/envs/combine`).

### Where the parameters live

`kmax 58` = 48 `lnN` + 10 `param` (`CMS_hmm_peak/sigma_cat0-4`). `pdf_index_ggh`
is `discrete` and not counted. **On top of those, 25 free background parameters
exist only in the workspace** and carry almost all the uncertainty. To list them:

    python3 - <<'EOF'
    import ROOT; ROOT.gROOT.SetBatch(True)
    f=ROOT.TFile.Open("datacard_comb_sig_all_ggh.root")   # keep this reference alive
    w=f.Get("w"); mc=w.genobj("ModelConfig")
    inmc={p.GetName() for p in mc.GetNuisanceParameters()}
    obs={o.GetName() for o in mc.GetObservables()}        # mh_ggh, CMS_channel
    for v in sorted(w.allVars(), key=lambda v: v.GetName()):
        n=v.GetName()
        if not v.isConstant() and n not in inmc and n not in obs and n!="r":
            print(n)
    EOF

Three background families per category, selected by `pdf_index_ggh`
(**0 = sumExp**, 1 = BWZRedux, 2 = FEWZxBern), each as SMF x core:

| family | core parameters |
|---|---|
| sumExp | `RooSumTwoExpPdf_a1_coeff`, `_a2_coeff`, `_f_coeff` |
| BWZRedux | `BWZ_Redux_a_coeff`, `_b_coeff`, `_c_coeff` |
| FEWZxBern | `FEWZxBern_c2`, `_c3`, `_c4` |

plus the per-category Chebychev SMF coefficients `a0_subCat0-4`, `a1_subCat0-4`,
`a3_subCat0` (cat0 is 3rd order, the rest 2nd), and the 5 normalisations
`shapeBkg_bkg_ch1-5__norm`.

**Watch out: the core coefficients are shared across all five categories.** All
five `my_workspace/workspace_bkg_cat*.root` give them identical names, so
`text2workspace` merges them into one parameter each; only the SMF coefficients
are genuinely per-category. If each category was meant its own background shape
freedom, this understates the background uncertainty. Not confirmed as intended.

### The grouping

11 groups. The card side sums to 58 = `kmax`; `jes`, `btag`, `muon`, `pileup`,
`prefiring`, `lumi`, `STX_qqH` and `pdf_alpha_s` use the same names as the VBF
grouping above so the two channels can be read side by side. The VBF-only groups
have no counterpart here (no `zpt`, no `bkg_DY_norm`, no `LHEFac_norm`).
`pdf_index_ggh` is in no group and is never frozen -- the envelope stays active in
every fit.

```
bkg_fit_params (25)
    RooSumTwoExpPdf_a1_coeff RooSumTwoExpPdf_a2_coeff
    RooSumTwoExpPdf_f_coeff BWZ_Redux_a_coeff BWZ_Redux_b_coeff
    BWZ_Redux_c_coeff FEWZxBern_c2 FEWZxBern_c3 FEWZxBern_c4 a0_subCat0
    a0_subCat1 a0_subCat2 a0_subCat3 a0_subCat4 a1_subCat0 a1_subCat1
    a1_subCat2 a1_subCat3 a1_subCat4 a3_subCat0 shapeBkg_bkg_ch1__norm
    shapeBkg_bkg_ch2__norm shapeBkg_bkg_ch3__norm shapeBkg_bkg_ch4__norm
    shapeBkg_bkg_ch5__norm

sig_shape (10)
    CMS_hmm_peak_cat0_ggh CMS_hmm_sigma_cat0_ggh CMS_hmm_peak_cat1_ggh
    CMS_hmm_sigma_cat1_ggh CMS_hmm_peak_cat2_ggh CMS_hmm_sigma_cat2_ggh
    CMS_hmm_peak_cat3_ggh CMS_hmm_sigma_cat3_ggh CMS_hmm_peak_cat4_ggh
    CMS_hmm_sigma_cat4_ggh

jes (21)
    Absolute Absolute_2016 Absolute_2017 Absolute_2018 BBEC1 BBEC1_2016
    BBEC1_2017 BBEC1_2018 EC2 EC2_2016 EC2_2017 EC2_2018 FlavorQCD HF
    HF_2016 HF_2017 HF_2018 RelativeBal RelativeSample_2016
    RelativeSample_2017 RelativeSample_2018

btag (6)
    wgt_btag_bc_2016postVFP wgt_btag_bc_2016preVFP wgt_btag_bc_2017
    wgt_btag_bc_2018 wgt_btag_bc_correlated wgt_btag_light_correlated

muon (3)
    wgt_muID wgt_muIso wgt_muTrig

pileup (1)
    wgt_pu

prefiring (1)
    wgt_l1prefiring

lumi (3)
    lumi_13TeV_2016 lumi_13TeV_2017 lumi_13TeV_2018

STX_qqH (8)
    wgt_THU_VBF_JET01 wgt_THU_VBF_Mjj1000 wgt_THU_VBF_Mjj120
    wgt_THU_VBF_Mjj350 wgt_THU_VBF_Mjj60 wgt_THU_VBF_Mjj700
    wgt_THU_VBF_PTH25 wgt_THU_VBF_Yield

pdf_alpha_s (2)
    pdf_Higgs_gg pdf_Higgs_qq

theory_xs_br (3)
    QCDscale_ggH QCDscale_qqH BR_mm
```

### Commands

Asimov is generated at `pdf_index_ggh=0`, i.e. **sumExp**, which is what
`--setParameters pdf_index_ggh=0` does. This is the expected pre-fit significance
setting (`-t -1 --expectSignal 1`, no postfit snapshot).

    W=.../datacard_comb_sig_all_ggh.root
    C="-M MultiDimFit $W -m 125 --algo singles -t -1 --expectSignal 1 \
       --setParameterRanges r=-2,4 --setParameters pdf_index_ggh=0 --robustFit 1 \
       --X-rtd MINIMIZER_freezeDisassociatedParams"

    # nominal
    combine $C --freezeParameters MH -n .nom

    # one fit per group
    while read -r g members; do
      ( combine $C --freezeParameters "MH,$members" -n ".fz_$g" > "f_$g.log" 2>&1 ) &
    done < groups.txt
    wait

~2 min each with `--robustFit 1`; 11 run concurrently. Read `r` and its errors
from the `limit` tree exactly as in step 7 above, then
`delta_sym = abs(sym_nominal - sym_group)`.

For the record, the matching expected pre-fit significance at sumExp:

    combine -M Significance $W -m 125 -t -1 --expectSignal 1 --setParameterRanges r=-2,4 \
      --setParameters pdf_index_ggh=0 --freezeParameters MH \
      --cminDefaultMinimizerStrategy 0 --X-rtd MINIMIZER_freezeDisassociatedParams
    # Significance: 1.55661      (1.67054 on the workspace's stored index 2)

### Results

|  fit | N | r | -1sigma | +1sigma | sym | **delta sym** |
|---|---|---|---|---|---|---|
| nominal (all floating) | — | 0.9996 | -1.1720 | +0.8831 | 1.02756 | — |
| freeze `bkg_fit_params` | 25 | 1.0001 | -0.4760 | +0.4882 | 0.48208 | **0.54548** |
| freeze `theory_xs_br` | 3 | 0.9998 | -1.1722 | +0.8660 | 1.01911 | **0.00845** |
| freeze `sig_shape` | 10 | 0.9996 | -1.1722 | +0.8775 | 1.02486 | **0.00269** |
| freeze `pdf_alpha_s` | 2 | 0.9996 | -1.1721 | +0.8797 | 1.02588 | **0.00167** |
| freeze `lumi` | 3 | 1.0024 | -1.1750 | +0.8793 | 1.02713 | **0.00042** |
| freeze `prefiring` | 1 | 1.0016 | -1.1738 | +0.8807 | 1.02727 | **0.00029** |
| freeze `pileup` | 1 | 1.0014 | -1.1736 | +0.8811 | 1.02735 | **0.00020** |
| freeze `btag` | 6 | 1.0039 | -1.1761 | +0.8786 | 1.02737 | **0.00019** |
| freeze `muon` | 3 | 1.0026 | -1.1754 | +0.8798 | 1.02763 | **†0.00007** |
| freeze `STX_qqH` | 8 | 0.9996 | -1.1720 | +0.8832 | 1.02762 | **†0.00006** |
| freeze `jes` | 21 | 1.0019 | -1.1748 | +0.8803 | 1.02755 | **0.00000** |

`†` marks a frozen fit that came out marginally **wider** than nominal, which is
impossible in exact minimisation -- numerical noise, read as zero.

- **`bkg_fit_params` is the whole result**: 0.545, 65x the next group. Freezing it
  halves sigma, 1.028 -> 0.482.
- Only `theory_xs_br` (0.0085), `sig_shape` (0.0027) and `pdf_alpha_s` (0.0017)
  are above the noise. The bottom seven are <= 0.0004.
- **`jes` is exactly 0.00000 across 21 nuisances.** Expected: the background is fit
  from data, so JES only moves the signal shape and that is already in `sig_shape`.
- No total-systematic figure. The union fit (every group frozen) returns invalid
  errors at both range edges because `--robustFit 1` has nothing left to profile,
  and a non-robustFit union would not share minimiser settings with the rows above.

### ggH-specific gotchas

1. **`--robustFit 1` is mandatory.** Without it the -1sigma error fails and combine
   reports the range boundary -- on default, strategy 0 and strategy 2 alike. The
   opposite of the VBF workspace, where robustFit was the thing that failed.
2. **`r=0,2` is too narrow for the sumExp Asimov.** sigma is large and asymmetric
   (-1.172/+0.883), so -1sigma sits at r = -0.17 and a floor at 0 truncates it.
   `r=-2,4` and `r=-5,5` give identical results, so -2,4 is enough.
3. **The Asimov choice moves everything.** On the workspace's stored index 2
   (FEWZxBern) nominal sigma is 0.617 and the significance 1.671; on sumExp it is
   1.028 and 1.557. Numbers from the two are not comparable -- always state which.
4. **Do not pass `--setParameters pdf_index_ggh=N` together with
   `--freezeParameters pdf_index_ggh`** -- MultiDimFit fails with no diagnostic
   beyond "WARNING: MultiDimFit failed". Leaving the index free avoids it entirely,
   and keeping the envelope active is the physically sensible choice anyway.
5. **`RooMultiPdf` has no cling dictionary in this env** (`getCurrentPdf` is
   unavailable from python, and `ROOT_INCLUDE_PATH`/`CPLUS_INCLUDE_PATH` do not fix
   it -- glibc headers are missing). Walk `pdf.getComponents()` to see the families
   instead, and get the index-to-family mapping from the analyst rather than
   inferring it.

### Files

`results/ggh_passB/` holds `groups.txt` (the group -> member lists fed to
`--freezeParameters`), `groups.json` and `raw_fits.json` (per-fit `r`, lower and
upper error). `run_ranking.sh` is **not** used for ggH -- it assumes a template
card with a sibling workspace and the VBF rule table; the ggH commands above are
run directly.

---

## Revision 2 (2026-10-12) — nested freezing, and MC stat split from data stat

**Supersedes nothing above; the earlier tables stay as the record of the first
method.** Everything above this heading used *marginal* freezing: each group
frozen on its own, from the unfrozen nominal, contribution
`abs(sym_nominal - sym_group)`. Those are "what would I gain by fixing this group
alone" numbers and they do **not** sum to the total — 85% in sigma, 72% in
variance on 21bin_btagZpt.

This revision adds the **nested** scheme used by ram1123 in
`common_workflow.sh::run_vbf_lhscan()`
(github.com/ram1123/copperheadV2, branch Week_Sep28, L1172-1214). Step *n* freezes
everything step *n-1* froze **plus** one more family, and a component is the
quadrature drop between consecutive steps:

    component_n = sqrt( sym_{n-1}^2 - sym_n^2 )

Because the freezes are cumulative, the components close on the total exactly
(ratio 1.0000 in both channels below) — that is the reason to prefer it. The
systematic families themselves are unchanged from the tables above.

The author's definitions, taken straight off the card:

    SYST=$(awk '$2=="lnN" || $2=="shape" || $2=="param" {print $1}' card.txt | sort -u | paste -sd,)
    RP=$(awk '$2=="rateParam" {print $1}' card.txt | sort -u | paste -sd,)

### VBF — `stage3_datacards_20bin_noFixMass`, Run2

Four steps. Settings as step 3 above (`--algo singles`, `r=-5,5`, `-t -1
--expectSignal 1`, no robustFit).

    combine $C --freezeParameters MH
    combine $C --freezeParameters "MH,$SYST"
    combine $C --freezeParameters "MH,$SYST,rgx{prop_bin.*}"
    combine $C --freezeParameters "MH,$SYST,rgx{prop_bin.*},$RP"

95 named systs, 8 DY rateParams.

| step (cumulative freeze) | r | -1sigma | +1sigma | sym | component | **value** |
|---|---|---|---|---|---|---|
| total (nothing frozen) | 1.0000 | -0.6877 | +0.7657 | 0.72673 | — | — |
| + named systs | 1.0000 | -0.6606 | +0.7361 | 0.69834 | Syst | **0.20112** |
| + MC stat  rgx{prop_bin.*} | 1.0000 | -0.6276 | +0.7000 | 0.66382 | MCStat | **0.21686** |
| + DY rateParams | 1.0000 | -0.6116 | +0.6793 | 0.64546 | DYNorm | **0.15503** |
| remainder = data statistics | — | — | — | 0.64546 | Stat (data) | **0.64546** |

Quadrature sum **0.72673** against a total of **0.72673** — closes to 1.0000.

- **MC stat (0.2169) exceeds all systematics combined (0.2011).**
- Data statistics (0.6455) dominates everything; this measurement is
  statistics-limited, not systematics-limited.
- `rgx{prop_bin.*}` is what isolates MC stat. It cannot come from the card, because
  `autoMCStats` creates `prop_bin*` at `text2workspace` time -- the `awk` on the
  card can never see them.
- Cross-check passed: the explicit union and
  `--freezeParameters MH,allConstrainedNuisances,$RP` both give -0.6120/+0.6789.
  The `allConstrainedNuisances`-includes-`prop_bin*` trap (gotcha 3) is harmless
  **in this ordering only**, because step 3 already froze them.

### ggH — `Run2_BtagZptFix_sept29_2026_ggh_passB`

Three steps, not four. Settings as the ggH section above (`r=-2,4`,
`--setParameters pdf_index_ggh=0`, `--X-rtd MINIMIZER_freezeDisassociatedParams`).

| step (cumulative freeze) | r | -1sigma | +1sigma | sym | component | **value** |
|---|---|---|---|---|---|---|
| total (nothing frozen) | 0.9996 | -1.1720 | +0.8831 | 1.02756 | — | — |
| + 58 card systs | 0.9996 | -1.1726 | +0.8561 | 1.01436 | Syst | **0.16415** |
| + 25 bkg fit params | 1.0001 | -0.4772 | +0.4789 | 0.47808 | BkgModel | **0.89463** |
| remainder = data statistics | — | — | — | 0.47808 | Stat (data) | **0.47808** |

Quadrature sum **1.02756** against a total of **1.02756** — closes to 1.0000.

- **There is no MC-stat step, and that is not the same as MC stat being zero.**
  This model has no `prop_bin*` and no `autoMCStats` line, so `rgx{prop_bin.*}`
  matches nothing and the step would be a no-op. The signal is an analytic DCB
  whose shape parameters (`sigma_subCat*`, `alpha1/2_subCat*`, `n1/2_subCat*`) are
  **constant** in the workspace -- the signal-template statistical uncertainty is
  simply **not represented in the model**. `CMS_hmm_peak/sigma_cat*` are a
  peak/resolution calibration prior, not a stat term. Worth raising separately.
- `bkg_fit_params` (**0.8946**) is the background model, fitted to data, so it sits
  on the data side of the stat split rather than the MC side.
- Systematics are 0.1641 -- a sixth of the background-model term.
- **Settings caveat:** the final all-frozen step had to run **without**
  `--robustFit 1`. With it, both errors come back invalid at the range edges
  (-3.000/+3.000) because robustFit has nothing left to profile. So that one step
  does not share minimiser settings with the two above it. Tried both; the
  robustFit version is kept as `higgsCombine.ggh_r02_bkg_robust...` for the record.

### Files

`results/20bin_noFixMass/nested_vbf.json` and `results/ggh_passB/nested_ggh.json`
hold the per-step sigmas and the components. `nested_breakdown.py` is a generic
summariser for this scheme.

---

## Re-running

    ./run_ranking.sh <output_dir> [card.txt]

`card.txt` defaults to the 21bin_btagZpt Run2 card; the workspace is the sibling
`.root` of the same stem and must already exist (stage3 writes it). The groups are
derived from whichever card is passed, so card and groups cannot drift apart --
and if the card carries a nuisance no rule matches, or the count disagrees with its
kmax, the run stops before any fit.

Verified to reproduce every row bit-identically on 21bin_btagZpt (diff 0.00e+00).
Per-year cards group correctly too -- 54 nuisances each on 21bin, 48 on 20bin -- so
a per-year ranking is just a different argument.

## Files

| File | What |
|---|---|
| `nuisance-groups.txt` | The grouping, readable listing |
| `nuisance-groups.json` | Same, machine-readable + per-nuisance process/year footprints |
| `derive_nuisance_groups.py` | Regenerates both from a card; asserts count == kmax |
| `run_ranking.sh` | Steps 1-6 |
| `summarize_ranking.py` | Step 7 + the table |
| `ranking.json` / `raw_fits.json` | Results / raw per-fit r and errors |
| `final-summary.md` | The result and what is open |

### Per-family nested chain — old row list, plus MC stat vs data stat

Same families as the marginal tables above, but **nested**: each step freezes all
the previous families plus one more, so the components partition the total instead
of overlapping. MC stat gets its own step where the model has one.

Two things to keep straight:

- **The chain order and the display order are different.** Components are computed
  in a fixed nesting order (descending marginal delta, recorded as `order` in
  `nested_chain_*.json`); the tables below are then **sorted by component size**,
  largest systematic first, with data statistics pinned last. Re-sorting the
  display does not change the values.
- **Nesting is order-dependent.** Whichever family is frozen first absorbs the
  shared correlation, so a different order would redistribute the components.
- Components are far larger than the marginal deltas (`jes` 0.164 vs 0.019)
  because they are quadrature pieces of sigma, not shrinkages of it.

MC stat is counted as a systematic here, per the analyst.

#### VBF — `stage3_datacards_20bin_noFixMass`

| rank | family | N | **component** |
|---|---|---|---|
| 1 | `MCStat` | 8 | **0.20978** |
| 2 | `jes` | 27 | **0.16371** |
| 3 | `bkg_DY_norm` | 8 | **0.10665** |
| 4 | `lumi` | 3 | **0.09273** |
| 5 | `pdf_alpha_s` | 18 | **0.08550** |
| 6 | `LHEFac_norm` | 8 | **0.06974** |
| 7 | `prefiring` | 3 | **0.06423** |
| 8 | `muon` | 12 | **0.04199** |
| 9 | `pileup` | 3 | **0.03691** |
| 10 | `STX_qqH` | 10 | **0.02595** |
| 11 | `btag` | 10 | **0.02502** |
| 12 | `zpt` | 1 | **0.00133** |
| — | **data statistics** | — | **0.64546** |

All systematics incl. MC stat, in quadrature: **0.33393**. With data statistics
**0.72673** that gives **0.72673** against a total of **0.72673** -- ratio 1.0000.

`MCStat` is the `rgx{prop_bin.*}` step. It is the largest single systematic
component, ahead of `jes`.

#### ggH — `Run2_BtagZptFix_sept29_2026_ggh_passB`

| rank | family | N | **component** |
|---|---|---|---|
| 1 | `bkg_fit_params` | 25 | **0.90745** |
| 2 | `theory_xs_br` | 3 | **0.06776** |
| 3 | `pdf_alpha_s` | 2 | **0.03250** |
| 4 | `sig_shape` | 10 | **0.02178** |
| 5 | `lumi` | 3 | **0.01290** |
| 6 | `prefiring` | 1 | **0.00466** |
| 7 | `muon` | 3 | **0.00297** |
| 8 | `pileup` | 1 | **0.00153** |
| 9 | `STX_qqH` | 8 | **0.00091** |
| 10 | `btag` | 6 | **0.00000** |
| 11 | `jes` * | 21 | **0.00000** |
| — | **data statistics** | — | **0.47808** |

All systematics in quadrature: **0.91093**, total **1.02876** -- ratio 1.0012.

- No MC-stat row: this model has no `prop_bin*`/`autoMCStats` (section 4.1).
- `*` the final chain step freezes everything, so `--robustFit 1` has nothing left
  to profile and was dropped for that one fit. The switch moves sigma by +0.0026,
  which is why that component floors to zero and the ratio is 1.0012 rather than
  1.0000. The VBF chain uses one setting throughout and closes exactly.

**On "closure".** For a nested chain the ratio is **algebra, not a test**:
consecutive differences telescope, `sum(sigma_{n-1}^2 - sigma_n^2) = sigma_0^2 -
sigma_N^2`, and adding the remainder returns `sigma_0^2` identically. It deviates
from 1 only if the fits are mutually inconsistent -- which is what ggH's 1.0012 is
reporting. The marginal method's shortfall (0.8499 on 21bin, 0.9060 on 20bin) is
different and physical: uncertainty shared between families, attributable to none
of them alone.

### Combined VBF + ggH — MC-stat contribution

Cards combined with `combineCards.py`. The **21 JES nuisances share names** between
the two cards and are therefore correlated; every other systematic has
channel-specific names (`lumi2016` vs `lumi_13TeV_2016`, `btag_bc_2017` vs
`wgt_btag_bc_2017`, ...) and stays uncorrelated. This correlation model was chosen
deliberately -- it is what the cards already say. `kmax 132 = 95 + 58 - 21`, 13
channels, 292 ModelConfig nuisances of which **160 are `prop_bin*`**, all from the
VBF templates. **MC stat is a VBF-only effect**; ggH contributes none.

    cd results/comb_vbf_ggh
    combineCards.py vbf=<VBF card> ggh=ggh_abs.txt > comb_vbf_ggh.txt
    text2workspace.py comb_vbf_ggh.txt -o comb_vbf_ggh.root -m 125

`ggh_abs.txt` is the ggH card with its `my_workspace/...` shape paths rewritten to
absolute -- they are the only relative ones, and `combineCards.py` does not fix
them up.

#### Validation first

| | expected significance |
|---|---|
| VBF alone | 1.51315 |
| ggH alone | 1.55664 |
| combined | **2.17096** |
| `sqrt(1.51315^2 + 1.55664^2)` | 2.1709 |

The combination reproduces the independent-channel expectation to four decimals.
**Do not skip this check** -- the first combined workspace I built returned 1.44394,
*below either channel alone*, which is impossible and was caused by gotcha 2 below.

#### The fit

    combine -M MultiDimFit comb_vbf_ggh.root -m 125 --algo singles \
            -t -1 --expectSignal 1 \
            --setParameterRanges r=0,3 \
            --setParameters pdf_index_ggh=0 \
            --robustFit 1 \
            --freezeParameters MH,pdf_index_ggh,BWZ_Redux_a_coeff,BWZ_Redux_b_coeff,\
BWZ_Redux_c_coeff,FEWZxBern_c2,FEWZxBern_c3,FEWZxBern_c4[,<family members>]

Nominal: **r = 1.0000, -0.471262 / +0.489360, sym = 0.480311**.

#### MC statistics, the headline number

| definition | value |
|---|---|
| **marginal** -- `prop_bin*` frozen alone, from the unfrozen nominal | **0.09942** |
| **nested** -- frozen last among the systematics | **0.08149** |

For comparison the VBF-only nested MC stat is 0.20978. It shrinks to 0.081 in the
combination because ggH adds sensitivity that carries no MC-stat term, so the same
absolute template uncertainty is a smaller share of a smaller total.

#### Full chain (sanity check)

| rank | family | N | **component** |
|---|---|---|---|
| 1 | `bkg_fit_params` | 25 | **0.22545** |
| 2 | `MCStat` | 160 | **0.08149** |
| 3 | `jes` | 27 | **0.05732** |
| 4 | `pdf_alpha_s` | 20 | **0.04500** |
| 5 | `theory_xs_br` | 3 | **0.04096** |
| 6 | `bkg_DY_norm` | 8 | **0.03839** |
| 7 | `muon` | 15 | **0.02434** |
| 8 | `LHEFac_norm` | 8 | **0.02368** |
| 9 | `lumi` | 6 | **0.02138** |
| 10 | `btag` | 16 | **0.01737** |
| 11 | `sig_shape` | 10 | **0.01481** |
| 12 | `prefiring` | 4 | **0.01348** |
| 13 | `pileup` | 4 | **0.01221** |
| 14 | `STX_qqH` | 18 | **0.00713** |
| 15 | `zpt` | 1 | **0.00000** |
| — | **data statistics** | — | **0.40281** |

Quadrature sum **0.48031** against a total of **0.48031** -- ratio 1.0000.
All systematics incl. MC stat: **0.26162**, against data statistics **0.40281**.
The combination is still statistics-limited.

#### Three gotchas specific to the combined workspace

1. **`text2workspace.py` segfaults** persisting the ggH `RooMultiPdf` --
   *"no interpreter information for class RooMultiPdf"* -- unless both of these are
   exported. The header is `RooMultiPdfCombine.h`, not `RooMultiPdf.h`, which makes
   the install look broken when it is not:

        export ROOT_INCLUDE_PATH="$E/include"
        export CPLUS_INCLUDE_PATH="$E/x86_64-conda-linux-gnu/sysroot/usr/include:$E/include"

2. **Never pass `--X-rtd MINIMIZER_freezeDisassociatedParams` to the combination.**
   It is *required* for ggH standalone (`MultiDimFit` fails without it) but here it
   also freezes the VBF nuisances: significance 2.17096 -> 1.44394. The ggH-scoped
   replacement is to freeze `pdf_index_ggh` **and** the six coefficients of the two
   inactive background families, as the command above does.

3. **`r` must not go negative.** A parametric background goes non-physical for
   `r < 0` and the error search fails: `r=-5,5` and `r=-2,4` both return a range
   edge on one side. Valid and mutually consistent: `r=0,3` gives -0.471/+0.489 and
   `r=0.2,2.5` gives -0.475/+0.489. `r=0,2.5` clips high, so the upper edge needs
   room too. `-1sigma` sits at 0.529, well clear of the floor at 0.

Driver: `run_combined_mcstat.sh`. Outputs: `results/comb_vbf_ggh/nested_combined.json`,
`families.json`, `steps.txt`.

### Correction (2026-10-16) — the ggH standalone sigma was inflated by 60%

**The ggH numbers in the sections above are wrong and are superseded by this one.**
They are left in place as the record of the first attempt, per the rule that
superseded methodology stays.

**What was wrong.** The ggH standalone fit reported sym = 1.02756. Two independent
checks say it should be ~0.64:

| route | implied sigma_ggH |
|---|---|
| the note's own expected significance, Z = 1.5566 -> 1/Z | 0.6424 |
| the VBF+ggH combination (VBF 0.72673, combined 0.48031) | 0.6400 |
| what was reported | **1.02756** |

**Two causes, both in the fit configuration:**

1. **`r` was allowed negative** (`--setParameterRanges r=-2,4`). With a parametric
   background the pdf goes non-physical for `r < 0` and the error search wanders
   into it. Same failure mode as the combined workspace.
2. **`--X-rtd MINIMIZER_freezeDisassociatedParams` inflates the error** even at
   `r >= 0`: it gives +0.883 on the up side where the targeted freeze gives +0.644.

**The corrected configuration** -- freeze the discrete index and the two inactive
background families by hand instead of using the rtd, and keep `r >= 0`:

    combine -M MultiDimFit datacard_comb_sig_all_ggh.root -m 125 --algo singles \
            -t -1 --expectSignal 1 --setParameterRanges r=0,3 \
            --setParameters pdf_index_ggh=0 --robustFit 1 \
            --freezeParameters MH,pdf_index_ggh,BWZ_Redux_a_coeff,BWZ_Redux_b_coeff,\
BWZ_Redux_c_coeff,FEWZxBern_c2,FEWZxBern_c3,FEWZxBern_c4[,<family members>]

Nominal: **r = 1.0000, -0.646991 / +0.643841, sym = 0.64542**, so
`1/sym = 1.5494` against Z = 1.5566 -- 0.5%. The interval is now nearly
symmetric, where before it was -1.172/+0.883, itself a hint something was wrong.
Range-independent: `r=0,3` -0.647/+0.644, `r=0,4` -0.639/+0.645, `r=0.2,2.5`
-0.639/+0.643.

#### Corrected ggH breakdown

| rank | family | N | **component** | % of total (variance) |
|---|---|---|---|---|
| 1 | `bkg_fit_params` | 25 | **0.42945** | 44.3% |
| 2 | `theory_xs_br` | 3 | **0.06582** | 1.0% |
| 3 | `sig_shape` | 10 | **0.03027** | 0.2% |
| 4 | `pdf_alpha_s` | 2 | **0.02962** | 0.2% |
| 5 | `lumi` | 3 | **0.00885** | 0.0% |
| 6 | `prefiring` | 1 | **0.00414** | 0.0% |
| 7 | `muon` | 3 | **0.00255** | 0.0% |
| 8 | `pileup` | 1 | **0.00136** | 0.0% |
| 9 | `STX_qqH` | 8 | **0.00104** | 0.0% |
| 10 | `btag` | 6 | **0.00063** | 0.0% |
| 11 | `jes` | 21 | **0.00000** | 0.0% |
| — | **data statistics** | — | **0.47808** | 54.9% |

Quadrature sum **0.64747** vs total **0.64542** -- ratio
1.0032; the 0.3% comes from the final all-frozen step, which
still needs `--robustFit 1` off because it has nothing left to profile.

**The conclusion changes.** Data statistics (0.478, 55% of the variance) now
exceeds the background fit (0.429, 44%). Before the correction the
background fit read 0.907 / 78%. All of the excess sat in that one row: the
all-frozen floor was always 0.478, so the inflated nominal was absorbed entirely
by the first step of the chain.

**VBF is unaffected** -- checked, not assumed. Four configurations agree to 0.7%:
`r=-5,5` plain -0.688/+0.766 (the published value), `r=0,3` plain identical,
`r=0,3` robust -0.684/+0.761, `r=-5,5` robust -0.683/+0.761. VBF's own 10% gap
against `1/Z` is genuine non-Gaussianity from the asymmetric likelihood, not a fit
artifact. The VBF tables above stand, as does the combined-fit section.

**The AN note** (`sections/subSec_results_ggh.tex`,
`tab:ggh-nuisance-family-impacts`) has been updated to these values: background fit
0.43 / 44%, data statistics 0.48 / 55%, total 0.65.

Results: `results/ggh_passB/nested_chain_ggh_corrected.json`.

---

## Revision 4 (2026-10-19) — post-fit breakdown of the combined fit, Table-9 format

Revisions 1-3 above are unchanged and still describe the per-channel
rankings. This revision adds one new product: a breakdown of the **combined
VBF+ggH** uncertainty in the format of Table 9 of the Run 2 paper
(`Sirunyan:2020two`), evaluated **post-fit** rather than pre-fit. It supersedes
nothing; the pre-fit combined numbers in "Combined VBF + ggH — MC-stat
contribution" above stand as the pre-fit variant.

### What "post-fit" means here

The two channels define it differently, and both were read from the scripts that
produced the significances in the note:

| | source | what it does |
| --- | --- | --- |
| VBF | `copperheadV2/scripts/produce_significance.sh` L39–118 | `MultiDimFit` over the **sideband-only** card, `assignValueOnly` of those nuisances into the full workspace as a snapshot, `GenerateOnly --snapshotName`, then `Significance --freezeParameters allConstrainedNuisances,<8 DY rateParams>` → 1.68501 |
| ggH | `<ggh datacards>/get_postfit.sh` L15–18 | `GenerateOnly` at pre-fit values, `MultiDimFit --saveWorkspace` **on that same Asimov**, then `Significance --snapshotName MultiDimFit --freezeParameters MH,allConstrainedNuisances` → 1.93 |

Step 4 of both freezes every constrained nuisance, so each published "post-fit
significance" is a stat-only significance evaluated at post-fit nuisance values.
Copied literally it would leave nothing to break down, so this revision takes
steps 1–3 (the post-fit values and the post-fit Asimov) and drops step 4's
freeze, which is the chain's last rung anyway. ggH's snapshot is a fit to an
Asimov it generated itself at pre-fit values and so returns pre-fit values; the
whole 1.56 → 1.93 gain there is the freeze.

**That gain is NOT reproduced by our breakdown, and the cross-check first
written here was wrong.** It used the *combined* `bkg_fit_params` component
(0.22545) where ggH's own (0.42945) belongs. With ggH's own numbers the
configuration "constrained nuisances frozen, background fit parameters
floating" gives σ = sqrt(0.47808² + 0.42945²) = 0.64264, i.e. 1/σ = 1.556 —
essentially the pre-fit 1.5566, not 1.93. So removing the width of the
constrained nuisances cannot explain ggH's published post-fit significance; the
gain would have to come from the *values* the snapshot freezes them at, which we
have not measured. Treat any significance predicted from these σ values as
carrying that caveat.

Hence the asymmetric treatment, which is deliberate and was the user's call:
**VBF nuisances take sideband values, ggH nuisances stay pre-fit.**

**Clarification (same day).** Calling the ggH side "pre-fit" is wrong, and the
framing above is corrected here rather than removed. The ggH channel is post-fit
*by construction*: `workspace_bkg_cat0_ggh.root` carries 786132 observed data
events and background coefficients already at fitted values
(`RooSumTwoExpPdf_a1_coeff` = -0.148, `f_coeff` = +0.261, `a0_subCat0` =
-0.0222), all with `const=0`, so they stay free in every rung and sit at their
data-determined values throughout. There is no pre-fit background to transplant.
What actually stays at nominal is only the ggH *constrained* nuisances, which
have no sideband counterpart, plus the VBF signal-region `prop_bin*`. So the
combined breakdown is post-fit in both channels; the two channels simply reach
that state by different routes -- VBF through the sideband fit, ggH through a
background model that is data-determined to begin with.

### Category assignment

The six rows are hierarchical, as in the paper: total = stat ⊕ syst, and
experimental / theoretical / size-of-simulated-samples decompose syst. They are
**not** all to be added. The paper's own numbers do not close per-side
(0.11⊕0.11⊕0.06 = 0.1667 against a quoted 0.16) because it evaluates each
contribution separately and quotes it per side; a single symmetric column
evaluated by nested freezing closes to 1.0000.

| category | families | parameters |
| --- | --- | ---: |
| Experimental | `bkg_fit_params`, `jes`, `lumi`, `prefiring`, `muon`, `pileup`, `btag`, `sig_shape`, `zpt`, `bkg_DY_norm` | 111 |
| Theoretical | `pdf_alpha_s`, `LHEFac_norm`, `theory_xs_br`, `STX_qqH` | 49 |
| Size of simulated samples | `rgx{prop_bin.*}` (VBF only) | 1 regex |
| **left floating** | `shapeBkg_bkg_ggh_ch1..5__norm` | 5 |

`bkg_DY_norm` (the 8 unconstrained `XSecAndNorm*DY01J/DY2J` `rateParam`s, range
[0.2,5]) is **experimental**, not theoretical, by decision on 2026-10-20. It
stays frozen -- the reference `run_vbf_lhscan` also freezes it, as its own
`DYNorm` row -- only the category changed.

The five ggH background normalizations are **left floating in every rung**,
because a background normalization is directly degenerate with the signal
strength. VBF has no `__norm` variables at all (checked: 0); its entire
background-normalization freedom is the 8 DY `rateParam`s above, which is why
`produce_significance.sh` must name them alongside `allConstrainedNuisances`.

**This was a bug before it was a decision.** The family list carries the
pre-combination spelling `shapeBkg_bkg_ch{1..5}__norm`, but `combineCards.py`
renamed the ggH channels, so the combined workspace has
`shapeBkg_bkg_ggh_ch{1..5}__norm`. Every rung logged `WARNING: cannot freeze
nuisance parameter shapeBkg_bkg_ch1__norm if it doesn't exist!` and left them
free. The first delivered numbers were therefore right by accident. They are now
dropped from the family list explicitly and all rungs re-run with zero such
warnings.

`bkg_fit_params` sits in Experimental by decision. The alternative — counting
free-floating parametric shape parameters as statistical, which is what the
paper's small 0.12 experimental row implies it did — gives stat 0.46 / syst 0.13
instead of 0.40 / 0.27. It was rejected because Ram's `run_vbf_lhscan` gives the
free-floating DY `rateParam`s their own breakdown row (`DYNorm`) rather than
folding them into `Stat`, and the ggH background parameters are free-floating in
exactly the same way.

### Results

Workspace `results/comb_vbf_ggh/comb_vbf_ggh_postfit.root`, snapshot
`CombVBFSBPostfit`. Raw rungs:

| rung | frozen | r | −1σ | +1σ | σ_sym |
| --- | --- | --- | --- | --- | --- |
| s0 nominal | — | 1.0173 | −0.47700 | +0.49784 | 0.48742 |
| s1 | Experimental (111) | 1.0187 | −0.41246 | +0.42485 | 0.41865 |
| s2 | + Theoretical (49) | 1.0001 | −0.40792 | +0.41738 | 0.41265 |
| s3 | + MC stat (= all) | 1.0001 | −0.40014 | +0.40983 | 0.40498 |

| Uncertainty source | Δμ |
| --- | ---: |
| Total uncertainty | 0.48742 |
| Statistical uncertainty | 0.40498 |
| Systematic uncertainty | 0.27124 |
| &nbsp;&nbsp;Experimental | 0.24962 |
| &nbsp;&nbsp;Size of simulated samples | 0.07916 |
| &nbsp;&nbsp;Theoretical | 0.07066 |

Earlier variant, with `bkg_DY_norm` counted as theoretical: Experimental
0.24521, Theoretical 0.08470, same total/stat/syst. s2 and s3 are unchanged by
the recategorisation, as only the s0->s1 split moves.

Closures, both exactly 1.0000: sub-rows/systematic and (syst⊕stat)/total. As
noted in revision 2 this is telescoping algebra, not a validation.

### The ggH post-fit significance: 1.93 IS reproducible on these cards

Measured on `datacard_comb_sig_all_ggh.root` (the card set the ggH ranking uses),
`-M Significance -t -1 --expectSignal 1 --setParameters pdf_index_ggh=0 --rMin 0
--rMax 3`:

| frozen | Z |
| --- | ---: |
| nothing (BASE only) | 1.55660 |
| `allConstrainedNuisances`, background fit parameters left floating -- the freeze list `get_postfit.sh` uses | 1.55643 |
| all 84 group members | 2.09979 |

And that card set's own `datacards/significance.txt` (2026-10-01 23:04) records
**`combined core function 0 post-fit Expected Significance: 1.55659`**, plus
1.64989 for index 1 and 1.67054 for index 2. Our 1.55643 reproduces it to 0.01%.

Consequences:

- freezing the constrained nuisances does nothing in this channel (1.55660 ->
  1.55643), because its systematics are negligible next to the background fit.
  Pre-fit and post-fit coincide here.
- freezing *everything*, background fit parameters included, gives 2.09979, which
  matches the calibration k/sigma = 1.00464/0.47808 = 2.1014 and is the number
  implied by our ggH breakdown's 0.47808 statistical row.
- **CORRECTION.** It was first written here that 1.93 does not come from this
  card set. Wrong. `datacards/significance_postfit.txt` (2026-10-02 02:56) in
  the same directory records `combined core function 0 post-fit Expected
  Significance: 1.92761`, plus 2.51945 for index 1 and 1.91838 for index 2. The
  workspace behind it had the **fit parameters frozen manually at the stage3
  step**, which `get_postfit.sh`'s `--freezeParameters MH,allConstrainedNuisances`
  then inherits.
- reproduced **exactly**, 1.92761, by freezing `allConstrainedNuisances` plus the
  30 fit SHAPE parameters (`bkg_shape_core` + `bkg_shape_smf` + `sig_shape`)
  while leaving the 5 `shapeBkg_bkg_ch*__norm` floating. Adding those 5 gives
  2.09979. So the entire 1.93 vs 2.10 gap is whether the background
  normalizations are frozen, and 1.93 is the value with them free -- consistent
  with leaving them free in the breakdown.
- the combined post-fit significance is **2.56074** with the same convention
  (and identically 2.56074 whether or not the 5 misspelled norm names are in the
  freeze list, since they never matched anything). Naive combination
  sqrt(1.92761^2 + 1.68501^2) = 2.56026, agreeing to 0.02%. So the agreement
  with `tab:summary_data_events_and_significance`'s 2.56 is real, not the
  coincidence claimed one revision earlier.

### Post-fit significance of the combination

Measured on the same workspace and snapshot, `-M Significance -t -1
--expectSignal 1 --snapshotName CombVBFSBPostfit --setParameters
pdf_index_ggh=0 --rMin 0 --rMax 3`:

| frozen | Z |
| --- | ---: |
| all 166 nuisance entries (the `p3_all` union) | **2.56074** |
| nothing (BASE only) | 2.17412 |

Three things this settles:

- 2.56074 reproduces the **2.56** already tabulated for ggH+VBF post-fit in
  `tab:summary_data_events_and_significance`, which was built as the naive
  combination sqrt(1.93^2 + 1.68501^2) = 2.5621. Agreement 0.005%.
- the post-fit nominal 2.17412 sits on top of the pre-fit nominal 2.17096
  (0.15%), i.e. moving the VBF nuisances to their sideband values does not shift
  the profiled significance -- consistent with the total uncertainty moving only
  from 0.48031 to 0.48742.
- implied sigma from the calibration k = Z*sigma = 1.04273 (combined pre-fit) is
  1.04273/2.56074 = 0.40720, against the `s3` rung's 0.40498, 0.5%. The
  prediction made before the run, 2.575, was 0.6% high.

### Gotcha 8 — the post-fit nominal rung clips, and widening r does not fix it

On the post-fit Asimov the all-floating rung's `robustFit` upward error scan
dies (`Maximum failed steps (max=5) reached and minimization failed at r =
1.42723`) and combine reports the distance to the range edge: `+1.983` at
`r=0,3` and `+3.981` at `r=0,5`. Only the nominal rung is affected — every
frozen rung has the ggH background fit parameters fixed, which removes the
r ↔ background-shape degeneracy the scan fails on.

What was ruled out, each by a control:

| control | result | conclusion |
| --- | --- | --- |
| rewritten workspace, no snapshot | `−0.471/+0.489`, identical to the pre-fit chain | the file is intact |
| pre-fit Asimov read back via `--toysFile` | `+2.000`, clipped | not caused by the sideband values |
| `r=0,5` | `+3.981`, clipped | not a range problem |
| strategy 2 | `+1.980`, clipped | not strategy alone |
| `--robustFit 0` | `−1.017/+0.453`, now clipped at the **lower** edge | MINOS is no better |
| `--cminFallbackAlgo Minuit2,Migrad,0:0.1` | `+1.983`, clipped | fallback does not help |
| **`--setRobustFitTolerance 0.2 --cminDefaultMinimizerStrategy 1`** | **`−0.477/+0.498`** | **the fix** |
| `--algo grid --points 100`, crossings at ΔNLL = 0.5 | σ_sym 0.48259 vs 0.48742 | independent confirmation, 1.0% |

`postfit_breakdown.py` hard-fails on an interval that looks like a range edge,
which is how the clipped rung was caught rather than published.

### Files

| file | role |
| --- | --- |
| `mk_postfit_ws.C` | sideband → combined snapshot transplant, `assignValueOnly`, current values left pre-fit |
| `run_combined_postfit.sh` | driver, with gotcha 4 in the header |
| `postfit_breakdown.py` | rung reader + nested differences + range-edge guard |
| `results/comb_vbf_ggh/sb_postfit_values.txt` | the 95 transplanted values |
| `results/comb_vbf_ggh/postfit_rungs.tsv` | per-rung freeze unions |
| `results/comb_vbf_ggh/postfit_breakdown.json` | final numbers |

Written into the note as `tab:comb-nuisance-family-impacts` and
`tab:comb-nuisance-group-definitions` in `sections/sec_summary.tex`.

### Per-channel tables given Table-9 summary rows (2026-10-20)

`tab:nuisance-family-impacts` (VBF) and `tab:ggh-nuisance-family-impacts` (ggH)
each gained a summary block above the statistical row, named per Table 9:

| row | VBF | ggH |
| --- | ---: | ---: |
| Systematic uncertainty | 0.33 (21%) | 0.43 (45%) |
| Size of simulated samples | 0.21 (8%) | -- |
| Statistical uncertainty | 0.65 (79%) | 0.48 (55%) |
| **Total uncertainty** | **0.73** | **0.65** |

Definition used for the new row is the paper's own: systematic =
sqrt(total^2 - stat^2). VBF gives 0.33393, identical to the quadrature of its 12
families (closure 1.0000). ggH gives 0.43359 against 0.43664 from its families,
a 0.7% gap, because the ggH chain closes only to 1.0070 -- the table quotes the
quadrature-difference value so that its own total and statistical rows close.
Both channels' systematic and statistical percentages sum to exactly 100%, the
`%` column being a fraction of the total variance.

`Size of simulated samples` was **moved out of** the VBF ranked list into the
summary block (user's call), so that list now has 11 families, not 12; the group
definitions table still defines 12 groups. ggH's row is a dash, not 0.00, since
the channel has no MC statistical component at all.

Source of the VBF numbers: `results/20bin_noFixMass/nested_chain_vbf.json`
(MCStat 0.20978), **not** `nested_vbf.json`, which carries a different chain with
MCStat 0.21686 -> 9% and would not match the table's 8%.

### Combined significances written into the note (2026-10-20)

All on `comb_vbf_ggh[_postfit].root`, `-M Significance -t -1 --expectSignal 1
--setParameters pdf_index_ggh=0 --rMin 0 --rMax 3`, freeze union = all families
plus `rgx{prop_bin.*}`; the post-fit rows add `--snapshotName CombVBFSBPostfit`.

| row | frozen | direct combined fit | naive sqrt(Z_ggH^2+Z_VBF^2) | agreement |
| --- | --- | ---: | ---: | ---: |
| pre-fit | none | 2.17097 | 2.17089 (1.5566 + 1.5132) | 0.004% |
| pre-fit stat-only | all | **2.59010** | 2.59010 (1.92761 + 1.73001) | exact |
| post-fit | all | 2.56074 | 2.56026 (1.92761 + 1.68501) | 0.02% |
| post-fit | none | 2.17412 | -- | -- |

`tab:summary_data_events_and_significance` updated: the two dashes in the Run 2
UL rows are filled with ggH pre-fit stat-only **1.93** and ggH+VBF pre-fit
stat-only **2.59**, and the caption now says the UL ggH+VBF significances come
from a simultaneous fit rather than the naive formula. The commented-out older
table in the same file already carried 2.59 for that cell, which corroborates it.

ggH's pre-fit stat-only equals its post-fit (both 1.92761) because freezing its
constrained nuisances at pre-fit or at post-fit values gives the same answer --
measured 1.55660 vs 1.55643 unfrozen. The caption states this so the repeated
1.93 does not read as a copy-paste error.

## Sanity check against the previous analysis (2026-10-20)

Read-only comparison, no change to the workflow. Sources: `ref_material/HMuMu/AN-19-124.pdf`
chapters 6-7 (section 7.5 splits uncertainties into 7.5.1 experimental and 7.5.2
theory) and the Pisa Run 2 VBF datacard, 97 nuisances, whose observation row
2114+2305+2751+3113+3389+3945 = 17617 identifies it as the HIG-19-006 VBF card.

### Categorisation: no mismatches

Every family we group is categorised the same way AN-19-124 categorises it.
Two points that confirm decisions taken here:

- **DY normalization is experimental.** Section 7.5.1 lists "Drell-Yan
  contribution from pileup and noise" as an experimental item and describes our
  two components exactly, whose normalizations "are allowed to float freely in
  the signal extraction, constrained only by low score events of the VBF-SR and
  VBF-SB". Confirms moving `bkg_DY_norm` out of Theoretical.
- **Background normalizations are unconstrained.** Figure 6.29's ggH impact plot
  shows `shapeBkg_bkg_ch1..5__norm` tagged *Unconstrained*. Confirms leaving them
  floating.

One convention difference, not an error: AN-19-124 puts "Statistics of
simulation ... autoMCStat" inside 7.5.1 Experimental, while we give "Size of
simulated samples" its own origin row because that is what paper Table 9 does.
Under their convention our Experimental 0.24962 and MC size 0.07916 would merge
to 0.26246.

### Inventory: six families they had that our cards do not

| family | their names | count | ours |
| --- | --- | ---: | ---: |
| Jet energy resolution | `JEReta{0,1,2,3}pt{0,1}` | 6 shape | 0 |
| Quark-gluon likelihood | `QGLweight{DY,EWK,Hmm,ST,TT,WW,WZ}` | 7 shape | 0 |
| Parton shower | `SignalPartonShower`, `EWKZjjPartonShower` | 2 shape | 0 |
| DY model shape (2016 vs 2017/18) | `DYModel` | 1 shape | 0 |
| non-DY process cross sections | `XSecAndNorm*{EWK,Hmm,ST,TT,WW,WZ}` | 20 lnN | 0 |
| signal QCD scale as a shape | `LHEFacHmm`, `LHERenHmm` | 2 shape | 0, we use `QCDscale_ggH/qqH` lnN |

Section 7.6.2 calls parton shower of the signal and of EWK Zjj production the
**leading theoretical uncertainty**, so our Theoretical row of 0.07066 is very
likely understated.

### ggH STXS: absent there too, and that is deliberate

The Pisa card carries the 10 VBF STXS names (`THU_VBF_JET01`, `Mjj60`, `Mjj120`,
`Mjj350`, `Mjj700`, `Mjj1000`, `Mjj1500`, `PTH25`, `PTH200`, `Yield`) --
identical to ours -- and **no** `THU_ggH_*`, although section 7.5.2 lists "ggH
STXS stage 1" as a theory uncertainty. The reading is that ggH THU was dropped
because it barely moves the fit, and our own numbers agree: `STX_qqH` is 0.00104
of 0.64542 in ggH standalone (0.2%) and 0.00713 of 0.48031 in the combination
(1.5%). So this item is **not** a gap, and an earlier claim here that the Pisa
card has no STXS at all was made from a truncated copy and is withdrawn.

### Two treatment differences, in our favour and against

- **DY normalization freedom.** Pisa floats only the 01J component
  (`XSecAndNorm201{6,7,8}DY01J`, `rateParam [0.2,5]`) and *constrains* the 2J
  component with `lnN` (e.g. `XSecAndNorm2016DY2J` = 1.12148). We float all 8,
  both 01J and 2J across 4 eras, as `rateParam [0.2,5]`. We therefore give the DY
  background more normalization freedom than the previous analysis did.
- **`zpt`**: 0 occurrences in the Pisa card, 1 in ours -- an addition.

Matching, with different decompositions: JES (their 21 `jes*` vs our 27, the
extra 5 being 2016APV plus `Total`), b tagging (9 `btagEventWeight_*` vs our 10
`btag_bc/light_*`), PDF (9 `PDFX{0,1,2}{ggH,vbfHmm,DY}` vs our 18
`pdf_alpha_s_unc_*` plus `pdf_Higgs_gg/qq`), muon scale (`MuScale` vs
`mu_roccor*`), `puWeight`, `PrefiringWeight`, `lumi`, `BR_mm` = 1.0123 in both,
and autoMCStats present in both (6 lines there, 8 here).

## Revision history

| rev | date | method | where in this file |
|---|---|---|---|
| 6 | 2026-11-05 | **the post-fit Asimov's global observables were never re-centred**: `mk_postfit_ws.C` transplants 95 sideband nuisance values but leaves every `<name>_In` at 0, so the combined post-fit fit returns r_hat = 1.01727 instead of 1.000. Affects only the combined post-fit NLL + breakdown; every all-frozen significance is bit-identical (measured) | Revision 6 section below |
| 5 | 2026-11-02 | **operational companion**: workspace + command + expected number for all three breakdowns and, newly, all four NLL curves (ggH, VBF nominal, VBF stat-only, combined pre/post-fit). Adds gotcha 9: `--algo grid` silently omits non-converged points and the gap lands on the combined fit's upper 68% crossing, which also weakens revision 4's grid cross-check | Revision 5 section below |
| 1 | 2026-10-01/02 | marginal (subtraction): each family frozen alone from the unfrozen nominal, `delta_sym = abs(sym_nominal - sym_family)`. Gives per-family rankings, but the components do **not** sum to the total (85% in sigma, 72% in variance) | the `MultiDimFit results` table (VBF) and the ggH `Results` table |
| 4 | 2026-10-19 | **post-fit breakdown of the combined fit** in the Run 2 paper's Table 9 format: VBF nuisances set to the H-sideband fit values both for Asimov generation and when frozen, ggH kept pre-fit; six hierarchical rows, total 0.48742. Needed `--setRobustFitTolerance 0.2 --cminDefaultMinimizerStrategy 1` for the nominal rung | Revision 4 section above, and `sections/sec_summary.tex` in the note |
| 3 | 2026-10-16 | **ggH standalone sigma corrected** 1.02756 -> 0.64542: `r` must stay >= 0 and the `--X-rtd` must be replaced by a targeted index+idle freeze. Data statistics overtakes the background fit. | the correction section above |
| 2 | 2026-10-12 | **nested (cumulative) freezing**, after Ram's (ram1123) `run_vbf_lhscan()` — github.com/ram1123/copperheadV2 `Week_Sep28` L1172-1214. Step *n* freezes everything step *n-1* froze plus one more family; components close on the total exactly (1.0000) and MC stat separates from data stat. **Preferred method.** | `Revision 2` |

Both are kept on purpose. Nested is the one to quote; marginal answers the
different question "what would I gain by fixing this one family alone", which is
what you want when deciding where to spend effort.

Two things inside revision 1 were corrected rather than kept as variants, and are
recorded here so the dead ends are not re-walked:

- the delta definition changed from quadrature `sqrt(sym_nom^2 - sym_fam^2)` to
  linear `abs(sym_nom - sym_fam)`. The tables above carry the linear form;
  `ranking.json` carries both, as `delta_sym` and `delta_quad`.
- `allConstrainedNuisances` was first used as a single stat reference. It silently
  freezes the `prop_bin*` MC-stat parameters too, which inflated the total
  systematic by 35% (0.333 vs the correct 0.247 on 21bin_btagZpt). See gotcha 3.
  It is safe in revision 2's ordering only because the MC-stat step comes first.

`run_ranking.sh` + `summarize_ranking.py` implement revision 1;
`nested_breakdown.py` is the revision 2 summariser.

---

## Revision 5 (2026-11-02) — reproducing the breakdowns and the NLL curves

Nothing in revisions 1-4 is superseded here. This section is the operational
companion to them: for each of the three fits, which workspace, which command,
where the output lands, and what number to expect. It also records the NLL
curves, which revisions 1-4 did not cover at all, and one new failure mode
(gotcha 9) that affects the combined grid scans and silently affected the
grid cross-check quoted in revision 4.

### The three workspaces

| fit | workspace | built by |
|---|---|---|
| ggH | `output/bdt_Run2_NanoV15_May24_2026_all/stage3/all/Run2_BtagZptFix_sept29_2026_ggh_passB/impacts_nll/nll/datacard_comb_sig_all_ggh.root` | `combineCards.py` over the 5 `datacard_cat*_ggh.txt`, then `text2workspace.py` |
| VBF | `/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_BtagZptFix_sept29_2026/stage3_datacards_20bin_noFixMass/score_Run2_NanoV15_BtagZptFix_sept29_2026/HMuMu_13TeV_Run2.root` | stage3; the same directory holds that channel's `significance.txt` |
| ggH + VBF | `results/comb_vbf_ggh/comb_vbf_ggh.root` (pre-fit) and `comb_vbf_ggh_postfit.root` (+ snapshot `CombVBFSBPostfit`) | `comb_vbf_ggh.txt` = `combineCards.py` of the two, then `mk_postfit_ws.C` for the post-fit twin |

The VBF workspace is the one shared by both the breakdown and the NLL curves;
`.agent-system/tasks/vbf-impacts-run2-20bin/results/HMuMu_13TeV_Run2.root` is a
byte-identical copy (674601 B), so either path gives the same answer.

### A. Uncertainty breakdowns

All three use revision 2's nested (cumulative) freezing. Quote these, not
revision 1's marginal numbers.

| fit | run | summariser | total | stat | syst |
|---|---|---|---|---|---|
| ggH | `run_ranking.sh` with the revision-3 corrections | `nested_breakdown.py` | 0.64542 | 0.47808 | 0.43359 |
| VBF | `run_ranking.sh` | `nested_breakdown.py` | 0.72673 | 0.64546 | 0.33393 |
| ggH + VBF | `run_combined_postfit.sh` | `postfit_breakdown.py` | 0.48742 | 0.40498 | 0.27124 |

`syst` is always `sqrt(total^2 - stat^2)`, the paper's own "difference in
quadrature"; it is not a rung in its own right. Stored at full precision in
`results/<dir>/nested_chain_*.json`. The combined fit's three disjoint origins
are Experimental 0.24962, Size of simulated samples 0.07916, Theoretical
0.07066 -- these partition the systematic and must **not** be added to it.

Two things that will bite on a re-run, both already in this file: the ggH
breakdown needs `r >= 0` plus a targeted index+idle freeze instead of
`--X-rtd` (revision 3, or the total comes out 1.02756 instead of 0.64542), and
`allConstrainedNuisances` silently includes `prop_bin*` (gotcha 3), which is
only safe because the MC-stat rung comes first.

### B. NLL curves

All three are expected curves on an internal Asimov (`-t -1 --expectSignal 1`),
`--algo grid --points 100`, plotted with `plot1DScan_nopoints.py --no-points`
and `--translate poi_translate.json` so the POI prints as `#mu`. The
`--no-points` flag exists because the environment's own `plot1DScan.py` draws
`"AP"` (markers); the `ref_material` copy draws `"AC"`. `TSpline3` passes
through every point, so one bad point is a visible kink -- and a missing point
is an invisible one (gotcha 9).

| fit | script | output | sigma_sym | vs breakdown total |
|---|---|---|---|---|
| ggH | `scripts/get_significance.sh` (tracked) NLL block | `.../impacts_nll/nll/nll_scan_idx0_100pts_clean.pdf` | 0.64098 | 0.64542, **-0.69%** |
| VBF nominal | `run_vbf_nll.sh` | `results/20bin_noFixMass/nll_scan_vbf.pdf` | 0.72555 | 0.72673, **-0.16%** |
| VBF stat-only | `run_vbf_nll.sh` | `results/20bin_noFixMass/nll_scan_vbf_statonly.pdf` | 0.64549 | 0.64546 (stat), **+0.005%** |
| ggH + VBF post-fit | `run_combined_nll.sh` + `clean_nll_points.py` | `results/comb_vbf_ggh/nll_scan_comb_postfit.pdf` | 0.49124 | 0.48742, **+0.78%** |
| ggH + VBF pre-fit | `run_combined_nll.sh` | `results/comb_vbf_ggh/nll_scan_comb_prefit.pdf` | 0.48264 | 0.48000 (pre-fit singles), **+0.55%** |

`best_idx="0"` (sumExp) in the ggH and combined scripts: **a separate study is
required to obtain the pdf index with the best fit** -- do not infer it from
the envelope scan, which records no index (see the revision-4 retraction).

For the VBF stat-only curve the 8 DY `rateParam`s must be named explicitly.
They are unconstrained, so `allConstrainedNuisances` does not reach them, and
without them the curve keeps the DY normalization freedom and comes out wide.

The combined curve uses `run_combined_postfit.sh`'s `$C` **verbatim**, with only
`--algo singles` swapped for `--algo grid --points 100` and `--saveNLL` added,
so the curve and the table come from one configuration. That carries
`--robustFit 1 --setRobustFitTolerance 0.2 --cminDefaultMinimizerStrategy 1`.
Both forms are produced: `prefit` (no snapshot, reproduces the pre-fit singles
control `n00.log`, -0.471/+0.489, sigma_sym 0.4800) and `postfit`
(+ `--snapshotName CombVBFSBPostfit`, the row the note quotes, singles
-0.477/+0.498, sigma_sym 0.48742 -> 0.49).

### Gotcha 9 — grid scans silently drop points, and the gap lands on the crossing

`--algo grid` writes only the points whose profiled fit converged. A failed
point is **absent from the tree**, not flagged, so `plot1DScan` splines straight
across it and reports a crossing that rests on no data. Always count the
entries before trusting an interval:

```python
t = ROOT.TFile.Open(f).Get("limit")        # keep the TFile handle alive,
print(t.GetEntries())                       # or PyROOT garbage-collects it
```

Measured on these workspaces, at `--cminDefaultMinimizerStrategy 0`:

| scan | points written | gap |
|---|---|---|
| combined, pre-fit | 70 / 100 | r = 1.365 .. 1.575, **straddling the upper 68% crossing at 1.503** |
| combined, post-fit (`g0_grid`, revision 4's cross-check) | 62 / 100 | the same gap, r = 1.365 .. 1.575 |
| ggH | 83 / 100 (after 2 kink points removed by hand) | away from the crossing |
| VBF, both curves | 100 / 100 | none |

So the failure is specific to the combined workspace above r ~ 1.37, where the
ggH background shape and `r` are degenerate -- the same degeneracy that made
the nominal rung's `robustFit` error scan die in gotcha 8. Two consequences:

1. A strategy-0 combined grid scan gives sigma_sym 0.48881 from an interpolated
   upper crossing. It happens to round to the 0.49 in the note, which is why
   this nearly went unnoticed; the agreement is not evidence of anything.
2. **Revision 4's grid cross-check is weaker than its comment claims.** The
   0.48259 quoted there against singles' 0.48742 came from the 62-point scan,
   with the same gap on the crossing. The 1.0% agreement stated in
   `run_combined_postfit.sh` should be read as "the two estimators are close",
   not as a validated 1.0%.

**Correction (same day): `--cminDefaultMinimizerStrategy 1` does NOT close the
gap.** That was claimed here first and is wrong. Strategy 1 wrote 67 points, not
70 -- slightly fewer -- and merely moved the gap to r = 1.395 .. 1.545, which
still contains the crossing. What it did do is bring the value much closer to
the singles reference (0.48113 vs 0.48000, +0.24%, against strategy 0's
+1.84%), so it is still the right setting; it is just not the fix for coverage.
The gap was inferred to be closed from the log's `Point 99/100` line, which
counts points *attempted*. Count the TREE.

The actual fix is a targeted dense refill. Keep the range at 0,3 and raise
`--points` so the grid is finer, then restrict which of them run, so the
deltaNLL reference (the initial fit at r ~ 1) is unchanged:

```
combine ... --algo grid --points 300 --firstPoint 130 --lastPoint 169 ...
# 300 points over [0,3] -> spacing 0.01, r_i = 0.005 + 0.01*i, so i=130..169 is r = 1.305..1.695
```

then `hadd` that onto the 100-point scan. Measured effect, pre-fit: the crossing
moves 1.4880 -> 1.4911 and sigma_sym 0.48113 -> 0.48264, with real points at
1.475/1.495/1.515 bracketing it instead of a 0.15-wide hole.

**`hadd` introduces a duplicate** at any r present in both scans (r = 1.695
here), with two different deltaNLL values. `TSpline3` cannot represent that, so
de-duplicate on r before plotting -- `clean_nll_points.py` does.

### Gotcha 10 -- one outlier threshold cannot serve both halves of an NLL curve

Removing kink points needs a scale-aware criterion. The residual against a
local quadratic (6 nearest neighbours; a straight line through the neighbours
counts real curvature as deviation and cascades) has a median of 0.0067 on the
combined post-fit curve, but the scatter is not uniform: ~0.0007 near the
minimum and ~0.04 on the upper branch. Consequences, all measured:

| cut | points kept | result |
|---|---|---|
| flat `abs(res) > 0.035` | 57 | upper branch clean, but a zigzag pair survives at r = 0.645/0.675 |
| flat `abs(res) > 0.02` | 48 | kills the zigzag, but eats the upper branch -- curve **truncated at r = 1.57**, 95% crossing lost |
| `abs(res) > 0.03` **OR** `abs(res)/localMAD > 4` | 53 | both clean, full range retained |

The 0.645/0.675 pair is only 0.025 in absolute terms but 34x the local scatter;
the upper-branch outliers are 0.17-0.55 absolute but only 1-3x theirs. Hence the
two criteria OR'd, which is what `clean_nll_points.py` implements. Protect
+-0.08 around each 68% crossing and the quoted sigma cannot move: 0.49120 ->
0.49124 across the whole cleaning, a shift of 0.00004.

### Final combined NLL numbers

| curve | points | 68% interval | sigma_sym | reference |
|---|---|---|---|---|
| post-fit (the one the note quotes) | 53 | 0.53728 .. 1.51976 | **0.49124** -> 0.49 | breakdown 0.48742, +0.78% |
| pre-fit | 85 | 0.52578 .. 1.49105 | **0.48264** -> 0.48 | singles `n00.log` 0.48000, +0.55% |

The post-fit curve's minimum is at mu = 1.017, not 1.000: the sideband-pulled
nuisances displace the Asimov minimum, which is also why its asymmetry
(+0.502/-0.480) exceeds the pre-fit's (+0.491/-0.474). **The note's 0.49 is a
post-fit number** (`sec_summary.tex` caption: "expected post-fit"), so the
post-fit curve is the one to put in the note; a pre-fit curve giving 0.48 is
not a discrepancy, it simply has a different reference.

### A copy of the combined outputs now sits with the VBF results

Unlike ggH and VBF, the combination is not a stage3 product -- the cards are
built in-place by `combineCards.py`, so `results/comb_vbf_ggh/` was its only
home. For consistency with the per-channel outputs a full copy (195 files, 13M)
now also lives beside the VBF stage2/stage3 trees:

```
/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_BtagZptFix_sept29_2026/comb_vbf_ggh/
```

That parent directory holds `stage2_histograms/` and every
`stage3_datacards_*/`, including the `stage3_datacards_20bin_noFixMass/` whose
`score_*/significance.txt` is the VBF channel's. The task-directory copy remains
the working one; treat the `/work/projects` copy as the archived result.

### Files added in this revision

| file | what |
|---|---|
| `run_combined_nll.sh` | combined pre-fit + post-fit NLL curves, from `run_combined_postfit.sh`'s `$C` |
| `run_vbf_nll.sh` | VBF nominal + stat-only NLL curves; the record of what was run ad hoc on 2026-11-01 |
| `clean_nll_points.py` | de-duplicates on r and drops kink points, dual criterion, crossings protected |
| `plot1DScan_nopoints.py` | env's `plot1DScan.py` plus `--no-points` (`SetMarkerSize(0)` before the `"AP"` draw) |

## Revision 6 (2026-11-05) — the post-fit Asimov's global observables were never re-centred

- `mk_postfit_ws.C` and its parent `scripts/produce_significance.sh` L76–86 transplant
  95 sideband nuisance values but leave every `<name>_In` global observable at 0.
- The post-fit Asimov is therefore internally inconsistent: the yields were built at
  `theta_hat_SB`, while the auxiliary measurements still assert `theta` was measured as 0.
- Consequence: the profiled combined fit settles between the two and `r` absorbs the
  mismatch — post-fit `r_hat = 1.01727`, against pre-fit `r_hat = 1.00003`.
- The Gaussian penalty at the snapshot point is 0.798 in NLL = **1.597 in 2*deltaNLL**,
  measured against a 68% interval that is defined at 2*deltaNLL = 1.
- The fix is one flag, no recompilation: `--setParameters <name>_In=theta_hat_SB`,
  which combine applies *after* `--snapshotName` and confirms line by line.

### What theta and theta_In mean

The likelihood is `L(mu, theta) = L_data(mu, theta) * prod_i Gauss(theta_In_i | theta_i, 1)`.
`theta_i` is a nuisance in units of its own quoted 1 sigma; `theta_In_i` is the *global
observable* — the value an external measurement returned, and therefore the centre of
the penalty. A nominal calibration means `theta_In = 0`: "the external measurement came
back exactly central". Generating an Asimov means choosing **all** observations, the 292
globals included; leaving them at 0 declares every external measurement nominal while
the yields say otherwise.

### Scope: measured, not argued

Re-centring moves a number only where a **constrained** nuisance is **profiled** on a
**post-fit** Asimov. Everything else in this file is invariant.

| deliverable | Asimov | nuisances during fit | affected |
|---|---|---|---|
| VBF breakdown (`run_ranking.sh`) | pre-fit, no snapshot | profiled | no — `theta = theta_In = 0` already consistent |
| VBF NLL (`run_vbf_nll.sh`) | pre-fit, no snapshot | profiled | no |
| ggH breakdown + NLL | pre-fit | profiled | no |
| VBF post-fit significance (`produce_significance.sh`) | post-fit | `allConstrainedNuisances,rateParams` frozen | **no — measured** |
| combined post-fit significance, all frozen | post-fit | frozen | **no — measured** |
| combined post-fit significance, nothing frozen | post-fit | profiled | **yes — measured, 2.17412 -> 2.14081** |
| combined post-fit NLL curve | post-fit | profiled | yes |
| combined post-fit breakdown, nominal rung | post-fit | profiled | yes |

`run_ranking.sh`'s own header already states the pre-fit case: "On Asimov the nuisances
sit at their nominal values at the best fit, so freezing at prefit is freezing at postfit
and no snapshot two-stage is needed."

### Null controls

Four-arm test per card: globals left at 0 versus `--setParameters` with 95
`<name>_In=theta_hat_SB`. The base arms reproduce the published numbers exactly, which is
what makes the nulls trustworthy.

VBF card `HMuMu_13TeV_Run2_Run2SBPostfit.root`, snapshot `Run2SBPostfit`,
`--toysFile higgsCombine.Run2PostfitAsimov_.GenerateOnly.mH125.123456.root`:

| arm | nuisances | globals at 0 | re-centred | delta |
|---|---|---:|---:|---:|
| A = the production recipe | `allConstrainedNuisances,rateParams` frozen | 1.68501 | 1.68501 | **0, bit-identical** |
| B = hypothetical | profiled | 1.51778 | 1.46869 | -3.2% |

1.68501 is verbatim the `Postfit Significance` row of that directory's `significance.txt`.

`comb_vbf_ggh_postfit.root`, snapshot `CombVBFSBPostfit`:

| arm | nuisances | globals at 0 | re-centred | delta |
|---|---|---:|---:|---:|
| A | `allConstrainedNuisances,rateParams` frozen | 2.29437 | 2.29437 | **0, bit-identical** |
| B | profiled | 2.17412 | 2.14081 | -1.5% |

2.17412 is verbatim the "post-fit / frozen none" row of the combined significance table
above.

### Why an all-frozen significance cannot move

`-M Significance` forms `q0 = 2*(NLL(r=0) - NLL(r_hat))`. With every constrained nuisance
held at the same value in both terms — and `--freezeAllGlobalObs=1` is combine's default,
so the globals are constant too — the product `prod_i Gauss(theta_In_i | theta_i, 1)` is
the *same constant* in numerator and denominator and cancels exactly. The value of
`theta_In` is therefore irrelevant to any all-frozen significance, which is every post-fit
significance quoted in this file. It matters only where the nuisances are free to be
profiled, because then the penalty's centre decides where they settle.

### Gotcha 11: the `--saveToys` Asimov carries no global observables

The generated post-fit Asimov holds three variables only — `mh_ggh`, `CMS_th1x`,
`CMS_channel`. Verified by listing `toys/toy_asimov`'s variable set in
`higgsCombine.CombPostfitAsimov.GenerateOnly.mH125.123456.root`. The globals are read
from the *workspace* at fit time, not from the toy file, so

- a `--setParameters *_In=...` flag has to be repeated on every command in the chain;
  the toy file cannot carry the re-centred point forward, and
- a future script that omits the flag silently reverts to `theta_In = 0` with no warning.

The durable alternative, not taken for this study: after `full.assignValueOnly(*nus)` in
`mk_postfit_ws.C`, loop the nuisances and set `wc->var(TString(v->GetName())+"_In")` to
the same value before `saveSnapshot`, under a new snapshot name so the published 0.48742
stays reproducible.

### Files added in this revision

| file | what |
|---|---|
| `recentre_globals_plan.md` | the plan, the measured scope table, the null controls, and the predictions fixed before running |
| `run_combined_recentred.sh` | the re-centred chain: 100-point NLL grid + the four breakdown rungs + a strategy-0 nominal cross-check, all with `$IN` appended |

### Results of the re-centred chain (2026-11-05)

- **The diagnosis is confirmed**: every rung returns `r_hat = 1.000` (`1.00003` on the
  grid scan), against `1.01727` with the globals left at 0.
- The two fully-frozen quantities are **bit-identical**, as the cancellation argument
  requires: Statistical `0.40498` and Size of simulated samples `0.07916`.
- The total narrows by 0.8% and the systematic by 2.6%; the hierarchy and both closure
  ratios (1.0000) are unchanged.
- **The minimizer fallback flipped**: the published `--setRobustFitTolerance 0.2
  --cminDefaultMinimizerStrategy 1` now *fails* upward, and plain strategy 0 succeeds.
- Cleaning moved the grid interval by 0.008% (0.48585 raw -> 0.48581), so it is purely
  cosmetic here — but `--crossings` must be passed or it deletes the crossing points.

| row | published (globals at 0) | re-centred | delta |
|---|---:|---:|---:|
| `r_hat` | 1.01727 | **1.00003** | — |
| Total uncertainty | 0.48742 | 0.48346 | -0.81% |
| Statistical uncertainty | 0.40498 | **0.40498** | **0** |
| Systematic uncertainty | 0.27124 | 0.26406 | -2.6% |
| Experimental | 0.24962 | 0.24186 | -3.1% |
| Theoretical | 0.07066 | 0.07043 | -0.3% |
| Size of simulated samples | 0.07916 | **0.07916** | **0** |
| NLL curve `sigma_sym` (grid) | 0.49124 | 0.48581 | -1.1% |

Rounded to the note's two decimals: Total 0.49 -> **0.48**, Systematic 0.27 -> **0.26**,
Experimental 0.25 -> **0.24**; Statistical 0.40, simulated-sample size 0.08 and
Theoretical 0.07 are unchanged. No `.tex` has been edited.

#### The width prediction was wrong, and why

The prediction fixed before running was that the width would *widen* about 1.5%, read
off the measured significance drop 2.17412 -> 2.14081. It narrowed instead, by 0.8%.
The extrapolation is invalid because a significance is the *global* drop from `r = 0`
to `r_hat` on a visibly asymmetric curve, while `sigma` is the *local* curvature at
`r_hat`; on this likelihood the two move in opposite directions. Separately, much of
the apparent change is not physics at all: the published 0.48742 is a robustFit
interval recovered with a loosened tolerance, and its own grid cross-check was 0.48259,
against which the re-centred 0.48346 is +0.2%.

#### Gotcha 12: the minimizer fallback is specific to the inconsistent Asimov

Revision 4 needed `--setRobustFitTolerance 0.2 --cminDefaultMinimizerStrategy 1` because
the nominal rung's upward robustFit scan died at `r = 1.42723`. On the re-centred Asimov
that prescription fails the other way -- combine warns "No valid high-error found, will
report difference to maximum of range" and reports `+2.000`, the `r=0,3` edge -- while
plain `--cminDefaultMinimizerStrategy 0` with no tolerance override gives a clean
`-0.47441/+0.49252`. `postfit_breakdown.py`'s own range-edge guard (`max(slo,shi) > 1.5`)
catches this, which is how it was found. Do not carry either setting across without
re-checking the interval.

#### Files written

| file | what |
|---|---|
| `results/comb_vbf_ggh_recentred/postfit_breakdown_rc.json` | the six re-centred rows at full precision |
| `results/comb_vbf_ggh_recentred/nll_scan_comb_postfit_rc.pdf` / `.png` | the re-centred post-fit NLL curve, `mu = 1.000 +0.495/-0.477` |
| `results/comb_vbf_ggh_recentred/STATE` | run log incl. the per-job guard "95 `_In` set" |

`postfit_breakdown.py` gained three optional argv (results dir, nominal-rung file, output
json name); the defaults are revision 4's, so the published command still reproduces
`postfit_breakdown.json` unchanged.

#### What was changed in the note, and what was not (2026-11-05)

- `sections/sec_summary.tex`: signal-strength line `0.27 -> 0.26`, and the breakdown table
  Total `0.49 -> 0.48`, Systematic `0.27 -> 0.26`, Experimental `0.24`.
- The table's source comment now points at `comb_vbf_ggh_recentred/postfit_breakdown_rc.json`.
- `trunk/pics/combined/results/nll_scan_comb_postfit.pdf` replaced in place by the
  re-centred curve; the `\includegraphics` path is unchanged and git keeps the old one.
- **`tab:summary_data_events_and_significance` is untouched** — its three UL columns are
  pre-fit, pre-fit stat-only, and post-fit *all-frozen*, every one of them invariant.
- Re-ran the AN's exact 2.56 freeze union (`p3_all` + `rgx{prop_bin.*}`) both ways:
  **2.56074 vs 2.56074**, bit-identical, so 1.69 / 2.56 in that table stand.

Statistical `0.40`, Size of simulated samples `0.08` and Theoretical `0.07` round to the
same two decimals as before, so those three rows are unchanged as well.

#### Archive sync (2026-11-05)

The archived copy beside the VBF stage2/stage3 trees was refreshed in the same pass, so
it no longer lags the task directory:

```
/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_BtagZptFix_sept29_2026/comb_vbf_ggh/
  HANDOFF.md                  (1550 -> 1730 lines, now carries revision 6)
  recentre_globals_plan.md    (new)
  run_combined_recentred.sh   (new)
  recentred/                  (new, 21 files, 488K -- the full re-centred chain)
```

The task-directory copy remains the working one; treat `/work/projects` as the archived
result. Anything appended to this file after this line has to be copied across again.
