# Plan: re-centre the post-fit Asimov's global observables (2026-11-05)

## Defect

`mk_postfit_ws.C` (and its parent `scripts/produce_significance.sh` L76-86) does
`full.assignValueOnly(*nus)`: it transplants the 95 H-sideband nuisance values
theta_hat_SB into the combined workspace, but leaves all 292 global observables
`<name>_In` at 0. The resulting Asimov is internally inconsistent -- the yields
say "theta was theta_hat_SB", the auxiliary measurements say "we measured 0" --
so the profiled fit settles between the two and r absorbs the difference:
r_hat = 1.01727 instead of 1.000. Total Gaussian penalty at the snapshot point
is 0.798 in NLL = 1.597 in 2*deltaNLL, against a 68% interval defined at 1.

## Scope: MEASURED, not argued

Re-centring moves a number only where a CONSTRAINED nuisance is PROFILED on a
POST-FIT Asimov. Everything else is invariant.

| deliverable | Asimov | nuisances | affected |
|---|---|---|---|
| VBF breakdown (`run_ranking.sh`) | pre-fit, no snapshot | profiled | no -- theta = theta_In = 0 already |
| VBF NLL (`run_vbf_nll.sh`) | pre-fit, no snapshot | profiled | no |
| ggH breakdown + NLL | pre-fit | profiled | no |
| VBF post-fit significance (`produce_significance.sh`) | post-fit | `allConstrainedNuisances,rateParams` frozen | no -- measured |
| comb post-fit significance, all frozen | post-fit | frozen | no -- measured |
| comb post-fit significance, nothing frozen | post-fit | profiled | YES -- measured |
| comb post-fit NLL curve | post-fit | profiled | YES |
| comb post-fit breakdown (nominal rung) | post-fit | profiled | YES |

### Null controls already measured (2026-11-05)

Four-arm test, `--setParameters <name>_In=theta_hat_SB` (95 entries) vs globals
left at 0. Base arms reproduce the published numbers exactly, which is what
makes the nulls trustworthy.

On the VBF card `HMuMu_13TeV_Run2_Run2SBPostfit.root`, snapshot `Run2SBPostfit`,
`--toysFile higgsCombine.Run2PostfitAsimov_.GenerateOnly.mH125.123456.root`:

| arm | nuisances | globals 0 | re-centred | delta |
|---|---|---|---|---|
| A = production recipe | `allConstrainedNuisances,rateParams` frozen | 1.68501 | 1.68501 | 0, bit-identical |
| B = hypothetical | profiled | 1.51778 | 1.46869 | -3.2% |

1.68501 is verbatim the `Postfit Significance` row of that directory's
`significance.txt`.

On `comb_vbf_ggh_postfit.root`, snapshot `CombVBFSBPostfit`:

| arm | nuisances | globals 0 | re-centred | delta |
|---|---|---|---|---|
| A | `allConstrainedNuisances,rateParams` frozen | 2.29437 | 2.29437 | 0, bit-identical |
| B | profiled | 2.17412 | 2.14081 | -1.5% |

2.17412 is verbatim the "post-fit / frozen none" row of HANDOFF's combined
significance table.

Reason arm A is exact: with every constrained nuisance held at the same value in
numerator and denominator, the product of Gauss(theta_In | theta, 1) is the same
constant at r=0 and at r_hat and cancels in the likelihood ratio. The value of
theta_In is therefore irrelevant to any all-frozen significance.

Also measured: the `--saveToys` Asimov carries NO global observables. Its only
variables are `mh_ggh`, `CMS_th1x`, `CMS_channel`, so the globals are read from
the workspace at fit time -- which is why the flag, not the toy file, is the
place to set them.

## Decisions

| area | choice | rationale |
|---|---|---|
| scope | combined post-fit NLL + breakdown only | per the table above |
| mechanism | `--setParameters <name>_In=theta_hat_SB` | one flag, two shell lines; applied after `--snapshotName`, verified |
| globals set | the 95 transplanted names | the other 197 are at theta=0, so theta_In=0 is already consistent |
| minimizer | strategy 0 first, fall back to `--setRobustFitTolerance 0.2 --cminDefaultMinimizerStrategy 1` | a self-consistent Asimov may dissolve the r <-> background degeneracy that killed the upward robustFit scan |
| AN | deltas first, no `.tex` touched | `comb_vbf_ggh_postfit.root` untouched so 0.48742 stays reproducible |

Rejected: baking the globals into `mk_postfit_ws.C` as a new snapshot
`CombVBFSBPostfitRC`. Correct and more durable, but only pays off if the
re-centred Asimov becomes the permanent recipe. The exact one-line change, for
later: after `full.assignValueOnly(*nus)`, loop the nuisances and
`wc->var(TString(v->GetName())+"_In")->setVal(full_value)` before `saveSnapshot`.

## Components

1. `IN=$(awk '{printf "%s_In=%s,", $1, $2}' sb_postfit_values.txt | sed 's/,$//')`
   from `results/comb_vbf_ggh/sb_postfit_values.txt`; assert 95 fields.
2. NLL: `run_combined_nll.sh` with `$IN` appended to `$COMMON`, `-n .cnll_postfit_rc`.
   The PRE-FIT arm is left alone -- it is already self-consistent.
3. `clean_nll_points.py --thresh 0.03 --ratio 4.0 --protect 0.08`, then
   `plot1DScan_nopoints.py --no-points --translate poi_translate.json`.
4. Breakdown: `run_combined_postfit.sh` with `$IN` appended to `$C`; nominal rung
   run under both minimizer settings; rungs from the existing
   `results/comb_vbf_ggh/postfit_rungs.tsv`; then `postfit_breakdown.py`.
5. Outputs to `results/comb_vbf_ggh_recentred/`; HANDOFF Revision 6 appended.
   `/work/projects/.../comb_vbf_ggh/` is not updated until the numbers are accepted.

## Verification -- predictions fixed BEFORE running

| check | pass criterion |
|---|---|
| flag took effect | exactly 95 `Set Default Value of Parameter *_In` lines per job |
| the diagnosis itself | re-centred post-fit r_hat = 1.000 +- 0.001, vs 1.01727 |
| width direction | sigma WIDENS by about 1.5% to about 0.494, predicted from the measured significance drop 2.17412 -> 2.14081 |
| grid vs singles | sigma_sym(grid) vs total(singles) within 1% |
| stat rung | the all-frozen rung is bit-identical at 0.40498 |
| range guard | no rung reports an interval at the `r=0,3` edge |

Reference values being replaced: total 0.48742, stat 0.40498, syst 0.27124,
Experimental 0.24962, Size of simulated samples 0.07916, Theoretical 0.07066;
NLL sigma_sym 0.49124.
