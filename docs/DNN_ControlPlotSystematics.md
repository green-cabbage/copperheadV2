---
title: DNN Control Plot Systematics
---

# Stage-2 DNN control plots with uncertainties

- Available code: [plot_DNN_score.py](../plotter/plot_DNN_score.py).
- Workflow: `run_analysis_pipeline.sh -m 2p` makes sideband and blinded-peak plots.
- Inputs: existing stage-2 histograms containing the selected variations.
- Outputs: linear/log PDFs, yield summaries, and copies of the selected YAML.

## Run the plotter

Run from the repository in its project environment; replace the input label and postfix:

```bash
WITH_VARIATIONS=1 \
STAGE2_SYSTEMATICS_CONFIG=configs/plotting/stage2_systematics.yaml \
STAGE2_UNCERTAINTY_SCOPE=background+signal \
STAGE2_UNCERTAINTY_BANDS=both \
STAGE2_PLOT_SAVE_PATH=validation/from_stage2_uncBand \
bash run_analysis_pipeline.sh -m 2p -y 2018 -l YOUR_LABEL -o YOUR_POSTFIX
```

- `WITH_VARIATIONS=1` selects variation-bearing inputs; it does not generate them.
- `-o` selects the **input postfix**. Change `STAGE2_PLOT_SAVE_PATH` for a new output directory.
- Choose a fresh output directory to retain earlier plots.
- Without a configuration, the plotter uses its original filenames and statistical bands.

## Choose the observable

The bands are not specific to the DNN score. One reserved key picks the observable:

```yaml
variable: dimuon_mass   # omit this key to plot the DNN score
pdf:
  - name: pdf_unc
    type: pdf_hessian
```

| Step | DNN score (default) | Kinematic variable |
|---|---|---|
| Fill | `run_analysis_pipeline.sh -m 2` | same, with `STAGE2_HIST_VARIABLE=dimuon_mass` |
| Histogram dir | `stage2_histograms/score_<label>_<postfix>/` | `stage2_histograms/dimuon_mass_<label>_<postfix>/` |
| Binning | `configs/MVA/VBF/dnn_binning.yaml` | `binning_linspace` in [plot_settings_vbfCat_MVA_input.json](../src/lib/histogram/plot_settings_vbfCat_MVA_input.json) |

- Stage2 and the plotter read the edges from that one json entry, so the plotter's binning check cannot disagree.
- The variable must be a column in the compacted stage-1 parquets and have a `binning_linspace` entry.
- `-m 2p` needs no new option: the plotter swaps the `score_` directory prefix for the variable itself.
- Kinematic mode evaluates no DNN, so no trained model, scaler or feature list has to be present.
- It also **skips the h-sidebands 125 GeV mass pin**, which exists only so sideband events are
  scored at the signal mass hypothesis and would otherwise pile every sideband event into one bin.
- Blinding is unchanged: `Reg_h-peak` (115-135 GeV) has its data zeroed, so a 110-150 GeV mass plot
  shows data only in the 110-115 and 135-150 GeV sidebands.

## Select systematic groups

Use named groups to draw separate colored dashed boundaries:

```yaml
pileup:
  - name: pileup
    up: wgt_pu_up
    down: wgt_pu_down
muon_roch:
  - name: muon_momentum
    up: mu_roccor_up
    down: mu_roccor_down
```

- Each group can contain multiple nuisances.
- `up` and `down` must match exact histogram variation labels.
- Optional per-nuisance `groups: [DY, DYVBF]` restricts the affected **MC processes**.
- Omit `groups` to apply the nuisance to all MC in the selected prediction.
- Duplicate nuisance names, variation labels, or YAML keys are rejected.
- Missing variations, unknown MC groups, and incompatible binning are errors.
- The legacy `systematics: [...]` format remains supported and draws one black systematic boundary pair.
- Do not mix `systematics:` with named groups. `systematics: []` selects zero systematic error.
- The supplied [example YAML](../configs/plotting/stage2_systematics.yaml) contains pileup only.

## PDF and alpha_s

These three estimators read the per-member weight columns stage1 writes
(`wgt_pdfMemberHessEig0NN_up`, `wgt_pdfAlphaS101_up`, `wgt_pdfAlphaS102_up`)
instead of an explicit `up`/`down` pair, so the entry carries `type` and no labels:

```yaml
pdf:
  - name: pdf_unc
    type: pdf_hessian
alpha_s:
  - name: alpha_s_unc
    type: alpha_s
```

| `type` | Estimator per bin | Reference |
|---|---|---|
| `pdf_hessian` | `sqrt( sum_k (F_k - F_0)^2 )` over the 100 eigenvector members | arXiv:2203.05506 Eq. (6.5) |
| `alpha_s` | `(F(0.120) - F(0.116)) / 2` | PDF4LHC15 Eqs. (27)-(28) |
| `pdf_alpha_s` | `sqrt(pdf^2 + alpha_s^2)` | PDF4LHC15 Eq. (28) |

- Ready-made files: [split](../configs/plotting/stage2_systematics_pdf_alphas.yaml),
  [comparison](../configs/plotting/stage2_systematics_pdf_alphas_compare.yaml).
- `pdf_hessian` plus `alpha_s` equals one `pdf_alpha_s`, exactly as `split_pdf_alpha_s` does in stage3.
- All three are symmetric: the up and down errors are equal by construction.
- They are **not** normalized to the nominal yield, because `wgt_pdf_unc_*` is not in the stage3 `shape_only` list.
- Unlike stage3 the downward band is not floored at zero; the log panel clips at the axis floor instead.
- `groups:` works as for any nuisance, so per-process decorrelation (DY, ggH, VBF) is one entry each.
- Repeating one estimator over the same MC groups is rejected; a combined entry beside its own components only warns, since that is the comparison plot.
- Every selected sample must carry all 100 members and both alpha_s members, or the plot fails rather than quietly shrinking the band.
- The grey total double counts when `pdf_alpha_s` sits beside `pdf_hessian`/`alpha_s`; read the dashed boundaries there.
- Constants live beside the code in `plotter/plot_DNN_score.py` and mirror `stage3/make_templates.py`.

## Plot options

| Setting | Choices and behavior |
|---|---|
| `STAGE2_UNCERTAINTY_SCOPE` | `background` (default): Data/B; `background+signal`: Data/(B+S) |
| `STAGE2_UNCERTAINTY_BANDS` | `stat+syst` (default): grey total; `syst-only`: group boundaries; `both`: both |
| `STAGE2_PLOT_SAVE_PATH` | Output root; default `validation/from_stage2/` |

- Signal enters B+S at physical normalization, independently of overlay scaling.
- No standalone B+S line is drawn; peak data remain blinded.
- Main legend: 18 pt. Separation d-score text: 15 pt.
- Direct CLI options: `--systematics-config`, `--uncertainty-scope`, `--uncertainty-bands`, `--save_path`.
- Nondefault uncertainty scope/band options require a configuration.

## How uncertainties are combined

- Sum each nuisance's shifts coherently across its affected samples before taking an up/down envelope.
- Combine independent nuisances within each group in quadrature, separately for up/down errors.
- Grey total: `sqrt(MC sumw2 + sum of group errors squared)`.
- Grey-band label: `total MC unc (stat+syst)`.
- Ratio-point errors: data statistics + MC statistics + selected systematics, propagated in quadrature through Data/prediction.
- Prediction-up contributes ratio-down error, and vice versa (first-order propagation).
- Without selected systematics, ratio-point errors contain data + MC statistics only.
- Nonpositive prediction denominators are masked.
- A nuisance is correlated across its applicable loaded samples/years; separate nuisances/groups are assumed independent.
- Avoid double counting: for example, choose individual JES components or JES Total, not both.

## Saved configuration and filenames

- Band-enabled filenames contain `_unc_<scope>_<bands>`; log plots also contain `_log`.
- Each configured PDF has a same-stem `.yaml` copy, preserving source comments.
- Matching `.txt` files record yields, binning, scope, and selected systematics.
- No configuration selected: **no YAML copy**, even if the default YAML is nonempty.
- An explicitly selected empty configuration is still copied; its systematic contribution is zero.
