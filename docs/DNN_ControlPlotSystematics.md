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
