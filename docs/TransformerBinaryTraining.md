---
title: Transformer training
---

# Training the binary ggH-vs-VBF transformer

`MVA_training/transformer_binary` trains the binary transformer that defines the transformer VBF channel
([Transformer VBF channel](TransformerVBF.md)). It reimplements the Sep08 recipe of
[b-hive-h-mu-mu](https://gitlab.cern.ch/xudongh/b-hive-h-mu-mu): same inputs, architecture, fold hash and training
defaults.

The model separates ggH (label 1) from VBF (label 0) signal MC:

| | |
| --- | --- |
| events | h-peak (`115 <= dimuon_mass < 135`), after the b-tag veto; no njets requirement |
| inputs | five object tokens (mu1, mu2, dimuon, jet1, jet2), global features (htsoft2, htsoft5, MET, njets), year embedding |
| kinematics | nominal only, so the score does not move with JES/JER |
| folds | 4, from a hash of (year, sample, run, lumi, event); model `k` never trains on fold `k` |
| loss | BCE weighted by \|w\|, each class normalised to 1 with the same year mixture |
| output | the ggH probability; VBF channel = `score < cutoff` |

A training covers one run period: `--years` must be all Run 2 or all Run 3. The Run-3 year vocabulary is
`2022preEE, 2022postEE, 2023, 2023BPix, 2024, 2025, 2026`; years missing from the inputs are unused rows of the year
embedding.

The training defaults (`training`), the signal sample groups (`samples`) and the HPO search space (`hpo`) are in
`configs/MVA/transformer_categorizer.yaml`.

| file | role |
| --- | --- |
| `ray_train.py` | fold/seed trainings on the AF Ray cluster, one GPU each (the usual entry point) |
| `train.py` | one fold, without Ray |
| `features.py` | stage-1 parquet → inputs, and the training selection |
| `folds.py` | fold hash and year vocabularies (bit-for-bit as b-hive) |
| `model.py` | the network |
| `hpo.py` | optional HPO (Ray Tune + Optuna) |
| `score.py`, `ray_score.py` | out-of-fold score into a copy of stage 1 |

All commands run from the repo root of a checkout on `/work` (the Ray cluster runs the code from that checkout), with
`PYTHONPATH=$PWD`. Python steps use the AF global environment (`/work/pixi/global/.pixi/envs/default/bin/python`,
written `python` below). Submit Ray jobs from a session without a GPU; `RAY_AUTH_MODE` and `RAY_AUTH_TOKEN` are preset
in AF Jupyter terminals (`source ~/.ray_auth.env` first if your session needs it).

!!! warning "Live checkout"
    A Ray job runs this checkout as it is when the job starts. Do not edit it or switch branches while a job is
    pending or running.

## 1. Stage-1 inputs

The trainer reads `<stage1_output>/<year>/compacted/<sample>/**/*.parquet`.

* **Samples:** the `GGH` and `VBF` groups of `configs/samples/samples.yaml`, resolved per year (now `ggh_powhegPS`;
  `vbf_powheg_dipole`, `vbf_powheg` for 2023). Add new sample names to those groups, or point
  `samples.signal_groups` in `transformer_categorizer.yaml` at other groups.
* **Columns**, each read as `<name>_nominal` if present, else `<name>`:
    * `mu1_{pt,eta,phi}`, `mu2_{pt,eta,phi}`, `dimuon_{pt,eta,phi}`;
    * `jet1_{pt,eta,phi,mass}`, `jet2_{pt,eta,phi,mass}`;
    * `htsoft2`, `htsoft5`, `MET_pt`, `MET_phi` (else `PuppiMET_pt`, `PuppiMET_phi`), `njets`;
    * `run`, `luminosityBlock`, `event`, `dimuon_mass`, `nBtagLoose`, `nBtagMedium`, `wgt_nominal`.

Nulls (missing jets) become 0; a missing column stops the job with `stage1 has none of [...]`. Check the inputs on the
CPU before using GPUs:

```bash
python -c "
from MVA_training.transformer_binary.features import load_signal
d = load_signal('<stage1_output>', ['2023'])
print(len(d['is_ggH']), 'events; ggH', int(d['is_ggH'].sum()))
"
```

## 2. Plan

One Ray task, and one GPU, per job:

```json
{
  "stage1": "<stage1_output>",
  "years": ["2022preEE", "2022postEE", "2023", "2023BPix"],
  "out": "<out>",
  "hparams": {},
  "jobs": [
    {"fold": 0, "stage": "selection+refit", "seed": 123456},
    {"fold": 1, "stage": "selection+refit", "seed": 123456},
    {"fold": 2, "stage": "selection+refit", "seed": 123456},
    {"fold": 3, "stage": "selection+refit", "seed": 123456}
  ]
}
```

| stage | what it does |
| --- | --- |
| `selection` | trains on the training folds minus a ~10% inner-validation split, early-stops on it (`patience` 10, `max_epochs` 1000): picks the epoch count |
| `refit` | retrains on all training folds for `"epochs"` epochs (needed in the job); the last epoch is the model |
| `selection+refit` | both in one task, refit with best epoch + 1 (use this) |

* `out` must be on `/work` or `/depot`; the Ray cluster has no `/home`.
* `hparams` overrides the `training` defaults. Leave it empty for the Sep08 recipe: batch 4096, AdamW (lr 1e-3, betas
  0.95/0.999, wd 0.01), 3 encoder layers, 8 heads, embed dim 128, dropout 0.1, mixed precision, gradient clip 1.0. The
  learning rate is constant, then decays geometrically to 1% over the last 30% of the epochs, so in practice only in
  the refit. For a plumbing test: `{"max_epochs": 3, "patience": 1}`.
* Extra entries of the same fold with other seeds measure the seed spread.

## 3. Train

```bash
python -m MVA_training.transformer_binary.ray_train submit --plan plan.json --max-workers 4   # prints the job id
python -m MVA_training.transformer_binary.ray_train status --job <job id>                      # status + log tail
```

`--max-workers` is the number of GPU workers (one per job; each `--cpus 8 --ram-gb 32` by default). The cluster starts
with the GPUs free at the time; the other tasks wait for one.

The job first caches the selected signal events in `<out>/data_cache/signal_<hash>.npz`, once per (stage1, years);
delete it if the inputs change under the same path. Every epoch is checkpointed (`last_epoch.pt`), and a pre-empted
task resumes from it (up to 5 retries). Resuming into an `out` written with other settings is an error: use a new
`out` per configuration. On the Run-2 signal sample (about 5M events) an epoch took about 2 minutes and a fold 20–30
epochs.

One fold without Ray (e.g. on a GPU session):

```bash
python -m MVA_training.transformer_binary.train --stage1 <stage1_output> --years 2022preEE,2022postEE,2023,2023BPix \
    --fold 0 --stage selection --out <out>/selection/fold0_seed123456
python -m MVA_training.transformer_binary.train ... --fold 0 --stage refit --epochs <best_epoch+1> \
    --out <out>/refit/fold0_seed123456
```

## 4. Outputs

```
<out>/
  data_cache/signal_<hash>.npz
  selection/fold<k>_seed<s>/  best_model.pt  last_epoch.pt  _SUCCESS.json   # best_epoch, best_val_loss, refit_epochs, history
  refit/fold<k>_seed<s>/      model.pt       last_epoch.pt  _SUCCESS.json   # model.pt: the model of fold k
  ray_results_<timestamp>.json                                              # one entry per task, errors included
```

The model is the four `refit/fold{0..3}_seed123456/model.pt`. Each holds the weights, the hyperparameters and the
training config (stage-1 path and years included). The selection's `_SUCCESS.json` has the per-epoch train and
validation loss.

## 5. Next steps

Score stage 1 into a copy, derive the cutoff and run the channel as in [Transformer VBF channel](TransformerVBF.md)
(steps 2–5), with the run period's years:

```bash
python -m MVA_training.transformer_binary.ray_score submit --stage1 <stage1_output> \
    --years 2022preEE,2022postEE,2023,2023BPix \
    --models '<out>/refit/fold{fold}_seed123456/model.pt' \
    --out-root <scored root>/stage1_output --out-subdir compacted --mass-window 110,150
```

The Run-3 cutoff (`cutoff.run3` in `transformer_categorizer.yaml`) is still `null`.

!!! note "Run-to-run noise"
    Two trainings with the same settings and seed are not bit-identical on GPU (cuDNN, TF32, mixed precision).
    Compare models by AUC or score correlation, not by exact scores.
