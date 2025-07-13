# Introduction

# Technical Details

## Obtain DNN data/MC plots
Before running the script `run_script.sh`, please update `label`, `year` and `mva_name`. Furthermore, we assume that stage2 is run for VBF category, which saves the stage2 histograms in: `/depot/cms/users/yun79/hmm/copperheadV1clean/{label}/stage2_histograms/score_{mva_name}/{year_param}/`
if region is specified as `h-peak`, the data histogram is automatically blinded.

```bash
cd ./validation/VBF/
bash run_script.sh
```


# References/Important links
