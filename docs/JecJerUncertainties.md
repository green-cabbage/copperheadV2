# Introduction
This has intructions to generate: JEC/JER up/down uncertainties. We assume a stage1 is already done with the necessary JEC and JER uncertainties saved.

# Technical Details
## Symbolic links

If a fresh git clone, first apply symbolic link to modules and src by:
```bash
cd validation/ggH/jec/
ln -s {full_path_to_repo}/src  .
ln -s {full_path_to_repo}/modules  .
```


## Generate JEC/JER up/down uncertainty plots
```bash
sh run_script.sh
```
The plots will be saved in `./plots` directory