# Introduction
This has intructions to generate: Fig 6.8, 6.10, 6.13, 6.14, 6.19 and Table 6.2 and 6.12.

# Technical Details


## Generate Figure 6.14 and 6.11
```bash
cd validation/ggH/categorization/
python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}
```
The example of this is saved in `validation/ggH/categorization/run_script.sh`
within `validation_plot.py`, there's an option to unblind. 


## Generate Figure 6.8m 6.13 and 6.19
```bash
cd validation/ggH/categorization/
python plot_6_8.py -label $label -cat $category -y ${year} --region ${region}
python plot_6_13.py -label $label -cat $category -y ${year} --region ${region}
python plot_6_19.py -label $label -cat $category -y ${year} --region ${region}
```
The example of this is saved in `validation/ggH/categorization/run_script.sh`
within the python scipts where relevant, there should be an option to unblind. 


## Generate Table 6.2 and 6.12
```bash
cd validation/ggH/categorization/
python getTable_6_2And6_12.py -label $label -cat $category -y ${year} --region ${region}
```

The example of this is saved in `validation/ggH/categorization/run_script.sh`


