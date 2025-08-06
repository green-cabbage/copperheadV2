#!/bin/bash
set -e
# we assume that the workflow stage1, stage2 have been already run


label="fullRun05Aug_2025"

stage2_load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/stage1_output"

category="ggh"

model="V2_fullRun_Jun21_2025_1n2Revised"

# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields/stage2_output" 
stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}/stage2_output" 


# # ------------------
# # Begin stage3
# # ------------------

stage3_label="${label}_X_${model}"
year="all"
python generate_datacard.py -load $stage2_save_path -cat $category --year $year --label $stage3_label



