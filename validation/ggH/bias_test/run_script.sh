#!/bin/bash
set -e
# we assume that the workflow stage1, stage2 have been already run


label="fullRun_Jun23_2025_1n2Revised"

stage2_load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/stage1_output"

category="ggh"

# model="V2_fullRun_Jun21_2025_1n2Revised"
# model="V2_Aug28_PosWgtRun0p7_MassResRun1"
model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_setRandomSeed_annhilateWeight_separateYears_Nov11repeat"

# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}/stage2_output" 
stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_recreate1_87SigOct31_2025_newEdgeTarget/stage2_output" 


# # ------------------
# # Begin stage3
# # ------------------

# stage3_label="${label}_X_${model}_w_allYearBDT_w_newBDTTargetYields_V2Jun28"
stage3_label="${label}_X_${model}_all7FitFuncs"

year="all"
python run_bias_test.py -load $stage2_save_path -cat $category --year $year --label $stage3_label



