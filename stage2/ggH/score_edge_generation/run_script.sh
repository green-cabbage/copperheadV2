#!/bin/bash
set -e

# label="V2_Jan29_JecOn_TrigMatchFixed_2016UlJetIdFix"
# label="DYMiNNLO_30Mar2025"
# label="DYamcNLO_11Apr2025"
# label="DYMiNNLO_11Apr2025"
# label="fullRun_May30_2025"
# label="fullRun_Jun21_2025"
label="fullRun_Jun23_2025_1n2Revised/"

category="ggh"
# model="V2_UL_Apr09_2025_DyMinnloTtStVvEwkGghVbf_hyperParamOnScaleWgt0_75"
# model="V2_UL_Apr11_2025_DyTtStVvEwkGghVbf"
# model="V2_UL_Apr11_2025_DyMinnloTtStVvEwkGghVbf"
# model="V2_UL_Jun09_2025"
# model="V2_fullRun_Jun21_2025_allYear"
# model="V2_fullRun_Jun21_2025_1n2Revised"
# model="V2_Aug28_PosWgtRun0p7_MassResRun1"
# model="V2_Aug16_2025DefaultFillNoneBackFix_AnnhilateNegWgts_AddRptNoEarlyStop_IssueNum2"
# model="V2_Aug16_2025Reprod_IssueNum2"
# model="V2_Aug16_2025DefaultFillNoneBackFix_IssueNum2"
# model="V2_Aug16_2025DefaultFillNoneBackFix_AnnhilateNegWgts_IssueNum2"
# model="V2_Aug16_2025DefaultFillNoneBackFix_AnnhilateNegWgts_AddRpt_IssueNum2"
# model="V2_fullRun_Jun21_2025_1n2Revised"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_setRandomSeed"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProd_AddRpt"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProd_RemoveRptKeepYr"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_setRandomSeed_FillNoneCorrected"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_setRandomSeed_annhilateWeight"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_setRandomSeed_applyClassWgtBalance"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_setRandomSeed_addOnlyRpt"
model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_setRandomSeed_addRptNYear"


# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/$category/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}/stage2_output"  # I like to specify the category in the save path
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_recreateOct24_2025/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields/stage2_output"  # I like to specify the category in the save path
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_recreate1_87SigOct28_2025_BDTJune23Recreated/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_recreate1_87SigOct28_2025_BDTJune23Recreated_newEdgeTarget/stage2_output" 
stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_recreate1_87SigOct31_2025_newEdgeTarget/stage2_output" 


years="2016preVFP 2016postVFP 2017 2018"
# years="2018"
# years="2016preVFP 2016postVFP 2018"
python determine_score_edge.py -load $stage2_save_path --years ${years}

# despite its name, this plots the AMS values from saved .csv output to pngs
python validation.py