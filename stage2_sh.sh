#!/bin/bash
set -e
# run stage2 twice. First to generate BDT scores (we assume that an appropriate BDT is already trained, then generate score bin edges once more, then finally run stage2 again to save both bdt scores and ggH sub-category index



label="fullRun_Jun23_2025_1n2Revised"
# label="fullRun05Aug_2025" # for datacard
# label="BSC_off_Aug26_2025"
# label="jetHornStudy_29Apr2025_JecDefaultJerOff"
# label="DYMiNNLO_11Apr2025"
stage2_load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/stage1_output"

category="ggh"
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/$category/stage2_output" # I like to specify the category in the save path
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/20Mar2025_$category/stage2_output" # I like to specify the category in the save path


# model="V2_Aug28_PosWgtRun0p7_MassResRun1"
# model="V2_Aug28_PosWgtRun0p7_MassResRun2"
# model="V2_Aug28_PosWgtRun0p7_MassResRun3"
# model="V2_Aug28_PosWgtRun0p7_MassResRun4"
# model="V2_Aug28_PosWgtRun0p7_MassResRun5"
# model="V2_Aug28_PosWgtRun0p7_MassResRun1_DyOnlyBkg"
# model="V2_Aug28_PosWgtRun0p7_MassResRun1_removeForwardJets"
# model="V2_Aug16_2025Reprod_IssueNum2"
# model="V2_Aug16_2025DefaultFillNoneBack_IssueNum2"
# model="V2_Aug16_2025DefaultFillNoneBack_AnnhilateNegWgts_IssueNum2"
# model="V2_Aug16_2025DefaultFillNoneBackFix_AnnhilateNegWgts_IssueNum2"
# model="V2_Aug16_2025DefaultFillNoneBackFix_IssueNum2"
# model="V2_Aug16_2025DefaultFillNoneBackFix_AnnhilateNegWgts_AddRpt_IssueNum2"
# model="V2_Aug16_2025DefaultFillNoneBackFix_AnnhilateNegWgts_AddRptNoEarlyStop_IssueNum2"
# model="V2_Aug16_2025AddIssue1To3_IssueNum2"
# model="V2_Aug16_2025Reprod_IssueNum2"
# model="V2_Aug16_2025DefaultFillNoneBackFix_IssueNum2"
# model="V2_Aug16_2025DefaultFillNoneBackFix_AnnhilateNegWgts_IssueNum2"
# model="V2_Aug16_2025DefaultFillNoneBackFix_AnnhilateNegWgts_AddRpt_IssueNum2"
# model="V2_fullRun_Jun21_2025_1n2Revised"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run2"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_setRandomSeed"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_setRandomSeed_repeat"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProd_AddRpt"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProd_RemoveRptKeepYr"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_setRandomSeed_FillNoneCorrected"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_setRandomSeed_annhilateWeight"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_setRandomSeed_applyClassWgtBalance"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_setRandomSeed_addOnlyRpt"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_setRandomSeed_addRptNYear"
# model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_setRandomSeed_addOnlyYear"
model="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_applyClassWgtBalance_separateYears"


# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_recreateOct24_2025/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}/stage2_output4DataCard" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_recreate1_87SigOct28_2025/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_recreate1_87SigOct28_2025_BDTJune23Recreated/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_recreate1_87SigOct28_2025_BDTJune23Recreated_newEdgeTarget/stage2_output" 
stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_recreate1_87SigOct31_2025_newEdgeTarget/stage2_output" 


bdt_edge_config_path="/work/users/yun79/Run3/copperheadV2/configs/MVA/ggH/BDT_edges.yaml"


# -----------------------------------------------------
# # for BDT edge calculation
# year="2018"
# # sample_l="ggh vbf dy ewk tt st ww wz zz other" 
# sample_l="ggh vbf data" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

# year="2017"
# # sample_l="ggh vbf dy ewk tt st ww wz zz other" 
# sample_l="ggh vbf data" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model


# year="2016postVFP"
# # sample_l="ggh vbf dy ewk tt st ww wz zz other" 
# sample_l="ggh vbf data" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

# year="2016preVFP"
# # sample_l="ggh vbf dy ewk tt st ww wz zz other" 
# sample_l="ggh vbf data" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
#------------------------------------------------------

year="2018"
sample_l="ggh vbf" 
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year --edge_cfg_path ${bdt_edge_config_path}
# sample_l="ggh vbf dy ewk tt st ww wz zz other" 
sample_l="data ggh vbf" 
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

year="2017"
sample_l="ggh vbf" 
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year --edge_cfg_path ${bdt_edge_config_path}
# sample_l="ggh vbf dy ewk tt st ww wz zz other" 
sample_l="data ggh vbf" 
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model


year="2016postVFP"
sample_l="ggh vbf" 
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year --edge_cfg_path ${bdt_edge_config_path}
# sample_l="ggh vbf dy ewk tt st ww wz zz other" 
sample_l="data ggh vbf" 
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

year="2016preVFP"
sample_l="ggh vbf" 
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year --edge_cfg_path ${bdt_edge_config_path}
# sample_l="ggh vbf dy ewk tt st ww wz zz other" 
sample_l="data ggh vbf" 
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

# # ------------------
# # stage2 specifically for fig 6.7
# # ------------------
# year="2018"
# # sample_l="ggh vbf dy ewk tt st"  # fig 6.7 only requires signal samples
# # sample_l="ewk tt st" # for comparison with dy
# sample_l="ggh vbf" # for comparison with dy
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model --do_6p7 # --do_jecUnc
# year="2017"
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model --do_6p7 # --do_jecUnc
# year="2016postVFP"
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model --do_6p7 # --do_jecUnc
# year="2016preVFP"
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model --do_6p7 # --do_jecUnc


# ------------------
# Begin stage3
# ------------------

# # # stage3_label="${label}_X_${model}"
# # # stage3_label="${label}_X_${model}_matchUCSD_values"
# # # stage3_label="test"
# # stage3_label="simple_fit_test_16Mar2025"
# # # stage3_label="${label}_X_${model}_ucsdFitFuncs_newBinEdges"
# # # stage3_label="${label}_X_${model}_Feb05_coreFuncFixed_newBinEdges"
# # # stage3_label="${label}_X_${model}_Feb09_newBinEdges_bySig"
# # # stage3_label="${label}_X_${model}_Feb11_newBinEdges_bySig_useRooDoubleCBFast_hPeakSigFit"
# # # stage3_label="${label}_X_${model}_Feb15_newBinEdges"
# # # stage3_label="${label}_X_${model}_Feb16_testBinEdges"
# # # stage3_label="${label}_X_${model}_Feb16_testBinEdges2"
# # # stage3_label="${label}_X_${model}_Feb19_fullSigFitRange"
# stage3_label="${label}_X_${model}"
# stage3_label="${label}_X_${model}_w_allYearBDT"
# stage3_label="${label}_X_${model}_w_Individual_yearBDT"
# stage3_label="${label}_X_${model}_w_allYearBDT_w_newBDTTargetYields"
# stage3_label="${label}_X_${model}_w_allYearBDT_w_AN_BDTTargetYields"
# stage3_label="${label}_X_${model}_w_allYearBDT_w_newBDTTargetYields_V2Jun28"
# stage3_label="${label}_X_${model}_w_allYearBDT_w_newBDTTargetYields_Aug12_2025"
# stage3_label="${label}_X_${model}"
# stage3_label="${label}_X_${model}_redoSep21_2025"
# stage3_label="${label}_X_${model}_redoSep23_2025"
# stage3_label="${label}_X_${model}_recreateOct24_2025"
# stage3_label="${label}_X_${model}_w_allYearBDT_w_newBDTTargetYields_recreateOct24Run2_2025"
# stage3_label="${label}_X_${model}_w_allYearBDT_w_newBDTTargetYields_recreate1_87SigOct28_2025"
# stage3_label="${label}_X_${model}_w_allYearBDT_w_newBDTTargetYields_recreate1_87SigOct28_2025_BDTJune23Recreated_Run2"
# stage3_label="${label}_X_${model}_recreate1_87SigOct29_2025_BDT_Run3"
# stage3_label="${label}_X_${model}_recreate1_87SigOct29_2025_BDT_Run3_repeat"
# stage3_label="${label}_X_${model}_recreate1_87SigOct29_2025_BDT_Run3_recalculateBDTEdges"
# stage3_label="${label}_X_${model}_recreate1_87SigOct29_2025_BDT_Run3_recalculateTargYieldNBDTEdges"
stage3_label="${label}_X_${model}_recreate1_87SigOct31_2025_BDT_AddRpt_recalculateTargYieldNBDTEdges"

echo "stage2 path: ${stage2_save_path}"
year="all"
python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label

# year="2018"
# python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label
# year="2017"
# python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label
# year="2016postVFP"
# python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label
# year="2016preVFP"
# python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label



