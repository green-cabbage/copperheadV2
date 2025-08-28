#!/bin/bash
set -e
# run stage2 twice. First to generate BDT scores (we assume that an appropriate BDT is already trained, then generate score bin edges once more, then finally run stage2 again to save both bdt scores and ggH sub-category index


# label="V2_Jan17_JecDefault_valerieZpt"
# label="V2_Jan29_JecOn_TrigMatchFixed_2016UlJetIdFix"
# label="UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff"
# label="UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff_newZptWgt25Mar2025"
# label="DYMiNNLO_jetpuidOff_newZptWgt25Mar2025"
# label="DYMiNNLO_30Mar2025"
# label="DYamcNLO_11Apr2025"
# label="DYMiNNLO_11Apr2025"
# label="fullRun_May30_2025"
# label="fullRun_Jun21_2025"
# label="fullRun_Jun23_2025_1n2Revised"
# label="fullRun05Aug_2025"
label="fullRun_Jun23_2025_1n2Revised"
# label="BSC_off_Aug26_2025"
stage2_load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/stage1_output"

category="ggh"
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/$category/stage2_output" # I like to specify the category in the save path
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/20Mar2025_$category/stage2_output" # I like to specify the category in the save path

# model="V2_UL_Jun09_2025"
# model="V2_fullRun_Jun21_2025_allYear"
# model="V2_fullRun_Jun21_2025_1n2Revised"
# model="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes"
# model="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_NoJ1Eta_MinMmjdEta"
# model="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_DimuVarsOnly"
# model="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_NoSingleJet_Mu2_MinMmjdEta"
# model="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_NoSingleJet_MinMmjVars"
# model="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_NoJet_Mu2_MinMmjVars"
# model="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_Dimu_Mu1_MinMmjdEta_Vars"
# model="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_Dimu_Mu1_Mu2_Vars"
# model="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_Dimu_Mu1_jet1Pt_Vars"
# model="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_Dimu_Mu1_Mu2_jet1Pt_Vars"
# model="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_Dimu_Mu1_Mu2_jet1jet2Pt_Vars"
# model="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_Dimu_Mu1_Mu2_jet1Pt_jjdPhi_Vars"
model="V2_fullRun_Jun21_2025_1n2Revised"
# model="V2_fullRun_Jun21_2025_1n2Revised_BigEbeMassResContrib_Aug25_2025"
# model="V2_fullRun_Jun21_2025_1n2Revised_BigEbeMassResContrib_Aug25_2025_orig"
# model="V2_fullRun_Jun21_2025_1n2Revised_BigEbeMassResContrib_Aug25_2025_wgt_1e6"
# model="V2_fullRun_Jun21_2025_1n2Revised_BigEbeMassResContrib_Aug25_2025_noSigWgt"
# model="V2_fullRun_Jun21_2025_1n2Revised_BigEbeMassResContrib_Aug25_2025_DYSigOnly"
# model="V2_fullRun_Jun21_2025_1n2Revised_BigEbeMassResContrib_Aug25_2025_DYSigOnly_BSC_Off"
# model="V2_fullRun_Jun21_2025_1n2Revised_BigEbeMassResContrib_Aug25_2025_DYSigOnly_BSC_Off_oldBDT"
# model="V2_fullRun_Jun21_2025_1n2Revised_BigEbeMassResContrib_Aug25_2025_DYSigOnly_BSC_Off_oldBDT_amcDY"
# model="V2_UL_Jan18_2025"

# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_Individual_yearBDT/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_AN_BDTTargetYields/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_NoJ1Eta_MinMmjdEta/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_DimuVarsOnly/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_NoSingleJet_Mu2_MinMmjdEta/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_NoSingleJet_MinMmjVars/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_NoJet_Mu2_MinMmjVars/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_Dimu_Mu1_MinMmjdEta_Vars/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_Dimu_Mu1_Mu2_Vars/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_Dimu_Mu1_jet1Pt_Vars/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_Dimu_Mu1_Mu2_jet1Pt_Vars/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_Dimu_Mu1_Mu2_jet1jet2Pt_Vars/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_Dimu_Mu1_Mu2_jet1Pt_jjdPhi_Vars/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_w_allYearBDT_w_newBDTTargetYields/stage2_output" 
stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}/stage2_output" 
# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}/stage2_outputForFig6_7" 



bdt_edge_config_path="/work/users/yun79/Run3/copperheadV2/configs/MVA/ggH/BDT_edges.yaml"

# year="2018"
# sample_l="ggh vbf" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
# # python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year --edge_cfg_path ${bdt_edge_config_path}
# # sample_l="data ggh vbf dy ewk tt st ww wz zz other" 
# sample_l="ggh vbf" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

# year="2017"
# sample_l="ggh vbf" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
# # python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year --edge_cfg_path ${bdt_edge_config_path}
# # sample_l="data ggh vbf dy ewk tt st ww wz zz other" 
# sample_l="ggh vbf" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model


# year="2016postVFP"
# sample_l="ggh vbf" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
# python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year --edge_cfg_path ${bdt_edge_config_path}
# # sample_l="data ggh vbf dy ewk tt st ww wz zz other" 
# sample_l="ggh vbf" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

# year="2016preVFP"
# sample_l="ggh vbf" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model
# python stage2/ggH/calculate_score_edges.py -load $stage2_save_path --year $year --edge_cfg_path ${bdt_edge_config_path}
# # sample_l="data ggh vbf dy ewk tt st ww wz zz other" 
# sample_l="ggh vbf" 
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model

# ------------------
# stage2 specifically for fig 6.7
# ------------------
year="2018"
sample_l="dy" # fig 6.7 only requires signal samples
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model --do_6p7 # --do_jecUnc
# year="2017"
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model --do_6p7 # --do_jecUnc
# year="2016postVFP"
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model --do_6p7 # --do_jecUnc
# year="2016preVFP"
# python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model --do_6p7 # --do_jecUnc


# # ------------------
# # Begin stage3
# # ------------------

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
# stage3_label="${label}_X_${model}"
stage3_label="${label}_X_${model}_w_allYearBDT_w_newBDTTargetYields_Aug12_2025"

# # year="all"
# # python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label

# year="2018"
# python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label
# year="2017"
# python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label
# year="2016postVFP"
# python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label
# year="2016preVFP"
# python run_stage3.py -load $stage2_save_path -cat $category --year $year --label $stage3_label



