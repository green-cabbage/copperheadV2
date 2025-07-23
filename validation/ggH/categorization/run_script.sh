#!/bin/bash
set -e
# run stage2 twice. First to generate BDT scores (we assume that an appropriate BDT is already trained, then generate score bin edges once more, then finally run stage2 again to save both bdt scores and ggH sub-category index

# sample_l="data dy ewk tt st ww wz zz" 
sample_l="data ggh vbf dy ewk tt st ww wz zz other" 

# label="V2_Jan29_JecOn_TrigMatchFixed_2016UlJetIdFix"
# label="UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff"
# label="UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff_newZptWgt25Mar2025"
# label="DYMiNNLO_jetpuidOff_newZptWgt25Mar2025"
# label="DYMiNNLO_30Mar2025"
# label="fullRun_May30_2025"
# label="fullRun_Jun21_2025"
label="fullRun_Jun23_2025_1n2Revised"

stage2_load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/stage1_output"

# category="20Mar2025_ggh"

# model_name="V2_UL_Mar24_2025_DyTtStVvEwkGghVbf_scale_pos_weight"
# model_name="V2_UL_Mar25_2025_DyGghVbf_scale_pos_weight_newZpt"
# model_name="V2_UL_Mar26_2025_DyGghVbf_scale_pos_weight_dyMiNNLO"
# model_name="V2_UL_Mar26_2025_DyTtStVvEwkGghVbf_scale_pos_weight_dyMiNNLO"
# model_name="V2_UL_Mar30_2025_DyMiNNLOGghVbf"
# model_name="V2_UL_Mar30_2025_DyMiNNLOGghVbf_removeJetVar"
# model_name="V2_UL_Mar30_2025_DyMiNNLOGghVbf_removeAllJetVar"
# model_name="V2_UL_Mar30_2025_DyMiNNLOGghVbf_onlyMuVar"
# model_name="V2_UL_Mar30_2025_DyMiNNLOGghVbf_onlyMuVar_ZeppenJjMass_DeltaVars"
# model_name="V2_UL_Apr09_2025_DyMinnloTtStVvEwkGghVbf_hyperParamOnScaleWgt0_75"
# model_name="V2_UL_Jun09_2025"
# model_name="V2_fullRun_Jun21_2025_allYear"
model_name="V2_fullRun_Jun21_2025_1n2Revised"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_NoJ1Eta_MinMmjdEta"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_DimuVarsOnly"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_NoSingleJet_Mu2_MinMmjdEta"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_NoSingleJet_MinMmjVars"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_NoJet_Mu2_MinMmjVars"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_Dimu_Mu1_MinMmjdEta_Vars"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_Dimu_Mu1_Mu2_Vars"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_Dimu_Mu1_jet1Pt_Vars"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_Dimu_Mu1_Mu2_jet1Pt_Vars"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_noEbeMassRes_Dimu_Mu1_Mu2_jet1jet2Pt_Vars"

# category="${model_name}_ggh"
# category="${model_name}_ggh_w_allYearBDT"
category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields"
# category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes"
# category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_NoJ1Eta_MinMmjdEta"
# category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_DimuVarsOnly"
# category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_NoSingleJet_Mu2_MinMmjdEta"
# category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_NoSingleJet_MinMmjVars"
# category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_NoJet_Mu2_MinMmjVars"
# category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_Dimu_Mu1_MinMmjdEta_Vars"
# category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_Dimu_Mu1_Mu2_Vars"
# category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_Dimu_Mu1_jet1Pt_Vars"
# category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_Dimu_Mu1_Mu2_jet1Pt_Vars"
# category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_Dimu_Mu1_Mu2_jet1jet2Pt_Vars"

# year="2018"
# year="2017"
# year="2016"
# year="2016postVFP"
# year="2016preVFP"
year="all"

# region="z-peak"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# region="signal"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# region="h-sidebands"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# # plot Fig 6.13 from AN-19-124
region="signal"
# python plot_6_8.py -label $label -cat $category -y ${year} --region ${region}
# python plot_6_13.py -label $label -cat $category -y ${year} --region ${region}
# python plot_6_19.py -label $label -cat $category -y ${year} --region ${region}
python getTable_6_2And6_12.py -label $label -cat $category -y ${year} --region ${region}

# # -----------------------------------------------------
# year="2016postVFP"

# region="z-peak"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# region="signal"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# region="h-sidebands"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# # plot Fig 6.13 from AN-19-124
# region="signal"
# python plot_6_13.py -label $label -cat $category -y ${year} --region ${region}
# python plot_6_19.py -label $label -cat $category -y ${year} --region ${region}

# # -----------------------------------------------------
# year="2016preVFP"

# region="z-peak"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# region="signal"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# region="h-sidebands"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# # plot Fig 6.13 from AN-19-124
# region="signal"
# python plot_6_13.py -label $label -cat $category -y ${year} --region ${region}
# python plot_6_19.py -label $label -cat $category -y ${year} --region ${region}


# # -----------------------------------------------------
# year="2016"

# region="z-peak"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# region="signal"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# region="h-sidebands"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# # plot Fig 6.13 from AN-19-124
# region="signal"
# python plot_6_13.py -label $label -cat $category -y ${year} --region ${region}
# python plot_6_19.py -label $label -cat $category -y ${year} --region ${region}


# # -----------------------------------------------------
# year="all"

# region="z-peak"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# region="signal"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# region="h-sidebands"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# # plot Fig 6.13 from AN-19-124
# region="signal"
# python plot_6_13.py -label $label -cat $category -y ${year} --region ${region}
# python plot_6_19.py -label $label -cat $category -y ${year} --region ${region}

