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
# label="BSC_off_Aug26_2025"

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
# model_name="V2_fullRun_Jun21_2025_1n2Revised"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_BigEbeMassResContrib_Aug25_2025_DYSigOnly_BSC_Off"
# model_name="V2_Aug28_PosWgtRun0p7_removeForwardJet_redo_MassResPow4"
model_name="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_setRandomSeed_annhilateWeight"


# # category="${model_name}_ggh"
# # category="${model_name}_ggh_w_allYearBDT"
# category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields"
# # category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes"
# # category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_NoJ1Eta_MinMmjdEta"
# # category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_DimuVarsOnly"
# # category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_NoSingleJet_Mu2_MinMmjdEta"
# # category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_NoSingleJet_MinMmjVars"
# # category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_NoJet_Mu2_MinMmjVars"
# # category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_Dimu_Mu1_MinMmjdEta_Vars"
# # category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_Dimu_Mu1_Mu2_Vars"
# # category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_Dimu_Mu1_jet1Pt_Vars"
# # category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_Dimu_Mu1_Mu2_jet1Pt_Vars"
# # category="${model_name}_ggh_w_allYearBDT_w_newBDTTargetYields_noEbeMassRes_Dimu_Mu1_Mu2_jet1jet2Pt_Vars"
category="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_recreate1_87SigOct31_2025_newEdgeTarget/stage2_output" 

# year="2018"
# year="2017"
# year="2016"
# year="2016postVFP"
# year="2016preVFP"
year="all"

region="z-peak"
python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

region="signal"
python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

region="h-sidebands"
python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# # plot Fig 6.13 from AN-19-124
region="signal"
python plot_6_8.py -label $label -cat $category -y ${year} --region ${region}
python plot_6_13.py -label $label -cat $category -y ${year} --region ${region}
python plot_6_19.py -label $label -cat $category -y ${year} --region ${region}
python getTable_6_2And6_12.py -label $label -cat $category -y ${year} --region ${region}
python plot_6_7.py -label $label -cat $category -y ${year} --region ${region}

# # # -----------------------------------------------------
# # plot 6.7 
# # model_name="V2_fullRun_Jun21_2025_1n2Revised"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_BigEbeMassResContrib_Aug25_2025"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_BigEbeMassResContrib_Aug25_2025_orig"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_BigEbeMassResContrib_Aug25_2025_wgt_1e6"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_BigEbeMassResContrib_Aug25_2025_noSigWgt"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_BigEbeMassResContrib_Aug25_2025_DYSigOnly"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_BigEbeMassResContrib_Aug25_2025_DYSigOnly_BSC_Off"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_BigEbeMassResContrib_Aug25_2025_DYSigOnly_BSC_Off_oldBDT"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_BigEbeMassResContrib_Aug25_2025_DYSigOnly_BSC_Off_oldBDT_amcDY"
# # model_name="V2_UL_Jan18_2025"
# # model_name="V2_UL_Jan19_2025_addTTST_Aug26"
# # model_name="V2_UL_Jan19_2025_addTTST_Aug28_MiNNLO"
# # model_name="V2_UL_Mar24_2025_Aug28_MiNNLO_scalePosWgtOff"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainOff"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainOff_noVBFsig"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainOff_noVBFsig_noJetEtas"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainOff_noVBFsig_noJetVars1"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainOff_noVBFsig_noJetVars2"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainOff_noVBFsig_noJetVars3"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainOff_noVBFsig_noJetVars4"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly_noVBFsig_noJetVars4"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noVBFsig_noJetVars4"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noVBFsig_noJetVars3"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noVBFsig_noJetVars5"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noVBFsig_noJetVars6"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noVBFsig_noJetVars7"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noVBFsig_jjdPhi"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noVBFsig_dPhisOn"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noVBFsig_noJetVars8"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noVBFsig_noJetVars9"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noVBFsig_noJetVars10"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noVBFsig_noJetVars11"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noVBFsig_noJetVars12"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noVBFsig_noJetVars13"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noVBFsig_noJetVars13_1n2Revised"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noJetVars13_1n2Revised"
# # model_name="V2_fullRun_Jun21_2025_1n2Revised_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noVBFsig_noJetVars13_bkgWgtApplied"
# # model_name="V2_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noVBFsig_noJetVars13_bkgWgtApplied_removeForwardJet"
# # model_name="V2_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noJetVars13_bkgWgtApplied_removeForwardJet"
# # model_name="V2_Aug28_scalePosWgtOff_AllMCAllYr_noEarlyStop_wgtTrainEbeMassResonly4Sig_noJetVars14_bkgWgtApplied_removeForwardJet"
# # model_name="V2_Aug28_scalePosWgtOff_AllYears_noEarlyStop_wgtTrainEbeMassResonly4Sig_noJetVars14_bkgWgtApplied_removeForwardJet"
# # model_name="V2_Aug28_scalePosWgtOff_AllYears_noEarlyStop_wgtTrainEbeMassResonly4Sig_noJetVars15_bkgWgtApplied_removeForwardJet"
# # model_name="V2_Aug28_scalePosWgtOff_AllYears_noEarlyStop_wgtTrainEbeMassResonly4Sig_noJetVars16_bkgWgtApplied_removeForwardJet"
# # model_name="V2_Aug28_scalePosWgtOff_AllYears_noEarlyStop_wgtTrainEbeMassResonly4Sig_noJetVars16_bkgWgtApplied_removeForwardJet_AddYear"
# # model_name="V2_Aug28_scalePosWgtOff_AllYears_noEarlyStop_wgtTrainEbeMassResonly4Sig_noJetVars16_bkgWgtApplied_removeForwardJet_AddYear_posSumWgt"
# # model_name="V2_Aug28_scalePosWgtOff_AllYears_noEarlyStop_wgtTrainEbeMassResonly4Sig_noJetVars16_bkgWgtApplied_removeForwardJet_AddYear_fillNoneChange"
# # model_name="V2_Aug28_scalePosWgtOff_AllYears_noEarlyStop_wgtTrainEbeMassResonly4Sig_noJetVars16_bkgWgtApplied_removeForwardJet_AddYear_fillNoneChange_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_noJetVars16_bkgWgtApplied_removeForwardJet_AddYear_RemoveNegWgt_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_noJetVars16_bkgWgtApplied_removeForwardJet_AddYear_RemoveNegWgt_2018Only_inclRestBkgMC"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_noJetVars16_bkgWgtApplied_removeForwardJet_AddYear_RemoveNegWgt_inclRestBkgMC"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_removeForwardJet_AddYear_RemoveNegWgt_inclRestBkgMC"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_removeForwardJet_AddYear_RemoveNegWgt_inclRestBkgMC_bigEbeMassContr"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_removeForwardJet_AddYear_RemoveNegWgt_inclRestBkgMC_bigEbeMassContr_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_removeForwardJet_AddYear_RemoveNegWgt_inclRestBkgMC_bigEbeMassContr2_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_removeForwardJet_AddYear_RemoveNegWgt_inclRestBkgMC_bigEbeMassContr3_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_removeForwardJet_AddYear_RemoveNegWgt_inclRestBkgMC_bigEbeMassContr3_NormalizeSigBdtWgt_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_removeForwardJet_AddYear_RemoveNegWgt_inclRestBkgMC_bigEbeMassContr2_NormalizeSigBdtWgt_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_removeForwardJet_AddYear_RemoveNegWgt_inclRestBkgMC_bigEbeMassContr_NormalizeSigBdtWgt_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_removeForwardJet_AddYear_RemoveNegWgt_inclRestBkgMC_bigEbeMassContr4_NormalizeSigBdtWgt_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_removeForwardJet_AddYear_RemoveNegWgt_inclRestBkgMC_bigEbeMassContr3_addRpt_NormalizeSigBdtWgt_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_removeForwardJet_AddYear_AnnhilateNegWgt_inclRestBkgMC_bigEbeMassContr3_addRpt_NormalizeSigBdtWgt_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_removeForwardJet_AddYear_AnnhilateNegWgt_inclRestBkgMC_bigEbeMassContr3_addRpt_NormalizeSigBdtWgt_addAllUlYr"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_removeForwardJet_AddYear_AnnhilateNegWgt_inclRestBkgMC_bigEbeMassContr5_addRpt_NormalizeSigBdtWgt_addAllUlYr"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_removeForwardJet_AddYear_AnnhilateNegWgt_inclRestBkgMC_bigEbeMassContr6_addRpt_NormalizeSigBdtWgt_addAllUlYr"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_removeForwardJetFix_AddYear_AnnhilateNegWgt_inclRestBkgMC_bigEbeMassContr3_addRpt_NormalizeSigBdtWgt_addAllUlYr"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_bigEbeMassContr3_addRpt_NormalizeSigBdtWgt_addAllUlYr"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_addEbeFacDY_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_addEbeFacDY_ebeMassQuad_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_addEbeFacDY_ebeMassSq_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_addEbeFacDY_ebeMassSq_PosWgtRun1_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_addEbeFacDY_ebeMassSq_PosWgtRun2_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_addEbeFacDY_ebeMassSq_PosWgtRun3_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_addEbeFacDY_ebeMassSq_PosWgtRun4_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_addEbeFacDY_ebeMassSq_PosWgtRun5_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgtApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_addEbeFacDY_ebeMassSq_PosWgtRun5_removeForwardJet_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgctApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_addEbeFacDY_ebeMassSq_PosWgtRun4_removeForwardJet_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgctApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_addEbeFacDY_ebeMassSq_PosWgtRun3_removeForwardJet_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgctApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_addEbeFacDY_ebeMassSq_PosWgtRun2_removeForwardJet_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgctApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_addEbeFacDY_ebeMassSq_PosWgtRun1_removeForwardJet_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgctApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_addEbeFacDY_ebeMassSq_removeForwardJet_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgctApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_addEbeFacDY_ebeMassSq_PosWgtRun6_removeForwardJet_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgctApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_addEbeFacDY_ebeMassSq_PosWgtRun7_removeForwardJet_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgctApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_ebeMassSq_PosWgtRun6_removeForwardJet_2018Only"
# # model_name="V2_Aug28_scalePosWgtOff_wgtTrainEbeMassResonly4Sig_bkgWgctApplied_AddYear_AnnhilateNegWgt_inclRestBkgMC_NormalizeEbeMassRes_addRpt_NormalizeSigBdtWgt_addAllUlYr_addEbeFacDY_ebeMassSq_PosWgtRun6_removeForwardJet_2018Only_inclOtherBkg"
# # model_name="V2_Aug28_PosWgtRun6_removeForwardJet_2018Only_inclOtherBkgButOnlyDyInTrain"
# # model_name="V2_Aug28_PosWgtRun6_removeForwardJet_inclOtherBkgButOnlyDyInTrain_2018Only"
# # model_name="V2_Aug28_PosWgtRun6_removeForwardJet_2018Only"
# # model_name="V2_Aug28_PosWgtRun6_removeForwardJet_inclOtherBkgButOnlyDyInTrainDfValInc_2018Only"
# # model_name="V2_Aug28_PosWgtRun6_removeForwardJet_redo_2018Only"
# # model_name="V2_Aug28_PosWgtRun6_removeForwardJet_redo"
# # model_name="V2_Aug28_PosWgtRun0p7_removeForwardJet_redo"
# # model_name="V2_Aug28_PosWgtRun0p9_removeForwardJet_redo"
# # model_name="V2_Aug28_PosWgtRun1p0_removeForwardJet_redo"
# # model_name="V2_Aug28_PosWgtRun1p1_removeForwardJet_redo"
# # model_name="V2_Aug28_PosWgtRun1p3_removeForwardJet_redo"
# # model_name="V2_Aug28_PosWgtRun0p7_removeForwardJet_redo_MassResPow4"
# # model_name="V2_Aug28_PosWgtRun0p7_MassResRun1"
# # model_name="V2_Aug28_PosWgtRun0p7_MassResRun2"
# # model_name="V2_Aug28_PosWgtRun0p7_MassResRun3"
# # model_name="V2_Aug28_PosWgtRun0p7_MassResRun4"
# # model_name="V2_Aug28_PosWgtRun0p7_MassResRun5"
# # model_name="V2_Aug28_PosWgtRun0p7_MassResRun1_removeForwardJets"
# # model_name="V2_Aug16_2025Reprod_IssueNum2"
# # model_name="V2_Aug16_2025DefaultFillNoneBack_IssueNum2"
# # model_name="V2_Aug16_2025DefaultFillNoneBack_AnnhilateNegWgts_IssueNum2"
# # model_name="V2_Aug16_2025DefaultFillNoneBackFix_AnnhilateNegWgts_IssueNum2"
# # model_name="V2_Aug16_2025DefaultFillNoneBackFix_IssueNum2"
# # model_name="V2_Aug16_2025DefaultFillNoneBackFix_AnnhilateNegWgts_AddRpt_IssueNum2"
# # model_name="V2_Aug16_2025DefaultFillNoneBackFix_AnnhilateNegWgts_AddRptNoEarlyStop_IssueNum2"
# model_name="V2_Aug16_2025AddIssue1To3_IssueNum2"

# category="${model_name}_ggh" # stage2 ouput name
# year="all"
# # year="2018"
# # year="2017"
# label="fullRun_Jun23_2025_1n2Revised"
# region="signal"
# python plot_6_7.py -label $label -cat $category -y ${year} --region ${region}
# # python plot_6_8.py -label $label -cat $category -y ${year} --region ${region}
# # python getTable_6_2And6_12.py -label $label -cat $category -y ${year} --region ${region}
# # python plot_6_13.py -label $label -cat $category -y ${year} --region ${region}
# # python plot_6_19.py -label $label -cat $category -y ${year} --region ${region}

# # year="2018"
# # python plot_6_7.py -label $label -cat $category -y ${year} --region ${region}
# # year="2017"
# # python plot_6_7.py -label $label -cat $category -y ${year} --region ${region}
# # year="2016postVFP"
# # python plot_6_7.py -label $label -cat $category -y ${year} --region ${region}
# # year="2016preVFP"
# # python plot_6_7.py -label $label -cat $category -y ${year} --region ${region}


# # # -----------------------------------------------------
# # year="all"

# # region="z-peak"
# # python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# # region="signal"
# # python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# # region="h-sidebands"
# # python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# # # plot Fig 6.13 from AN-19-124
# # region="signal"
# # python plot_6_13.py -label $label -cat $category -y ${year} --region ${region}
# # python plot_6_19.py -label $label -cat $category -y ${year} --region ${region}

