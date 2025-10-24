#!/bin/bash
# Stop execution on any error
set -e

#
data_l="A B C D E F G H I"
# bkg_l="DY TT ST VV EWK OTHER"
bkg_l="DY"

# bkg_l="DY VV EWK OTHER"

sig_l="ggH VBF"

status="Private_Work"

# label="April19_NanoV12_JEROff"
# label="jetHornStudy_29Apr2025_JecOnJerOff"
# label="jetHornStudy_29Apr2025_JecOnJerOn"
# label="jetHornStudy_29Apr2025_JecOnJerOn_tightJetPuId"
# label="synchMay30_2025_b4PR"
# label="synchMay30_2025_afterPR"

# label="jetHornStudy_29Apr2025_JecOnJerStrat2_jetHornPtCut50"
# label="jetHornStudy_29Apr2025_JecOnJerStrat2_jetHornPtCut30"
# label="jetHornStudy_29Apr2025_JecOnJerOff_jetHornPtCut30"
# label="jetHornStudy_29Apr2025_JecOnJerStrat1_jetHornPtCut30"
# label="jetHornStudy_29Apr2025_JecOnJerStrat1n2_jetHornPtCut30"
# label="May09_2025_SynchWithRam"
# label="jetHornStudy_29Apr2025_JecOnJerStrat1n2"
# label="jetHornStudy_29Apr2025_JecOnJerStrat1n2_jetHornTightPuId"
# label="jetHornStudy_29Apr2025_JecOnJerOff"
# label="jetHornStudy_29Apr2025_JecOnJerStrat1"
# label="jetHornStudy_29Apr2025_JecOnJerStrat2"
# label="synchMay30_2025_b4PR"
# label="fullRun_May30_2025"
# label="synchJun21_2025_afterPR"
# label="synchJun21_2025_afterPR_latestZpt"
# label="fullRun_Jun21_2025"
# label="fullRun_Jun23_2025_1n2Revised"
# label="fullRun_Jun25_2025_DefJESJER"
# label="vbf_dy_validationMay30_2025"
# label="fullRun_Jun23_2025_vbfFilterStudy"
# label="Run2_nanoAODv12_08June"
# label="fullRun_Jul08_2025_BscFitOff"
# label="fullRun_Jul17_2025_qglFixed"
# label="fullRun05Aug_2025"
# label="fullRun_Jun23_2025_1n2Revised"
# label="Run3Oct21_2025"
label="Run3Oct22_2025_KITMuScaleSmearOn"

# year="2018"
# lumi="59.83"

# year="2017"
# lumi="41.48"

# year="2016postVFP"
# lumi="19.50"

# year="2016preVFP"
# lumi="16.81"

# lumi_dict = {
#     "2018" : 59.83,
#     "2017" : 41.48,
#     "2016postVFP": 19.50,
#     "2016preVFP": 16.81,
#     "2022preEE" : None,
# }

# load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/${label}/stage1_output_test/${year}/f0_1/"
# load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/${label}/stage1_output/${year}/f0_1/"
load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/${label}/stage1_output/${year}/f1_0/"
# load_path="/depot/cms/users/shar1172/hmm/copperheadV1clean/${label}/stage1_output/${year}/f1_0/"

# vars2plot="jet dijet dimuon mu"
# vars2plot="jet dijet"
vars2plot="jet dimuon"
# vars2plot="jet"
# vars2plot="dimuon "

# region="z-peak signal h-sidebands"
# region="h-sidebands"
# region="z-peak h-sidebands signal"
# region="h-sidebands"
# region="signal"
# region="z-peak signal"
region="z-peak"

# # python validation_plotter_unified.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat nocat ggh vbf -reg $region --label $label --use_gateway 
# # python validation_plotter_unified_quick.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat vbf -reg $region --label $label 
# # python validation_plotter_unified_quick.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat vbf -reg $region --label $label  --vbf_filter_study

# # python validation_plotter_unified.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat ggh -reg $region --label $label  --vbf_filter_study --use_gateway 
# python validation_plotter_unified.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat nocat ggh vbf -reg $region --label $label  --use_gateway 


# year="2017"
# lumi="41.48"
# load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/${label}/stage1_output/${year}/f1_0/"
# python validation_plotter_unified.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat nocat ggh vbf -reg $region --label $label  --use_gateway 

# year="2016postVFP"
# lumi="19.50"
# load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/${label}/stage1_output/${year}/f1_0/"
# python validation_plotter_unified.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat nocat ggh vbf -reg $region --label $label  --use_gateway 

# year="2016preVFP"
# lumi="16.81"
# load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/${label}/stage1_output/${year}/f1_0/"
# python validation_plotter_unified.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat nocat ggh vbf -reg $region --label $label  --use_gateway 


year="2024"
lumi="108.96"
load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/${label}/stage1_output/${year}/f1_0/"
# python validation_plotter_unified.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat nocat ggh vbf -reg $region --label $label  --use_gateway 
python validation_plotter_unified.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat nocat -reg $region --label $label  --use_gateway 