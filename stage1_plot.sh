#!/bin/bash
# Stop execution on any error
set -e

#
data_l="A B C D E F G H"
bkg_l="DY TT ST VV EWK OTHER"
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
label="fullRun_May30_2025"


year="2018"
lumi="59.97"

# year="2017"
# lumi="41.5"

# lumi_dict = {
#     "2018" : 59.97,
#     "2017" : 41.5,
#     "2016postVFP": 19.5,
#     "2016preVFP": 16.8,
#     "2022preEE" : None,
# }


# label="fullRun_May30_2025"
# load_path="/depot/cms/users/shar1172/hmm/copperheadV1clean/${label}/stage1_output/${year}/f1_0/"
# load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/${label}/stage1_output_test/${year}/f0_1/"
load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/${label}/stage1_output/${year}/f1_0/"


vars2plot="jet dijet dimuon mu"
# vars2plot="jet dijet"
# vars2plot="dimuon"
# vars2plot="jet"
# vars2plot="mu"

region="z-peak signal h-peak h-sidebands"
python validation_plotter_unified.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat nocat -reg $region --label $label --use_gateway 
python validation_plotter_unified.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat vbf -reg $region --label $label --use_gateway 
python validation_plotter_unified.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat ggh -reg $region --label $label --use_gateway 

