#!/bin/bash
# Stop execution on any error
set -e

category="vbf"
# label="Jan07_test"
# label="fullRun_Jun23_2025_1n2Revised"
# label="jetHornStudy_29Apr2025_JecOnJerStrat1n2_jetHornTightPuId"

label="Run2_nanoAODv12_08June"


# load_path="/depot/cms/users/shar1172/hmm/copperheadV1clean/${label}/stage1_output/${year}/f1_0/"


# python dnn_preprocessor.py --label $label  -cat $category
python dnn_train.py --label $label 