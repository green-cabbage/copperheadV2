#!/bin/bash
set -e

# -------------------------------------------------------
#  pre-fit signal significance
# -------------------------------------------------------
# best_idx="0" # sumExp
best_idx="1" # BWZRedux
# best_idx="2" # FEWZxBern

combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat0_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat0"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
echo "${category} pre-fit Expected Significance:" >> significance.txt
combine -M Significance -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
cat comb_expSignificance.log | grep "Significance:" >> significance.txt


combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat1_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat1"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
echo "${category} pre-fit Expected Significance:" >> significance.txt
combine -M Significance -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
cat comb_expSignificance.log | grep "Significance:" >> significance.txt


combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat2_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat2"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
echo "${category} pre-fit Expected Significance:" >> significance.txt
combine -M Significance -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
cat comb_expSignificance.log | grep "Significance:" >> significance.txt


combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat3_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat3"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
echo "${category} pre-fit Expected Significance:" >> significance.txt
combine -M Significance -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
cat comb_expSignificance.log | grep "Significance:" >> significance.txt


combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh.txt >  datacard_comb_sig_cat4_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat4"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
echo "${category} pre-fit Expected Significance:" >> significance.txt
combine -M Significance -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
cat comb_expSignificance.log | grep "Significance:" >> significance.txt


combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh_test.txt datacard_cat2_ggh_test.txt datacard_cat3_ggh_test.txt datacard_cat4_ggh_test.txt >  datacard_comb_sig_all_ggh.txt # NOTE: combine cards command is different for EACH category
category="all"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
echo "combined pre-fit Expected Significance:" >> significance.txt
combine -M Significance -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
cat comb_expSignificance.log | grep "Significance:" >> significance.txt


# -------------------------------------------------------
#  post-fit signal significance
# -------------------------------------------------------

# combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat0_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat0"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# echo "${category} post-fit Expected Significance:" >> significance.txt
# combine -M Significance -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
# cat comb_expSignificance.log | grep "Significance:" >> significance.txt

# combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat1_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat1"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# echo "${category} post-fit Expected Significance:" >> significance.txt
# combine -M Significance -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
# cat comb_expSignificance.log | grep "Significance:" >> significance.txt

# combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat2_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat2"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# echo "${category} post-fit Expected Significance:" >> significance.txt
# combine -M Significance -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
# cat comb_expSignificance.log | grep "Significance:" >> significance.txt

# combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat3_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat3"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# echo "${category} post-fit Expected Significance:" >> significance.txt
# combine -M Significance -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
# cat comb_expSignificance.log | grep "Significance:" >> significance.txt

# combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh.txt >  datacard_comb_sig_cat4_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat4"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# echo "${category} post-fit Expected Significance:" >> significance.txt
# combine -M Significance -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
# cat comb_expSignificance.log | grep "Significance:" >> significance.txt


# combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh_test.txt datacard_cat2_ggh_test.txt datacard_cat3_ggh_test.txt datacard_cat4_ggh_test.txt >  datacard_comb_sig_all_ggh.txt # NOTE: combine cards command is different for EACH category
# category="all"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# echo "combined post-fit Expected Significance:" >> significance.txt
# combine -M Significance -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
# cat comb_expSignificance.log | grep "Significance:" >> significance.txt



# -------------------------------------------------------
#  pre-fit MultiDimFit signal strength fit
# -------------------------------------------------------
# best_idx="0" # sumExp
best_idx="1" # BWZRedux
# best_idx="2" # FEWZxBern


combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat0_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat0"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
combine -M MultiDimFit --algo singles -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root --rMin=0 --rMax=2 > multidim_${category}Singles.log
echo "pre-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt

combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat1_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat1"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
combine -M MultiDimFit --algo singles -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root  > multidim_${category}Singles.log
echo "pre-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt

combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat2_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat2"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
combine -M MultiDimFit --algo singles -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root  > multidim_${category}Singles.log
echo "pre-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt

combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat3_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat3"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
combine -M MultiDimFit --algo singles -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root  > multidim_${category}Singles.log
echo "pre-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt

combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh.txt >  datacard_comb_sig_cat4_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat4"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
combine -M MultiDimFit --algo singles -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root  > multidim_${category}Singles.log
echo "pre-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt


combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh_test.txt datacard_cat2_ggh_test.txt datacard_cat3_ggh_test.txt datacard_cat4_ggh_test.txt >  datacard_comb_sig_all_ggh.txt # NOTE: combine cards command is different for EACH category
category="all"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
combine -M MultiDimFit --algo singles -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root  > multidim_${category}Singles.log
echo "pre-fit combined multiDim fit using Asimov toys:"  >> signal_strengths.txt
cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt



# -------------------------------------------------------
#  post-fit MultiDimFit signal strength fit
# -------------------------------------------------------



# combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat0_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat0"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# combine -M MultiDimFit --algo singles -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > multidim_${category}Singles.log
# echo "post-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
# cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt



# combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat1_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat1"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# combine -M MultiDimFit --algo singles -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > multidim_${category}Singles.log
# echo "post-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
# cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt


# combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat2_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat2"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# combine -M MultiDimFit --algo singles -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > multidim_${category}Singles.log
# echo "post-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
# cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt



# combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat3_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat3"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# combine -M MultiDimFit --algo singles -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > multidim_${category}Singles.log
# echo "post-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
# cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt


# combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh.txt >  datacard_comb_sig_cat4_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat4"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# combine -M MultiDimFit --algo singles -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > multidim_${category}Singles.log
# echo "post-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
# cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt


# combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh_test.txt datacard_cat2_ggh_test.txt datacard_cat3_ggh_test.txt datacard_cat4_ggh_test.txt >  datacard_comb_sig_all_ggh.txt # NOTE: combine cards command is different for EACH category
# category="all"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# combine -M MultiDimFit --algo singles -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > multidim_${category}Singles.log
# echo "post-fit combined multiDim fit using Asimov toys:"  >> signal_strengths.txt
# cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt


# -------------------------------------------------------
# expected NLL scan
# -------------------------------------------------------
# Re-run that study before changing best_idx.
best_idx="0"   # 0 = sumExp, 1 = BWZRedux, 2 = FEWZxBern

npts="100"
card="datacard_comb_sig_all_ggh"
idle="BWZ_Redux_a_coeff,BWZ_Redux_b_coeff,BWZ_Redux_c_coeff,FEWZxBern_c2,FEWZxBern_c3,FEWZxBern_c4"

# the POI is called r inside combine; display it as mu
printf '{"r": "#mu"}\n' > poi_translate.json

text2workspace.py ${card}.txt -m 125

combine -M MultiDimFit -d ${card}.root \
    --algo grid \
    --points ${npts} \
    --setParameterRanges r=0,3 \
    --setParameters pdf_index_ggh=${best_idx} \
    --freezeParameters MH,pdf_index_ggh,${idle} \
    --cminDefaultMinimizerStrategy 0 \
    --saveNLL \
    -n fixed_pdf_${best_idx} \
    -m 125 \
    -t -1 --expectSignal 1 \
    --X-rtd REMOVE_CONSTANT_ZERO_POINT=1

# --translate relabels the POI as mu on both the x axis and the
PLOT1DSCAN="${PLOT1DSCAN:-plot1DScan.py}"
python3 "${PLOT1DSCAN}" higgsCombinefixed_pdf_${best_idx}.MultiDimFit.mH125.root \
    --main-label Expected \
    --translate poi_translate.json \
    -o nll_scan_idx${best_idx}

# # -------------------------------------------------------
# # for printing plotGof
# # -------------------------------------------------------

# index="1"
# # cat="cat4"
# cat="all"

# text2workspace.py -m 125 datacard_comb_sig_${cat}_ggh.txt

# combine -M GoodnessOfFit datacard_comb_sig_${cat}_ggh.root --algo saturated -m 125 --freezeParameters MH -n .goodnessOfFit_data

# combine -M GoodnessOfFit datacard_comb_sig_${cat}_ggh.root --algo saturated -m 125 --freezeParameters MH -n .goodnessOfFit_toys -t 100 #1050

# combineTool.py -M CollectGoodnessOfFit --input higgsCombine.goodnessOfFit_data.GoodnessOfFit.mH125.root higgsCombine.goodnessOfFit_toys.GoodnessOfFit.mH125.123456.root -m 125.0 -o gof.json

# plotGof.py gof.json --statistic saturated --mass 125.0 -o gof_plot