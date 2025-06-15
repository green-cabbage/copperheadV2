import awkward as ak
import dask_awkward as dak
import argparse
import sys
import os
import numpy as np
import json
from collections import OrderedDict
import cmsstyle as CMS
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib
plt.style.use(hep.style.CMS)
from omegaconf import OmegaConf
from modules.utils import fillSampleValues, getDimuMassBySubCat
from modules.basic_functions import filterRegion
import ROOT
import ROOT as rt
import copy

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# Add it to sys.path
sys.path.insert(0, parent_dir)
# Now you can import your module
from src.lib.histogram.plotting import plotFig_6_13


nSubCats = 5

def createUnitHistogram(name, nbins, xmin, xmax):
    hist = ROOT.TH1D(name, name, nbins, xmin, xmax)
    # Set all bin contents (1-based indexing for ROOT histograms)
    for bin_idx in range(1, nbins + 1):
        hist.SetBinContent(bin_idx, 1.0)
        hist.SetBinError(bin_idx, 0.0)

    return hist

def createUnitHistogram_roofit(x):
    name = "flat unit histogram"
    nbins = x.getBins()       # Number of bins (default or defined)
    xmin  = x.getMin()        # Lower range
    xmax  = x.getMax()        # Upper range
    print(f"nbins: {nbins}")
    print(f"xmin: {xmin}")
    print(f"xmax: {xmax}")
    th1 = createUnitHistogram(name, nbins, xmin, xmax)
    th1_rooHist = rt.RooDataHist(name, name, rt.RooArgSet(x), th1) 
    return th1_rooHist

def getColor(name):
    # color_list = [
    #     rt.kGreen,
    #     rt.kBlue,
    #     rt.kRed,
    #     rt.kOrange,
    #     rt.kViolet,
    # ]
    if "bwzredux" in name.lower():
        return rt.kRed, rt.kSolid
    elif "bwz" in name.lower() and "bern" in name.lower():
        return rt.kBlue
    elif "s-power" in name.lower():
        return rt.kCyan, rt.kDashDotted
    elif "s-exponential" in name.lower():
        return rt.kOrange, rt.kDashed 
    elif "bwz" in name.lower() and "gamma" in name.lower():
        return rt.kGreen, rt.kDotted
    elif "fewz" in name.lower() and "bern" in name.lower():
        return rt.kViolet
    elif "landau" in name.lower() and "bern" in name.lower():
        return rt.kGray
    else:
        print("Error, color not available for the function!")
        raise ValueError

def getBWZ_gamma(x):
    name = f"BWZ_a_coeff"
    a_coeff_bwz = rt.RooRealVar(name,name, -0.02,-0.5,0.5)
    name = "BWZ"
    BWZ = rt.RooModZPdf(name, name, x, a_coeff_bwz) 

    name = f"Gamma_a_coeff"
    a_coeff_gamma = rt.RooRealVar(name,name, -0.00005,-0.05,0.05)
    gamma = rt.RooGenericPdf("Gamma", "exp(@1*@0)/pow(@0,2)", rt.RooArgList(x, a_coeff_gamma))

    name = f"frac"
    frac = rt.RooRealVar(name,name, 0.5, 0.0, 1.0) 
    name = "BWZGamma"
    coreBWZGamma = rt.RooAddPdf(name, name, [BWZ, gamma], [frac])

    param_l = [
        a_coeff_bwz,
        BWZ,
        a_coeff_gamma,
        gamma,
        frac,
    ]# list of variables to return so that they don't get deleted in python functions. Otherwise the roofit pdfs don't work
    return coreBWZGamma, param_l

def plot_6_19(dataDict_by_subCat, save_fname, nSubCats=5):
    device = "cpu"
    nSubCats=1
    for target_subCat in range(nSubCats):
        dataDict_target = dataDict_by_subCat[target_subCat]
        mass_name = "mh_ggh"
        mass = rt.RooRealVar(mass_name, mass_name, 120, 110, 150)
        # nbins = 800
        nbins = 100
        mass.setBins(nbins)
        subCat_mass_arr  = ak.to_numpy(dataDict_target["dimuon_mass"]) # convert to numpy for rt.RooDataSet
        roo_datasetData = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
        roo_histData = rt.RooDataHist("rooHist_BWZRedux","rooHist_BWZRedux", rt.RooArgSet(mass), roo_datasetData)
        
        # fit BWZ redux
        name = f"BWZ_Redux_a_coeff"
        a_coeff = rt.RooRealVar(name,name, -0.02,-0.5,0.5)
        name = f"BWZ_Redux_b_coeff"
        b_coeff = rt.RooRealVar(name,name, -0.000111,-0.1,0.1)
        name = f"BWZ_Redux_c_coeff"
        c_coeff = rt.RooRealVar(name,name, 0.5,-10.0,10.0)
        name = "BWZRedux" # source: https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/5ae49dd944479b79af5692ff47fd7f1d9de16e91/interface/HMuMuRooPdfs.h#L11
        coreBWZRedux = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
        _ = coreBWZRedux.fitTo(roo_histData, EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
        fitResult = coreBWZRedux.fitTo(roo_histData, EvalBackend=device, PrintLevel=0 ,Save=True,)
        # print(f"fitResult: {fitResult}")

        # fit Sum exp
        name = f"RooSumTwoExpPdf_a1_coeff"
        a1_coeff = rt.RooRealVar(name,name, 0.00001,-2.0,1)
        name = f"RooSumTwoExpPdf_a2_coeff"
        a2_coeff = rt.RooRealVar(name,name, 0.1,-2.0,1)
        name = f"RooSumTwoExpPdf_f_coeff"
        f_coeff = rt.RooRealVar(name,name, 0.9,0.0,1.0)
    
        name = "S-Exponential"
        coreSumExp = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
        _ = coreSumExp.fitTo(roo_histData, EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
        fitResult = coreSumExp.fitTo(roo_histData, EvalBackend=device, PrintLevel=0 ,Save=True,)

        # fit Sum Power law
        name = f"RooSumTwoPowerLawPdf_a1_coeff"
        a1_coeff_pow = rt.RooRealVar(name,name, 0.00001,-2.0,1)
        name = f"RooSumTwoPowerLawPdf_a2_coeff"
        a2_coeff_pow = rt.RooRealVar(name,name, 0.1,-2.0,1)
        name = f"RooSumTwoPowerLawPdf_f_coeff"
        f_coeff_pow = rt.RooRealVar(name,name, 0.9,0.0,1.0)
    
        name = "S-Power-Law"
        coreSumPow = rt.RooSumTwoPowerLawPdf(name, name, mass, a1_coeff_pow, a2_coeff_pow, f_coeff_pow) 
        _ = coreSumPow.fitTo(roo_histData, EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
        fitResult = coreSumPow.fitTo(roo_histData, EvalBackend=device, PrintLevel=0 ,Save=True,)

        # fit BWZ Gamma
        coreBWZGamma, param_l_bwz_gamma = getBWZ_gamma(mass)
        _ = coreBWZGamma.fitTo(roo_histData, EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
        fitResult = coreBWZGamma.fitTo(roo_histData, EvalBackend=device, PrintLevel=0 ,Save=True,)
        print(f"coreBWZGamma : \n")
        fitResult.Print()
        # raise ValueError
        
        # plot
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        # Define upper and lower pads
        pad1 = ROOT.TPad("pad1", "Distribution", 0, 0.3, 1, 1.0)
        pad2 = ROOT.TPad("pad2", "Ratio", 0, 0.0, 1, 0.3)
        
        # Adjust margins
        pad1.SetBottomMargin(0)  # Upper plot does not need bottom margin
        pad2.SetTopMargin(0)     # Lower plot does not need top margin
        pad2.SetBottomMargin(0.3)

        pad1.SetTicks(2, 2)
        pad2.SetTicks(2, 2)
        pad1.Draw() # value plot
        pad2.Draw() # ratio plot
    
        pad1.cd()
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        frame = mass.frame()
        roo_histData.plotOn(frame)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),"Data", "P")
        
        color, style = getColor(coreBWZRedux.GetName())
        name = coreBWZRedux.GetName()
        coreBWZRedux.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
        
        # -----------------------------------------------------------
        color, style = getColor(coreSumExp.GetName())
        name = coreSumExp.GetName()
        coreSumExp.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

        # ---------------------------cond--------------------------------
        color, style = getColor(coreSumPow.GetName())
        name = coreSumPow.GetName()
        coreSumPow.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")

        # -----------------------------------------------------------
        color, style = getColor(coreBWZGamma.GetName())
        name = coreBWZGamma.GetName()
        coreBWZGamma.plotOn(frame, DataError="SumW2", Name=name, LineColor=color, LineStyle=style)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
        
        frame.Draw()
        legend.Draw()
        

        # ratio
        pad2.cd()
        ratio_frame= mass.frame()
        

        flat_unit_hist = createUnitHistogram_roofit(mass)
        # Note: DataError=None flag is needed to set our GetYaxis().SetRangeUser() to our desired values. Otherwise, the size of the error bars plays a role.
        flat_unit_hist.plotOn(ratio_frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0), Invisible=True, DataError=None)

         
        flat_pdf = ROOT.RooPolynomial("BWZRedux ratio", "BWZRedux ratio", mass)
        color, style = getColor(flat_pdf.GetName())
        flat_pdf.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)
        # -----------------------------------------------------------
        
        ratio_sumExp = rt.RooGenericPdf("S-Exponential ratio", "@0/@1", rt.RooArgList(coreBWZRedux,coreSumExp))
        color, style = getColor(ratio_sumExp.GetName())
        ratio_sumExp.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)
        # -----------------------------------------------------------

        ratio_sumPow = rt.RooGenericPdf("S-Power-Law ratio", "@0/@1", rt.RooArgList(coreBWZRedux,coreSumPow))
        color, style = getColor(ratio_sumPow.GetName())
        ratio_sumPow.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)

        # -----------------------------------------------------------

        ratio_bwzGamma = rt.RooGenericPdf("BWZGamma ratio", "@0/@1", rt.RooArgList(coreBWZRedux,coreBWZGamma))
        color, style = getColor(ratio_bwzGamma.GetName())
        ratio_bwzGamma.plotOn(ratio_frame, DataError="SumW2", LineColor=color, LineStyle=style)
        
        # set ranges and label sizes after all things are plotted
        ratio_frame.GetXaxis().SetLabelSize(0.10)
        ratio_frame.SetXTitle(f"Dimuon Mass (GeV)")
        ratio_frame.GetXaxis().SetTitleSize(0.10)
        
        ratio_frame.GetYaxis().SetLabelSize(0.08)
        ratio_frame.GetYaxis().SetRangeUser(0.98, 1.02)
        ratio_frame.SetTitle("")
        ratio_frame.Draw()
        
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{save_fname}_subCat{target_subCat}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-label",
    "--label",
    dest="label",
    default="",
    action="store",
    help="label",
    )
    parser.add_argument(
    "-cat",
    "--category",
    dest="category",
    default="ggH",
    action="store",
    help="string value production category we're working on",
    )
    parser.add_argument(
    "-save",
    "--save_path",
    dest="save_path",
    default="plots",
    action="store",
    help="string value production category we're working on",
    )
    # parser.add_argument(
    # "-samp",
    # "--samples",
    # dest="samples",
    # default=[],
    # nargs="*",
    # type=str,
    # action="store",
    # help="list of samples to process for stage2. Current valid inputs are data, signal and DY",
    # )
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="all",
    action="store",
    help="label",
    )
    parser.add_argument(
    "-reg",
    "--region",
    dest="region",
    default="signal",
    action="store",
    help="region value to plot, available regions are: h_peak, h_sidebands, z_peak and signal (h_peak OR h_sidebands)",
    )
    args = parser.parse_args()
    # load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/{args.category}/stage2_output/*/"
    year = args.year
    if year == "all":
        year = "*"
    load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/{args.category}/stage2_output/{year}/"
    # events = dak.from_parquet(f"{load_path}/*data.parquet")
    # print(events.fields)
    print(f"load_path : {load_path}")
    lumi_dict = {
        "2018" : 59.83,
        "2017" : 41.48,
        "2016postVFP": 19.50,
        "2016preVFP": 16.81,
        "all" : 137,
    }
    lumi_val = lumi_dict[year]
    sample_groups = {
        "data" : "data*",
    }
    sample_dict = {
        group: {
            "wgt_nominal" : [],
            "dimuon_mass": [],
            "subCategory_idx": [],
        } for group in sample_groups.keys()
    }
    if args.region != "signal":
        print("Error, region is not signal!")
        raise ValueError
    for group, group_fname in sample_groups.items():
        full_load_path = load_path+f"*{group_fname}.parquet" 
        events = dak.from_parquet(full_load_path)
        events = filterRegion(events, region=args.region)
        sample_dict = fillSampleValues(events, sample_dict, group)


    plot_setting_fname = "../../../src/lib/histogram/plot_settings_vbfCat_MVA_input.json"
    plot_setting_fname = "plot_settings_vbfCat_MVA_input.json"
    with open(plot_setting_fname, "r") as file:
        plot_settings = json.load(file)
    plot_var = "BDT_score"
    binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    save_fname = "plots/Fig6_19"
    # status = "Private"
    status = "Simulation"
    dataDict_by_subCat = getDimuMassBySubCat(sample_dict, sample="data", nSubCats=nSubCats)
    # print(f"sample_dict: {sample_dict}")
    print(f"dataDict_by_subCat: {dataDict_by_subCat}")
    
    plot_6_19(dataDict_by_subCat, save_fname)
    

    
