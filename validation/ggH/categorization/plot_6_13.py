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

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# Add it to sys.path
sys.path.insert(0, parent_dir)
# Now you can import your module
from src.lib.histogram.plotting import plotFig_6_13



def filterRegion(events, region="h-peak"):
    dimuon_mass = events.dimuon_mass
    if region =="h-peak":
        region = (dimuon_mass > 115.03) & (dimuon_mass < 135.03)
    elif region =="h-sidebands":
        region = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))
    elif region =="signal":
        region = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
    elif region =="z-peak":
        region = (dimuon_mass >= 70) & (dimuon_mass <= 110.0)

    events = events[region]
    return events

def tranformBDT_score(computed_zip):
    """
    helper function that changes the range from [0,1] to [-1,-1]
    """
    score_name = "BDT_score"
    BDT_score = computed_zip[score_name]
    computed_zip[score_name] = (BDT_score-0.5)*2
    return computed_zip

def tranformBDT_edges(bin_edges):
    """
    helper function that transforms bin edges from [0,1] to [-1,-1]
    """
    transformed_edges = []
    for bin_edge in bin_edges:
        if bin_edge <=0 or bin_edge>=1:
            continue
        transformed_edge = (bin_edge-0.5)*2
        transformed_edges.append(transformed_edge)
    return transformed_edges

def fillSampleValues(events, sample_dict, sample: str):
    # find which sample group sample_name belongs to
    if sample in sample_dict.keys():
        fields2load = ["wgt_nominal", "BDT_score", "dimuon_mass", "subCategory_idx"]
        
        # compute in parallel fields to load
        computed_zip = ak.zip({
            field : events[field] for field in fields2load
        }).compute()
        computed_zip = tranformBDT_score(computed_zip)
        for field in fields2load:
            # sample_dict[sample][field].append(
            #     ak.to_numpy(computed_zip[field])
            # )
            sample_dict[sample][field] = ak.to_numpy(computed_zip[field])
            
    else:
        print(f"sample {sample} not present in sample_dict!")

    return sample_dict


def getHWHM(fwhm, counts, bin_centers):
    hwhm = fwhm/2
    max_ix = np.argmax(counts)
    max_center = bin_centers[max_ix]
    bin_center_l = max_center - hwhm/2
    bin_center_r = max_center + hwhm/2
    # print(f"hwhm: {hwhm:.3f}")
    # print(f"max_center: {max_center:.3f}, Left: {bin_center_l:.3f}, Right: {bin_center_r:.3f}")
    
    return bin_center_l, bin_center_r
    
def compute_hwhm_with_edges(counts, bin_edges):
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    max_count = np.max(counts)
    half_max = max_count / 2

    # Identify bins where counts are above half max
    above_half_max = counts >= half_max
    indices = np.where(above_half_max)[0]

    if len(indices) < 2:
        return None, None, None

    left_idx = indices[0]
    right_idx = indices[-1]

    fwhm = bin_centers[right_idx] - bin_centers[left_idx]
    bin_center_l, bin_center_r = getHWHM(fwhm, counts, bin_centers)
    hwhm = fwhm/2
    return hwhm, bin_center_l, bin_center_r

def getDimuMassBySubCat(sample_dict, sample="", nSubCats=5):
    dimuon_mass = sample_dict[sample]["dimuon_mass"]
    wgt_nominal = sample_dict[sample]["wgt_nominal"]
    subCat_ixs = sample_dict[sample]["subCategory_idx"]
    dict_by_subCat = {}
    for target_subCat in range(nSubCats):
        subCat_filter = target_subCat == subCat_ixs
        dimuon_mass_subCat = dimuon_mass[subCat_filter]
        wgt_nominal_subCat = wgt_nominal[subCat_filter]
        subCat_dict = {
            "dimuon_mass" : dimuon_mass_subCat,
            "wgt_nominal" : wgt_nominal_subCat,
        }
        dict_by_subCat[target_subCat] = subCat_dict
    return dict_by_subCat

def getHWHM_withEdges(sample_dict, binning):
    dimuon_mass = sample_dict["dimuon_mass"]
    wgt_nominal = sample_dict["wgt_nominal"]
    hist, _ = np.histogram(dimuon_mass, bins=binning, weights=wgt_nominal)
    # print(f"hist: {hist}")
    # print(f"hist len: {len(hist)}")
    hwhm, bin_center_l, bin_center_r = compute_hwhm_with_edges(hist, binning)
    return hwhm, bin_center_l, bin_center_r

def getYield_hwhm(sample_dict, hwhm_left, hwhm_right):
    dimuon_mass = sample_dict["dimuon_mass"]
    wgt_nominal = sample_dict["wgt_nominal"]
    
    hwhm_filter = (hwhm_left <= dimuon_mass) & (hwhm_right >= dimuon_mass)
    hwhm_yield = np.sum(wgt_nominal[hwhm_filter])
    # print(f"hwhm_yield: {hwhm_yield}")
    return hwhm_yield

def getSignificanceHist(sample_dict, nSubCats=5):
    # divide the dimuon mass arrays by subcategory values
    sigDict_by_subCat = getDimuMassBySubCat(sample_dict, sample="signal", nSubCats=nSubCats)
    bkgDict_by_subCat = getDimuMassBySubCat(sample_dict, sample="background", nSubCats=nSubCats) 
    # print(f"sigDict_by_subCat: {sigDict_by_subCat}")
    # print(f"bkgDict_by_subCat: {bkgDict_by_subCat}")
    
    # calculate HWHM and the edges
    signficanceBySubCat = {}
    for subCat_target in range(nSubCats):
        binning = np.linspace(110, 150, 51) # signal fit region
        # print(f"sigDict_by_subCat[subCat_target]: {sigDict_by_subCat[subCat_target]}")
        hwhm, hwhm_left, hwhm_right = getHWHM_withEdges(sigDict_by_subCat[subCat_target], binning)
        # print(f"hwhm: {hwhm}")
        # print(f"hwhm_left: {hwhm_left}")
        # print(f"hwhm_right: {hwhm_right}")
        
        sigYield_hwhm = getYield_hwhm(sigDict_by_subCat[subCat_target], hwhm_left, hwhm_right)
        bkgYield_hwhm = getYield_hwhm(bkgDict_by_subCat[subCat_target], hwhm_left, hwhm_right)
        # print(f"sigYield_hwhm: {sigYield_hwhm}")
        # print(f"bkgYield_hwhm: {bkgYield_hwhm}")
        # sigYield_hwhm = sigYield_hwhm *3
        # bkgYield_hwhm = bkgYield_hwhm *3
        significance = sigYield_hwhm/ (bkgYield_hwhm**(0.5))
        signficanceBySubCat[subCat_target] = significance
        # print(f"subcat {subCat_target} significance: {significance}")
    print(f"signficanceBySubCat: {signficanceBySubCat}")
    # convert the dictionary with significance values to np his arrays
    significanceHist = np.zeros(nSubCats)
    for subcat, significance in signficanceBySubCat.items():
        significanceHist[subcat] = significance
        
    print(f"significanceHist: {significanceHist}")
    return significanceHist


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
        "signal" : "sigMC*",
        "background" : "bkgMC*",
    }
    sample_dict = {
        group: {
            "wgt_nominal" : [],
            "BDT_score": [],
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
    bkg_MC = sample_dict["background"]
    sig_MC = sample_dict["signal"]
    save_fname = "plots/Fig6_13.pdf"
    print(f"sample_dict: {sample_dict}")
    print(f"binning: {binning}")
    print(f"bkg_MC: {bkg_MC}")
    print(f"sig_MC: {sig_MC}")
    # status = "Private"
    status = "Simulation"
    bdt_edges = OmegaConf.load(f"{load_path}/BDT_edges.yaml")[year]
    # print(f"bdt_edges b4 transform: {bdt_edges}")
    
    bdt_edges = tranformBDT_edges(bdt_edges)
    # print(f"bdt_edges after transform: {bdt_edges}")
    print(f"binning: {binning}")
    print(f"bdt_edges: {bdt_edges}")
    bdt_edges4plot = np.append(bdt_edges, [-0.9, 0.9]) # add the extreme edge values, -0.9 and 0.9
    bdt_edges4plot.sort() 
    print(f"bdt_edges4plot: {bdt_edges4plot}")
    subCatSignificance_hist = getSignificanceHist(sample_dict)
    plotFig_6_13(
        binning, bkg_MC, sig_MC, save_fname,
        title = "", 
        x_title = plot_settings[plot_var].get("xlabel"), 
        lumi = lumi_val,
        status = status,
        bdtCat_boundaries=bdt_edges,
        significance_tuple = (subCatSignificance_hist, bdt_edges4plot)
    )

    
