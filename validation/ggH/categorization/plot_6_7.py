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
import ROOT
import ROOT as rt
import copy
from array import array
ROOT.gStyle.SetOptStat(0) # remove stats box
import dask.dataframe as dd
import matplotlib.cm as cm
from modules.utils import filterRegion, pair_and_remove, plotScatter, plot2D
import pandas as pd

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# Add it to sys.path
sys.path.insert(0, parent_dir)
# Now you can import your module

def plotWgtAnnhilation(parq_path, save_dir):
    """
    wrapper function
    """
    df = dd.read_parquet(parq_path).compute()
    df = filterRegion(df, region="h-peak")

    # print(f"{parq_path} \n df: {df}")
    # print(f"{parq_path} \n df: {len(df)}")
    colsOfInterest = ["mu1_eta", "mu2_eta","dimuon_pt"]
    matches, remaining = pair_and_remove(df, cols=colsOfInterest)
    # print(f"matches: {matches}")
    # print(f"remaining: {remaining}")
    print(f"df: {len(df)}")
    print(f"matches: {len(matches)}")
    print(f"remaining: {len(remaining)}")
    # print(f"remaining all positive: {np.all(remaining['wgt_nominal'] >= 0)}")

    for var in variables:
        plot_var = getPlotVar(var)
        if plot_var == "dimuon_mass":
            binning = np.linspace(70, 110, 50)
        elif plot_var == "jj_mass":
            binning = np.linspace(0, 2500, 100)
        else:
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
        xlabel =  plot_settings[plot_var].get("xlabel")
        df_dict = {
            "DY" : df,
            "DY pair match removed" : remaining,
        }
        compareMC(df_dict, binning, var, xlabel, save_dir, applyWgt=True, do_logscale=False)




def getPlotVar(var: str):
    """
    Helper function that removes the variations in variable name if they exist
    """
    if "_nominal" in var:
        plot_var = var.replace("_nominal", "")
    else:
        plot_var = var
    return plot_var

def get_unique_colors(n, cmap_name="tab10"):

    """
    Return a list of n unique colors (as RGBA tuples) compatible with pyplot.
    
    Parameters
    ----------
    n : int
        Number of colors to generate.
    cmap_name : str, optional
        Name of a matplotlib colormap (default "tab10").
    
    Returns
    -------
    list of RGBA tuples
    """
    cmap = cm.get_cmap(cmap_name, n)  # sample 'n' distinct colors
    return [cmap(i) for i in range(n)]
    
def weighted_quantile(values, quantile, sample_weight=None):
    """
    Compute weighted quantile of given data.
    
    values : np.ndarray
        Data (e.g. BDT scores).
    quantile : float
        Quantile in [0, 1] (e.g. 0.3 for 30%).
    sample_weight : np.ndarray or None
        Weights (same length as values).
    """
    values = np.array(values)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    else:
        sample_weight = np.array(sample_weight)
    
    # sort by values
    sorter = np.argsort(values)
    values = values[sorter]
    weights = sample_weight[sorter]

    # compute cumulative normalized weights
    cumsum = np.cumsum(weights)
    cutoff = quantile * np.sum(weights)

    return values[np.searchsorted(cumsum, cutoff)]

# def plot_6_7BySubCat(df, binning, var, xlabel, save_dir):
    
#     # --- binning ---
#     bdt_edges = np.array([-1.00, -0.28, -0.10, 0.08, 0.23, 0.32, 0.43, 0.51, 1.00])
    
#     # --- plotting ---
#     # plt.figure(figsize=(7,5))
#     fig, ax_main = plt.subplots()
    
#     colors = ["black","red","blue","orange","green","cyan","magenta","gray"]
#     score_name =  "BDT_score"
#     for (lo, hi), color in zip(zip(bdt_edges[:-1], bdt_edges[1:]), colors):
#         mask = (df[score_name] > lo) & (df[score_name] <= hi)
#         if not mask.any():
#             continue
#         wgt_var= "wgt_nominal"
#         hist, bins = np.histogram(df.loc[mask, var], bins=binning, weights=df.loc[mask, wgt_var])
#         # hist, bins = np.histogram(df.loc[mask, var], bins=binning, density=True)
#         hist = hist / np.sum(hist)
#         hep.histplot(
#             hist,
#             bins,
#             label=f"{lo:.2f} < BDT < {hi:.2f}",
#             histtype="step",
#             color=color,
#             ax=ax_main,
#         )

#     plt.xlabel(xlabel)
#     plt.ylabel("A.U.")
#     plt.title("")
#     # plt.title("Normalized dimuon mass by BDT slice")
#     plt.legend(fontsize=12, loc="best", ncol=1)
#     # plt.legend(ncol=2)
#     # plt.tight_layout()
#     # plt.show()
#     CenterOfMass = 13
#     # status = "Simulation"
#     # hep.cms.label(data=False, loc=0, label=status, com=CenterOfMass, ax=ax_main)
#     hep.cms.label(data=False, loc=0, com=CenterOfMass, ax=ax_main)
#     fig_name = f"{save_dir}/{plot_var}BySubCat.pdf"
#     # fig_name = f"{plot_var}.pdf"
#     plt.savefig(fig_name)


def plot_6_7FineGrain(df, binning, var, xlabel, save_dir):
    
    # --- binning ---
    # bdt_edges = np.array([-1.00, -0.28, -0.10, 0.08, 0.23, 0.32, 0.43, 0.51, 1.00])
    n_cats = 20
    # n_cats = 12
    bdt_edges = np.linspace(-1,1,n_cats+1)
    # bdt_edges = np.array([-1.   , -0.625, -0.5  , -0.375, -0.25 , -0.125,        0.   ,  0.125,  0.25 ,  0.375,  0.5  ,  0.625,  0.75 ,  0.875,        1.   ])
    # n_cats = len(bdt_edges)-1
    
    # --- plotting ---
    # plt.figure(figsize=(7,5))
    fig, ax_main = plt.subplots()
    
    colors = get_unique_colors(n_cats, cmap_name="tab20")
    score_name =  "BDT_score"
    for (lo, hi), color in zip(zip(bdt_edges[:-1], bdt_edges[1:]), colors):
        mask = (df[score_name] > lo) & (df[score_name] <= hi)
        if not mask.any():
            continue
        wgt_var= "wgt_nominal"
        hist, bins = np.histogram(df.loc[mask, var], bins=binning, weights=df.loc[mask, wgt_var])
        # hist, bins = np.histogram(df.loc[mask, var], bins=binning, density=True)
        hist = hist / np.sum(hist)
        hep.histplot(
            hist,
            bins,
            label=f"{lo:.2f} < BDT < {hi:.2f}",
            histtype="step",
            color=color,
            ax=ax_main,
        )

    plt.xlabel(xlabel)
    plt.ylabel("A.U.")
    plt.title("")
    # plt.title("Normalized dimuon mass by BDT slice")
    plt.legend(fontsize=12, loc="best", ncol=1)
    # plt.legend(ncol=2)
    # plt.tight_layout()
    # plt.show()
    CenterOfMass = 13
    # status = "Simulation"
    # hep.cms.label(data=False, loc=0, label=status, com=CenterOfMass, ax=ax_main)
    hep.cms.label(data=False, loc=0, com=CenterOfMass, ax=ax_main)
    fig_name = f"{save_dir}/{plot_var}FineGrain.pdf"
    # fig_name = f"{plot_var}.pdf"
    plt.savefig(fig_name)

def plot_6_7BDTCatMerged(df, binning, var, xlabel, save_dir):
    
    # --- binning ---
    bdt_edges = np.array([-1.00, -0.28, -0.10, 0.08, 0.23, 0.32, 0.43, 0.51, 1.00])
    
    # --- plotting ---
    # plt.figure(figsize=(7,5))
    fig, ax_main = plt.subplots()
    
    colors = ["black","red","blue","orange","green","cyan","magenta","gray"]
    score_name =  "BDT_score"
    hist_l = []
    for (lo, hi), color in zip(zip(bdt_edges[:-1], bdt_edges[1:]), colors):
        mask = (df[score_name] > lo) & (df[score_name] <= hi)
        if not mask.any():
            continue
        wgt_var= "wgt_nominal"
        hist, bins = np.histogram(df.loc[mask, var], bins=binning, weights=df.loc[mask, wgt_var])
        hist_l.append(hist)
    # print(f"hist_l: {hist_l}")
    hist = sum(hist_l)
    # print(f"hist: {hist}")
    hist = hist/np.sum(hist)
    hep.histplot(
        hist,
        bins,
        label=f"combined BDT category",
        histtype="step",
        color=color,
        ax=ax_main,
    )

    plt.xlabel(xlabel)
    plt.ylabel("A.U.")
    plt.title("")
    # plt.title("Normalized dimuon mass by BDT slice")
    plt.legend(fontsize=12, loc="best", ncol=1)
    # plt.legend(ncol=2)
    # plt.tight_layout()
    # plt.show()
    CenterOfMass = 13
    # status = "Simulation"
    # hep.cms.label(data=False, loc=0, label=status, com=CenterOfMass, ax=ax_main)
    hep.cms.label(data=False, loc=0, com=CenterOfMass, ax=ax_main)
    fig_name = f"{save_dir}/{plot_var}_BDTCatMerged.pdf"
    # fig_name = f"{plot_var}.pdf"
    plt.savefig(fig_name)


def plot_6_7(df, binning, var, xlabel, save_dir):
    
    # --- binning ---
    bdt_edges = np.array([-1.00, -0.28, -0.10, 0.08, 0.23, 0.32, 0.43, 0.51, 1.00])
    
    # --- plotting ---
    # plt.figure(figsize=(7,5))
    fig, ax_main = plt.subplots()
    
    colors = ["black","red","blue","orange","green","cyan","magenta","gray"]
    score_name =  "BDT_score"
    for (lo, hi), color in zip(zip(bdt_edges[:-1], bdt_edges[1:]), colors):
        mask = (df[score_name] > lo) & (df[score_name] <= hi)
        if not mask.any():
            continue
        wgt_var= "wgt_nominal"
        hist, bins = np.histogram(df.loc[mask, var], bins=binning, weights=df.loc[mask, wgt_var])
        # hist, bins = np.histogram(df.loc[mask, var], bins=binning, density=True)
        hist = hist / np.sum(hist)
        hep.histplot(
            hist,
            bins,
            label=f"{lo:.2f} < BDT < {hi:.2f}",
            histtype="step",
            color=color,
            ax=ax_main,
        )

    plt.xlabel(xlabel)
    plt.ylabel("A.U.")
    plt.title("")
    # plt.title("Normalized dimuon mass by BDT slice")
    plt.legend(fontsize=12, loc="best", ncol=1)
    # plt.legend(ncol=2)
    # plt.tight_layout()
    # plt.show()
    CenterOfMass = 13
    # status = "Simulation"
    # hep.cms.label(data=False, loc=0, label=status, com=CenterOfMass, ax=ax_main)
    hep.cms.label(data=False, loc=0, com=CenterOfMass, ax=ax_main)
    fig_name = f"{save_dir}/{plot_var}.pdf"
    # fig_name = f"{plot_var}.pdf"
    plt.savefig(fig_name)


def plot_6_7BySubCat(df, binning, var, xlabel, save_dir):
    
    # --- binning ---
    bdt_edges = np.array([ # 2018 UL subcat edges
        0.0,
        0.39327239990234375,
        0.5264375805854797,
        0.736026406288147,
        0.8443986773490906,
        1.1
    ])
    bdt_edges = bdt_edges*2 -1

    # --- plotting ---
    # plt.figure(figsize=(7,5))
    fig, ax_main = plt.subplots()
    
    colors = ["black","red","blue","orange","green","cyan","magenta","gray"]
    score_name =  "BDT_score"
    for (lo, hi), color in zip(zip(bdt_edges[:-1], bdt_edges[1:]), colors):
        mask = (df[score_name] > lo) & (df[score_name] <= hi)
        if not mask.any():
            continue
        wgt_var= "wgt_nominal"
        hist, bins = np.histogram(df.loc[mask, var], bins=binning, weights=df.loc[mask, wgt_var])
        # hist, bins = np.histogram(df.loc[mask, var], bins=binning, density=True)
        hist = hist / np.sum(hist)
        hep.histplot(
            hist,
            bins,
            label=f"{lo:.2f} < BDT < {hi:.2f}",
            histtype="step",
            color=color,
            ax=ax_main,
        )

    plt.xlabel(xlabel)
    plt.ylabel("A.U.")
    plt.title("")
    # plt.title("Normalized dimuon mass by BDT slice")
    plt.legend(fontsize=12, loc="best", ncol=1)
    # plt.legend(ncol=2)
    # plt.tight_layout()
    # plt.show()
    CenterOfMass = 13
    # status = "Simulation"
    # hep.cms.label(data=False, loc=0, label=status, com=CenterOfMass, ax=ax_main)
    hep.cms.label(data=False, loc=0, com=CenterOfMass, ax=ax_main)
    fig_name = f"{save_dir}/{plot_var}BySubCat.pdf"
    # fig_name = f"{plot_var}.pdf"
    plt.savefig(fig_name)

def compareMC(df_dict, binning, var, xlabel, save_dir, unweighted=False, abs_wgt=False, removeNegWgt=False, applyWgt=False, do_logscale=True):
    
    # --- plotting ---
    # plt.figure(figsize=(7,5))
    plt.clf()
    fig, ax_main = plt.subplots()
    
    wgt_var= "wgt_nominal"

    for label, df in df_dict.items():
        if abs_wgt:
            wgt = abs(df[wgt_var])
            var_val = df[var]
        elif removeNegWgt:
            is_pos = df[wgt_var] >=0 
            wgt = df[wgt_var][is_pos]
            var_val = df[var][is_pos]
        else:
            wgt = df[wgt_var]
            var_val = df[var]
        hist, bins = np.histogram(var_val, bins=binning, weights=wgt)
        if not applyWgt: # every other options lead to normalizing the histogram
            hist = hist / np.sum(hist)
        hep.histplot(
            hist,
            bins,
            label=label,
            histtype="step",
            ax=ax_main,
        )
    plt.xlabel(xlabel)
    if applyWgt:
        plt.ylabel("yield")
        if do_logscale:
            plt.yscale("log")
    else:
        plt.ylabel("A.U.")
    plt.title("")
    plt.legend(fontsize=12, loc="best", ncol=1)
    # plt.legend(ncol=2)
    # plt.tight_layout()
    # plt.show()
    CenterOfMass = 13
    # status = "Simulation"
    # hep.cms.label(data=False, loc=0, label=status, com=CenterOfMass, ax=ax_main)
    hep.cms.label(data=False, loc=0, com=CenterOfMass, ax=ax_main)
    plot_var = getPlotVar(var)
    if unweighted:
        fig_name = f"{save_dir}/{plot_var}_sigMC_compUnWgted.pdf"
    elif abs_wgt:
        fig_name = f"{save_dir}/{plot_var}_sigMC_compAbsWgt.pdf"
    elif removeNegWgt:
        fig_name = f"{save_dir}/{plot_var}_sigMC_compPosWgt.pdf"
    elif applyWgt:
        fig_name = f"{save_dir}/{plot_var}_sigMC_compXsecNormalized.pdf"
    else: 
        fig_name = f"{save_dir}/{plot_var}_sigMC_comp.pdf"
    plt.savefig(fig_name)

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
        year_param = "*"
    elif year == "2016":
        year_param = "2016*"
    else:
        year_param = year
    # load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/{args.category}/stage2_output/{year_param}/"
    load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/{args.category}/stage2_outputForFig6_7/{year_param}/"
    # events = dak.from_parquet(f"{load_path}/*data.parquet")
    # print(events.fields)
    bdt_edges = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0]
    print(f"load_path : {load_path}")
    lumi_dict = {
        "2018" : 59.83,
        "2017" : 41.48,
        "2016postVFP": 19.50,
        "2016preVFP": 16.81,
        "2016": 36.3,
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
    # for group, group_fname in sample_groups.items():
    #     full_load_path = load_path+f"*{group_fname}.parquet" 
    #     events = dak.from_parquet(full_load_path)
    #     events = filterRegion(events, region=args.region)
    #     sample_dict = fillSampleValues(events, sample_dict, group)

    full_load_path = load_path+f"processed_events_sigMC*.parquet" 
    # full_load_path = load_path+f"processed_events_sigMC_ggh.parquet" 
    df = dd.read_parquet(full_load_path).compute()
    df = filterRegion(df, region="h-peak")
    # change BDT score range from [0,1] to [-1,1]
    # print(df)
    print(df.columns)
    df["BDT_score"] = (df["BDT_score"] *2 ) -1
    print(df.columns)
    print(df.isna().any().any()) 
    print(df.isna().sum().sum())
    # raise ValueError


    # plot_setting_fname = "../../../src/lib/histogram/plot_settings_vbfCat_MVA_input.json"
    plot_setting_fname = "../../../src/lib/histogram/plot_settings_gghCat_BDT_input.json"
    # plot_setting_fname = "plot_settings_vbfCat_MVA_input.json"
    with open(plot_setting_fname, "r") as file:
        plot_settings = json.load(file)

    save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/Fig6_7"
    # save_dir = f"plots/{args.category}/{args.year}signal/Fig6_7"
    os.makedirs(save_dir, exist_ok=True)
    
    bdt_inputs = [
        'dimuon_cos_theta_cs', 
        'dimuon_phi_cs', 
        'dimuon_rapidity', 
        'dimuon_pt', 
        'jet1_eta_nominal', 
        # 'jet2_eta_nominal', 
        'jet1_pt_nominal', 
        'jet2_pt_nominal', 
        'jj_dEta_nominal', 
        'jj_dPhi_nominal', 
        'jj_mass_nominal', 
        # 'mmj1_dEta', 
        # 'mmj1_dPhi',  
        'mmj_min_dEta_nominal', 
        'mmj_min_dPhi_nominal', 
        'mu1_eta', 
        'mu1_pt_over_mass', 
        'mu2_eta', 
        'mu2_pt_over_mass', 
        'zeppenfeld_nominal',
        'njets_nominal'
    ]
    variables = bdt_inputs + ["dimuon_mass", "dimuon_ebe_mass_res", "jet2_eta_nominal", "rpt_nominal"]
    # variables =  ["dimuon_mass", "dimuon_ebe_mass_res"
    threshold_targets = [0.3, .65, .8, .95]
    for var in variables:
        plot_var = getPlotVar(var)
        if plot_var == "dimuon_mass":
            binning = np.linspace(115, 135, 50)
        else:
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
        xlabel =  plot_settings[plot_var].get("xlabel")
        thresholds = []
        for threshold_target in threshold_targets:
            threshold = weighted_quantile(df["BDT_score"], threshold_target, sample_weight=df["wgt_nominal"])
            thresholds.append(threshold)
        thresholds = np.array(thresholds)
        print("BDT score threshold (30% cumulative weight):", thresholds)
        plot_6_7(df, binning, var, xlabel, save_dir)
        plot_6_7BySubCat(df, binning, var, xlabel, save_dir)
        plot_6_7FineGrain(df, binning, var, xlabel, save_dir)
        plot_6_7BDTCatMerged(df, binning, var, xlabel, save_dir)

    save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/Scatter"
    os.makedirs(save_dir, exist_ok=True)

    x_var = "dimuon_pt"
    plotScatter(df, variables, x_var, save_dir)
    # ----------------
    save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/Hist2D"
    os.makedirs(save_dir, exist_ok=True)
    x_var = "dimuon_pt"
    plot2D(df, variables, x_var, plot_settings, save_dir)
    
    
    # raise ValueError


    # # # ----------------------------------------------------
    # # #  Add sigMC comparison
    # # # ----------------------------------------------------
    # # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/sigMC_comp"
    # # os.makedirs(save_dir, exist_ok=True)
    
    # # full_load_path = load_path+f"processed_events_sigMC_ggh.parquet" 
    # # ggh_df = dd.read_parquet(full_load_path).compute()
    # # full_load_path = load_path+f"processed_events_sigMC_vbf.parquet" 
    # # vbf_df = dd.read_parquet(full_load_path).compute()
    
    # # # for var in ["dimuon_pt", "dimuon_mass"]:
    # # for var in variables:
    # #     plot_var = getPlotVar(var)
    # #     if plot_var == "dimuon_mass":
    # #         binning = np.linspace(115, 135, 50)
    # #     else:
    # #         binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    # #     xlabel =  plot_settings[plot_var].get("xlabel")
    # #     df_dict = {
    # #         "ggH" : ggh_df,
    # #         "VBF" : vbf_df
    # #     }
    # #     compareMC(df_dict, binning, var, xlabel, save_dir)


    # # # ----------------------------------------------------
    # # #  Add sigMC vs bkg comparison
    # # # ----------------------------------------------------
    # # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/sigBkgMC_comp"
    # # os.makedirs(save_dir, exist_ok=True)
    
    # # full_load_path = load_path+f"processed_events_sigMC*.parquet" 
    # # sig_df = dd.read_parquet(full_load_path).compute()
    # # sig_df = filterRegion(sig_df, region="h-peak")
    # # full_load_path = load_path+f"processed_events_bkgMC_dy.parquet" 
    # # dy_df = dd.read_parquet(full_load_path).compute()
    # # dy_df = filterRegion(dy_df, region="h-peak")
    # # print(dy_df.columns)
    # # # raise ValueError
    # # # for var in ["dimuon_pt", "dimuon_mass"]:
    # # for var in variables:
    # #     plot_var = getPlotVar(var)
    # #     if plot_var == "dimuon_mass":
    # #         binning = np.linspace(115, 135, 50)
    # #     else:
    # #         binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    # #     xlabel =  plot_settings[plot_var].get("xlabel")
    # #     df_dict = {
    # #         "ggH+VBF" : sig_df,
    # #         "DY" : dy_df
    # #     }
    # #     compareMC(df_dict, binning, var, xlabel, save_dir)


    # # # ----------------------------------------------------
    # # #  Add ggH vs VBF vs bkg comparison
    # # # ----------------------------------------------------
    # # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/ggHVBF_DYMC_comp"
    # # os.makedirs(save_dir, exist_ok=True)
    
    # # full_load_path = load_path+f"processed_events_sigMC_ggh.parquet" 
    # # ggh_df = dd.read_parquet(full_load_path).compute()
    # # ggh_df = filterRegion(ggh_df, region="h-peak")
    # # full_load_path = load_path+f"processed_events_sigMC_vbf.parquet" 
    # # vbf_df = dd.read_parquet(full_load_path).compute()
    # # vbf_df = filterRegion(vbf_df, region="h-peak")
    # # full_load_path = load_path+f"processed_events_bkgMC_dy.parquet" 
    # # dy_df = dd.read_parquet(full_load_path).compute()
    # # dy_df = filterRegion(dy_df, region="h-peak")
    # # print(dy_df.columns)
    # # # raise ValueError
    # # # for var in ["dimuon_pt", "dimuon_mass"]:
    # # for var in variables:
    # #     plot_var = getPlotVar(var)
    # #     if plot_var == "dimuon_mass":
    # #         binning = np.linspace(115, 135, 50)
    # #     else:
    # #         binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    # #     xlabel =  plot_settings[plot_var].get("xlabel")
    # #     df_dict = {
    # #         "ggH" : ggh_df,
    # #         "VBF" : vbf_df,
    # #         "DY" : dy_df
    # #     }
    # #     compareMC(df_dict, binning, var, xlabel, save_dir)
    # #     compareMC(df_dict, binning, var, xlabel, save_dir, unweighted=True)
    # #     compareMC(df_dict, binning, var, xlabel, save_dir, abs_wgt=True)
    # #     compareMC(df_dict, binning, var, xlabel, save_dir, removeNegWgt=True)
    # #     compareMC(df_dict, binning, var, xlabel, save_dir, applyWgt=True)


    # # # ----------------------------------------------------
    # # #  check DY in z peak region with no ggH cat cuts
    # # # ----------------------------------------------------
    # # # extract directly from stage1
    # # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/DYMC_comp"
    # # os.makedirs(save_dir, exist_ok=True)

    # # if year == "all":
    # #     year_param = "*"
    # # else:
    # #     year_param = year
    # # stage1_load_path=f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/stage1_output/{year_param}/f1_0/"
    # # # stage1_load_path=f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/stage1_output/2018/f1_0/"

    # # full_load_path = stage1_load_path+f"dy*/*/*.parquet" 
    # # print(full_load_path)
    # # fields2load = variables + ["wgt_nominal"]
    # # dy_df = dd.read_parquet(full_load_path)[fields2load].compute()
    # # dy_df_zpeak = filterRegion(dy_df, region="z-only")
    # # dy_df_hpeak = filterRegion(dy_df, region="h-peak")
    
    
    # # print(dy_df.columns)
    # # for var in variables:
    # #     plot_var = getPlotVar(var)
    # #     if plot_var == "dimuon_mass":
    # #         binning = np.linspace(70, 110, 50)
    # #     else:
    # #         binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    # #     xlabel =  plot_settings[plot_var].get("xlabel")
    # #     df_dict = {
    # #         "DY ($85 < m_{\mu\mu} < 95$)" : dy_df_zpeak,
    # #         "DY H peak" : dy_df_hpeak,
    # #     }
    # #     compareMC(df_dict, binning, var, xlabel, save_dir)

    # # # ----------------------------------------------------
    # # #  add DY with ggH channel cut
    # # # ----------------------------------------------------
    
    # # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/DYMCggHCut_comp"
    # # os.makedirs(save_dir, exist_ok=True)
    # # full_load_path = load_path+f"processed_events_bkgMC_dy.parquet" 
    # # dy_df_gghCut = dd.read_parquet(full_load_path).compute()
    # # dy_df_gghCut = filterRegion(dy_df_gghCut, region="h-peak")
    
    # # for var in variables:
    # #     plot_var = getPlotVar(var)
    # #     if plot_var == "dimuon_mass":
    # #         binning = np.linspace(70, 110, 50)
    # #     elif plot_var == "jj_mass":
    # #         binning = np.linspace(0, 2500, 100)
    # #     else:
    # #         binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    # #     xlabel =  plot_settings[plot_var].get("xlabel")
    # #     df_dict = {
    # #         "DY ($85 < m_{\mu\mu} < 95$)" : dy_df_zpeak,
    # #         "DY H peak" : dy_df_hpeak,
    # #         "DY H peak + ggH channel cut" : dy_df_gghCut,
    # #     }
    # #     compareMC(df_dict, binning, var, xlabel, save_dir)
    # #     compareMC(df_dict, binning, var, xlabel, save_dir, applyWgt=True)

    # # ----------------------------------------------------
    # #  compare DY sample with sample wgt annhilation
    # # ----------------------------------------------------
    
    # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/DYMCNegWgtPair_comp"
    # os.makedirs(save_dir, exist_ok=True)
    # full_load_path = load_path+f"processed_events_bkgMC_dy.parquet" 
    # plotWgtAnnhilation(full_load_path, save_dir)
    
    # # dy_df = dd.read_parquet(full_load_path).compute()
    # # dy_df = filterRegion(dy_df, region="h-peak")

    # # print(f"dy_df: {dy_df}")
    # # print(f"dy_df: {len(dy_df)}")
    # # colsOfInterest = ["mu1_eta", "mu2_eta","dimuon_pt"]
    # # matches, remaining = pair_and_remove(dy_df, cols=colsOfInterest)
    # # print(f"matches: {matches}")
    # # print(f"remaining: {remaining}")
    # # print(f"matches: {len(matches)}")
    # # print(f"remaining: {len(remaining)}")

    # # for var in variables:
    # #     plot_var = getPlotVar(var)
    # #     if plot_var == "dimuon_mass":
    # #         binning = np.linspace(70, 110, 50)
    # #     elif plot_var == "jj_mass":
    # #         binning = np.linspace(0, 2500, 100)
    # #     else:
    # #         binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    # #     xlabel =  plot_settings[plot_var].get("xlabel")
    # #     df_dict = {
    # #         "DY" : dy_df,
    # #         "DY pair match removed" : remaining,
    # #     }
    # #     compareMC(df_dict, binning, var, xlabel, save_dir, applyWgt=True, do_logscale=False)


    # # ----------------------------------------------------
    # #  compare top sample with sample wgt annhilation
    # # ----------------------------------------------------
    
    # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/TTMCNegWgtPair_comp"
    # os.makedirs(save_dir, exist_ok=True)
    # full_load_path = load_path+f"processed_events_bkgMC_tt.parquet" 
    # plotWgtAnnhilation(full_load_path, save_dir)
    # save_dir = f"plots/{args.label}_x_{args.category}/{args.year}_signal/STMCNegWgtPair_comp"
    # os.makedirs(save_dir, exist_ok=True)
    # full_load_path = load_path+f"processed_events_bkgMC_st.parquet" 
    # plotWgtAnnhilation(full_load_path, save_dir)
    
    
    