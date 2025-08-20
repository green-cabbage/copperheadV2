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

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# Add it to sys.path
sys.path.insert(0, parent_dir)
# Now you can import your module

def plot_6_7(df, binning, var, xlabel):
    
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
    plt.legend(fontsize=9, loc="best", ncol=2)
    # plt.legend(ncol=2)
    # plt.tight_layout()
    # plt.show()
    CenterOfMass = 13
    status = "Simulation"
    hep.cms.label(data=False, loc=0, label=status, com=CenterOfMass, ax=ax_main)
    plt.savefig("test.png")

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
    load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/{args.category}/stage2_output/{year_param}/"
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
    df = dd.read_parquet(full_load_path).compute()
    # change BDT score range from [0,1] to [-1,1]
    df["BDT_score"] = (df["BDT_score"] *2 ) -1
    print(df)
    # raise ValueError


    plot_setting_fname = "../../../src/lib/histogram/plot_settings_vbfCat_MVA_input.json"
    # plot_setting_fname = "plot_settings_vbfCat_MVA_input.json"
    with open(plot_setting_fname, "r") as file:
        plot_settings = json.load(file)
    # plot_var = "BDT_score"
    plot_var = "dimuon_mass"
    if plot_var == "dimuon_mass":
        binning = np.linspace(115, 135, 50)
    else:
        binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
    xlabel =  plot_settings[plot_var].get("xlabel")
    save_fname = f"plots/{args.label}_x_{args.category}/{args.year}_signal/Fig6_7"

    plot_6_7(df, binning, plot_var, xlabel)

    
