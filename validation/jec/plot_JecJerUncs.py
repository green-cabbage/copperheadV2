import awkward as ak
import dask_awkward as dak
import argparse
import sys
import os
import numpy as np
import json
from collections import OrderedDict
from modules.utils import filterRegion, applyRegionCatCuts
from src.lib.get_parameters import getParametersForYr
from distributed import Client
import time    
import tqdm
import hist.dask as hda
from hist import Hist
import dask
import glob
import copy
import matplotlib.pyplot as plt

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Add it to sys.path
sys.path.insert(0, parent_dir)
# Now you can import your module
from src.lib.histogram.plotting import fillHist

def getPlotVar(var: str):
    """
    Helper function that removes the variations in variable name if they exist
    """
    if "_nominal" in var:
        plot_var = var.replace("_nominal", "")
    else:
        plot_var = var
    return plot_var

def getPlotSettings(category):
    """
    helper function
    """
    if category == "ggh":
        plot_setting_fname = "../../src/lib/histogram/plot_settings_gghCat_BDT_input.json"
    else: # in no cat case, just use vbfCat plot settings
        plot_setting_fname = "../../src/lib/histogram/plot_settings_vbfCat_MVA_input.json"
    print(f"plot_setting_fname: {plot_setting_fname}")
    with open(plot_setting_fname, "r") as file:
        plot_settings = json.load(file)
    return plot_settings


def getVariationVariable(variable : str, variation : str):
    """
    helper function
    """
    variation_var = variable.replace("nominal", variation)
    return variation_var

def compute_variations(events, hist_empty, sample, categories, regions, variables, variations):
    hist_dictByCat = {}
    for category in categories:
        # add axis for systematic variation
        # sample_hist_dictByVar = {} 
        plot_settings = getPlotSettings(category)
        hist_dictByVar = {}
        for var in variables:
            plot_var = getPlotVar(var)
            if plot_var not in plot_settings.keys():
                print(f"variable {var} not configured in plot settings!")
                continue
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
            # binning = np.append(binning, [8000])
            # print(f"{var} {category} binning: {binning}")
            
            sample_hist_byVar = hist_empty.Var(binning, name=plot_var).Double() 
            
            for region_name in regions:
                for variation in (variations + ["nominal"]):
                # for variation in ["nominal"]:
                    add_vbfFiltered_DY = False
                    events_filtered = applyRegionCatCuts(events, category, region_name, sample, variation, add_vbfFiltered_DY)
                    to_fill_setting = {
                    "region" : region_name,
                    "channel" : category,
                    "variation" : variation,
                    "sample_group": sample,
                    }
                    variation_var = getVariationVariable(var, variation)
                    print(f"{region_name} variation_var: {variation_var}")
                    values = ak.fill_none(events_filtered[variation_var], value=-999.0)
                    weights = events_filtered["wgt_nominal"]

                    # print(f"var: {var}")
                    # print(f"events: {events}")
                    # print(f"variation: {variation}")
                    # print(f"{var} {category} {variation} {region_name} values: {values.compute()}")
                    # print(f"{var} {category} {variation} {region_name} weights: {weights.compute()}")
                    
                    sample_hist_byVar = fillHist(sample_hist_byVar, to_fill_setting, plot_var, values, weights)
                    
            hist_dictByVar[plot_var] = sample_hist_byVar
        hist_dictByCat[category] = hist_dictByVar
    hist_dictByCat = dask.compute(hist_dictByCat)[0]
    

    # print(f"hist_dictByCat.keys(): {hist_dictByCat.keys()}")
    # print(f"hist_dictByCat.values(): {hist_dictByCat.values()}")
    # raise ValueError
    return hist_dictByCat


def plot_variations(computed_hist_dict, sample, categories, regions, variables, variations2validate, save_path="plots"):
    for category in categories:
        plot_settings = getPlotSettings(category)
        for region_name in regions:
            for var in variables:
                plot_var = getPlotVar(var)
                plot_var = getPlotVar(var)
                if plot_var not in plot_settings.keys():
                    print(f"variable {var} not configured in plot settings!")
                    continue
                binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
                computed_hist = computed_hist_dict[category][plot_var]
                
                # Create plot
                plt.figure(figsize=(6, 4))
                
                
                
                
                for variation_base in variations2validate:
                    variations = ["nominal"] + [f"{variation_base}_up", f"{variation_base}_down"]
                    # variations = ["nominal"]
                    print(f"variations: {variations}")
                    # print(f"{var} {category} {region_name} computed_hist: {computed_hist}")
                    
                    for variation in variations:
                        to_project_setting_val = {
                            "region" : region_name,
                            "channel" : category,
                            "variation" : variation,
                            "sample_group": sample,
                            "val_sumw2" : "value"
                        }
                        hist_val = computed_hist[to_project_setting_val].project(plot_var).values()
                        # print(f"{category} {region_name} {variation} {var} hist_val: {hist_val}")
                        # print(f"{category} {region_name} {variation} {var} hist_val: {len(hist_val)}")
                        # hist_val = computed_hist[to_project_setting_val].project(plot_var).values(flow=True)
                        # print(f"{category} {region_name} {variation} {var} hist_val with flow: {hist_val}")
                        # print(f"{category} {region_name} {variation} {var} hist_val with flow: {len(hist_val)}")

                        # Plot step-style histograms
                        midpoints = (binning[1:] + binning[:-1]) / 2
                        plt.step(midpoints, hist_val, where='mid', label=f'{plot_var} {variation}', linewidth=0.2)
                    # Axis labels and title
                    plt.xlabel(plot_var)
                    plt.ylabel('Counts')
                    plt.title('Comparison of 3 Histograms')
                    plt.legend()
                    
                    # Optional: grid
                    plt.grid(True, linestyle='--', alpha=0.6)
                    
                    # Save to PDF
                    plt.savefig(f'{save_path}/{plot_var}_Reg{region_name}Cat{category}Var{variation_base}.pdf')
                    plt.clf()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="string value of year we are calculating",
    )
    parser.add_argument(
    "-var",
    "--variables",
    dest="variables",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of variables to plot (ie: jet, mu, dimuon)",
    )
    parser.add_argument(
    "-load",
    "--load_path",
    dest="load_path",
    default="/depot/cms/users/yun79/results/stage1/test_full/f0_1",
    action="store",
    help="load path",
    )
    parser.add_argument(
    "-label",
    "--label",
    dest="label",
    default="",
    action="store",
    help="label",
    )
    parser.add_argument(
    "-save",
    "--save_path",
    dest="save_path",
    default="./validation/figs/",
    action="store",
    help="save path",
    )
    parser.add_argument(
    "-lumi",
    "--lumi",
    dest="lumi",
    default="",
    action="store",
    help="string value of integrated luminosity to label",
    )
    parser.add_argument(
    "--status",
    dest="status",
    default="",
    action="store",
    help="Status of results ie Private, Preliminary, In Progress",
    )
    parser.add_argument(
    "--ROOT_style",
    dest="ROOT_style",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses pyROOT functionality instead of mplhep",
    )
    parser.add_argument(
    "-reg",
    "--region",
    dest="regions",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="region value to plot, available regions are: h_peak, h_sidebands, z_peak and signal (h_peak OR h_sidebands)",
    )
    parser.add_argument(
    "--use_gateway",
    dest="use_gateway",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses dask gateway client instead of local",
    )
    parser.add_argument(
    "-cat",
    "--categories",
    dest="categories",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="region value to plot, available regions are: h_peak, h_sidebands, z_peak and signal (h_peak OR h_sidebands)",
    )
    args = parser.parse_args()
    # load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{args.label}/{args.category}/stage2_output/*/"
    year = args.year
    load_path = args.load_path
    # events = dak.from_parquet(f"{load_path}/*data.parquet")
    # print(events.fields)
    print(f"load_path : {load_path}")

    config = getParametersForYr("../../configs/parameters/" , year)
    lumi_dict = {
        "2018" : 59.83,
        "2017" : 41.48,
        "2016postVFP": 19.50,
        "2016preVFP": 16.81,
        "2016": 36.3,
        "all" : 137,
    }
    lumi_val = lumi_dict[year]

    if args.use_gateway:
            from dask_gateway import Gateway
            gateway = Gateway(
                "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
                proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
            )
            cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
            client = gateway.connect(cluster_info.name).get_client()
            print("Gateway Client created")
    else:
        client =  Client(n_workers=63,  threads_per_worker=1, processes=True, memory_limit='10 GiB') 
        print("Local scale Client created")

    
    # possible_samples = ["data", "ggh", "vbf", "dy", "ewk", "tt", "st", "ww", "wz", "zz","other"]
    # sample_groups = {
    #     "data": ["data"],
    #     "ggh": ["ggh"],
    #     "vbf": ["vbf"],
    #     "dy": ["dy"],
    #     "top": ["tt", "st"],
    #     "ewk": ["ewk"],
    #     "diboson": ["ww", "wz", "zz"],
    #     "other": ["other"],
    # }
    samples = ["vbf_powheg"]
    
    variables = ["jet1_pt_nominal", "jet2_pt_nominal"]
    # variables = ["jet1_pt", "jet2_pt"]
    n_jer_vars = 6
    # n_jer_vars = 1 #FIXME
    # variations2validate = [f"jer{i}" for i in range(1, n_jer_vars+1)] # we need to keep this separate
    # variations2validate = ["Absolute"] # FIXME
    variations2validate = config["jec_parameters"]["jec_unc_to_consider"]
    print(f"variations2validate: {variations2validate}")
    # add up and down
    variations_with_shifts = [f"{variation}_up" for variation in variations2validate] + [f"{variation}_down" for variation in variations2validate] # TODO: extract the variations from config (use the same method from run_stage1.py)
    # print(f"variations2validate: {variations2validate}")
    # print(f"variations_with_shifts: {variations_with_shifts}")

    # ----------------------------------
    # initialize histograms
    # ----------------------------------
    possible_regions = ["z-peak", "signal", "h-peak", "h-sidebands"] # full list of possible regions to loop over
    possible_channels = ["nocat", "vbf", "ggh"] # full list of possible channels to loop over
    possible_variations = ["nominal"] + variations_with_shifts
    sample_groups = samples
    sample_hist_empty = (
            hda.Hist.new.StrCat(possible_regions, name="region")
            .StrCat(possible_channels, name="channel")
            .StrCat(["value", "sumw2"], name="val_sumw2")
            .StrCat(sample_groups, name="sample_group")
            .StrCat(possible_variations, name="variation")
            # .StrCat(years, name="year")
    )

    # ----------------------------------
    # begin plotting
    # ----------------------------------
    regions = args.regions
    categories = args.categories
    # categories = ["vbf"] #FIXME
    print(f"categories: {categories}")
    
    
    for sample in samples:
        save_path = f"plots/{year}/{sample}"
        os.makedirs(save_path, exist_ok=True)
        full_load_path = load_path+f"/{sample}*/*/*.parquet" 
        print(f"full_load_path: {full_load_path}")
        filelist = glob.glob(full_load_path)
        print(f"filelist len: {len(filelist)}")
        # -----------------------------------------------
        # Load events
        # -----------------------------------------------
        
        events = dak.from_parquet(filelist)
        print(f"events.fields : {events.fields}")
        # raise ValueError
        computed_hist_dict = compute_variations(events, sample_hist_empty, sample, categories, regions, variables, variations_with_shifts)
        plot_variations(computed_hist_dict, sample, categories, regions, variables, variations2validate, save_path=save_path)
        # print(f"computed_hists: {computed_hists}")
        # print(f"events.fields: {events.fields}")

        
        

    print("Success!")
    # for category in args.categories:
    #     sample_hist_dictByVarComputed_byCat = sample_hist_dictByVarComputed[category]
    #     if category == "ggh":
    #         plot_setting_fname = "./src/lib/histogram/plot_settings_gghCat_BDT_input.json"
    #     else: # in no cat case, just use vbfCat plot settings
    #         plot_setting_fname = "./src/lib/histogram/plot_settings_vbfCat_MVA_input.json"
    #     with open(plot_setting_fname, "r") as file:
    #         plot_settings = json.load(file)
        
    #     for region_name in args.regions:
    #         for var in tqdm.tqdm(variables2plot):
    #             if args.linear_scale:
    #                 do_logscale = False
    #             else:
    #                 do_logscale = True  
    #             full_save_path = args.save_path+f"/{args.year}/mplhep/Reg_{region_name}/Cat_{category}/{args.label}"
                # plotComputedHistograms(sample_hist_dictByVarComputed_byCat, var, plot_settings, full_save_path, sample_groups, region_name, category, do_logscale=do_logscale)
                    


            
