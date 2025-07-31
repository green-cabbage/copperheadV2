import awkward as ak
import dask_awkward as dak
import argparse
import sys
import os
import numpy as np
import json
from collections import OrderedDict
from modules.utils import filterRegion
from distributed import Client
import time    
import tqdm
import hist.dask as hda
from hist import Hist
import dask
import glob
import copy

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


def compute_N_plot_variations(events, sample, categories, regions, variables):
    hist_dictByVar = {}

    for category in categories:
        # add axis for systematic variation
        # sample_hist_dictByVar = {} 
        if category == "ggh":
            plot_setting_fname = "../../src/lib/histogram/plot_settings_gghCat_BDT_input.json"
        else: # in no cat case, just use vbfCat plot settings
            plot_setting_fname = "../../src/lib/histogram/plot_settings_vbfCat_MVA_input.json"
        print(f"plot_setting_fname: {plot_setting_fname}")
        with open(plot_setting_fname, "r") as file:
            plot_settings = json.load(file)
        for region_name in regions:
            # events = copy.deepcopy(events)
            events = filterRegion(events, region=region_name)
            for var in variables:
                plot_var = getPlotVar(var)
                hist_dictBySample = {}
                
                # for process in available_processes:
                if plot_var not in plot_settings.keys():
                    print(f"variable {var} not configured in plot settings!")
                    continue
                binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
                # print(f"var: {var}")
                # print(f"events: {events}")
                sample_hist_byVar = sample_hist.Var(binning, name=var).Double()
                to_fill_setting = {
                "region" : region_name,
                "channel" : category,
                "variation" : "nominal",
                "sample_group": sample,
                }
                values = ak.fill_none(events[var], value=-999.0)
                weights = events["wgt_nominal"]
                
                sample_hist_byVar = fillHist(sample_hist_byVar, to_fill_setting, var, values, weights)
                hist_dictByVar[var] = sample_hist_byVar

    hist_dictByVar = dask.compute(hist_dictByVar)[0]
    return hist_dictByVar
                
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
    if year == "all":
        year_param = "*"
    elif year == "2016":
        year_param = "2016*"
    else:
        year_param = year
    load_path = args.load_path
    # events = dak.from_parquet(f"{load_path}/*data.parquet")
    # print(events.fields)
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
    variations2validate = [f"jer{i}" for i in range(1, n_jer_vars+1)]
    # add up and down
    variations_with_shifts = [f"{variation}_up" for variation in variations2validate] + [f"{variation}_down" for variation in variations2validate] # TODO: extract the variations from config (use the same method from run_stage1.py)
    # print(f"variations2validate: {variations2validate}")
    # print(f"variations_with_shifts: {variations_with_shifts}")

    # ----------------------------------
    # initialize histograms
    # ----------------------------------
    regions = ["z-peak", "signal", "h-peak", "h-sidebands"] # full list of possible regions to loop over
    channels = ["nocat", "vbf", "ggh"] # full list of possible channels to loop over
    variations = ["nominal"] + variations_with_shifts
    sample_groups = samples
    sample_hist = (
            hda.Hist.new.StrCat(regions, name="region")
            .StrCat(channels, name="channel")
            .StrCat(["value", "sumw2"], name="val_sumw2")
            .StrCat(sample_groups, name="sample_group")
            .StrCat(variations, name="variation")
            # .StrCat(years, name="year")
    )

    # ----------------------------------
    # begin plotting
    # ----------------------------------
    for sample in samples:
        full_load_path = load_path+f"/{sample}*/*/*.parquet" 
        print(f"full_load_path: {full_load_path}")
        filelist = glob.glob(full_load_path)
        print(f"filelist len: {len(filelist)}")
        # -----------------------------------------------
        # Load events
        # -----------------------------------------------
        
        events = dak.from_parquet(filelist)

        computed_hists = compute_N_plot_variations(events, sample, args.categories, regions, variables)
        # print(f"computed_hists: {computed_hists}")
        # print(f"events.fields: {events.fields}")
        
    
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
                    

    raise ValueError

            
