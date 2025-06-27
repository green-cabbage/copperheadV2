import awkward as ak
import dask_awkward as dak
import numpy as np
import json
import argparse
import os
from src.lib.histogram.ROOT_utils import setTDRStyle, CMS_lumi, reweightROOTH_data, reweightROOTH_mc #reweightROOTH
from src.lib.histogram.plotting import plotDataMC_compare
from distributed import Client
import time    
import tqdm
import cmsstyle as CMS
from collections import OrderedDict
import glob
import copy
import hist.dask as hda
from hist import Hist
import dask

def get_scalar_ptCentrality(events):
    pt_centrality_scalar = events.dimuon_pt - abs(events.jet1_pt_nominal + events.jet2_pt_nominal)/2
    pt_centrality_scalar = pt_centrality_scalar / abs(events.jet1_pt_nominal - events.jet2_pt_nominal)
    return pt_centrality_scalar
    

# real process arrangement
group_data_processes = ["data_A", "data_B", "data_C", "data_D", "data_E",  "data_F", "data_G", "data_H"]
# group_DY_processes = ["dy_M-100To200", "dy_M-50"] # dy_M-50 is not used in ggH BDT training input
# group_DY_processes = ["dy_M-100To200"]
# group_DY_processes = ["dy_M-100To200","dy_VBF_filter"]
group_DY_processes = [
    "dy_M-50",
    "dy_M-100To200",
    "dy_M-50_aMCatNLO",
    "dy_M-100To200_aMCatNLO",
    "dy_m105_160_amc",
    "dy_m105_160_vbf_amc",
    "dy_VBF_filter_customJMEoff",
    "dy_VBF_filter_fromGridpack",
    "dy_VBF_filter_NewZWgt",
    # "dyTo2L_M-50_0j",
    # "dyTo2L_M-50_1j",
    # "dyTo2L_M-50_2j",
    # "dyTo2L_M-50_incl",
    "dy_M-100To200_MiNNLO",
    "dy_M-50_MiNNLO"
]


# group_DY_processes = ["dy_M-100To200","dy_VBF_filter_customJMEoff"]
# group_DY_processes = [] # just VBf filter

group_Top_processes = ["ttjets_dl", "ttjets_sl", "st_tw_top", "st_tw_antitop", "tt_inclusive", "st_t_top", "st_t_antitop"]
group_Ewk_processes = ["ewk_lljj_mll50_mjj120"]
group_VV_processes = ["ww_2l2nu", "wz_3lnu", "wz_2l2q", "wz_1l1nu2q", "zz"]# diboson
# group_ggH_processes = ["ggh_amcPS"]
group_ggH_processes = ["ggh_powhegPS"]
group_VBF_processes = ["vbf_powheg_dipole"]

group_dict = {
    "data": group_data_processes,
    "DY": group_DY_processes,
    "Top": group_Top_processes,
    "Ewk": group_Ewk_processes,
    "VV": group_VV_processes,
    "ggH": group_ggH_processes,
    "VBF": group_VBF_processes
}

def find_group_name(process_name, group_dict):
    for group_name, processes in group_dict.items():
        if process_name in processes:
            return group_name
    return "other"


def fillHist(sample_hist, to_fill_setting, values, weights):
    values_filter = values!=-999.0
    values = values[values_filter]
    weights = weights[values_filter]
    to_fill_setting[var] = values
    to_fill_value = to_fill_setting.copy()
    to_fill_value["val_sumw2"] = "value"
    sample_hist.fill(**to_fill_value, weight=weights)
    
    to_fill_sumw2 = to_fill_setting.copy()
    to_fill_sumw2["val_sumw2"] = "sumw2"
    sample_hist.fill(**to_fill_sumw2, weight=weights * weights)
    return sample_hist
                    

def getPlotVar(var: str):
    """
    Helper function that removes the variations in variable name if they exist
    """
    if "_nominal" in var:
        plot_var = var.replace("_nominal", "")
    else:
        plot_var = var
    return plot_var


def applyRegionCatCuts(events, category: str, region_name: str):
    # do mass region cut
    mass = events.dimuon_mass
    z_peak = ((mass > 70) & (mass < 110))
    h_sidebands =  ((mass > 110) & (mass < 115.03)) | ((mass > 135.03) & (mass < 150))
    h_peak = ((mass > 115.03) & (mass < 135.03))
    if region_name == "signal":
        region = h_sidebands | h_peak
    elif region_name == "h-peak":
        region = h_peak 
    elif region_name == "h-sidebands":
        region = h_sidebands 
    elif region_name == "z-peak":
        region = z_peak 
    else: 
        print("ERROR: acceptable region!")
        raise ValueError
    
    # do category cut
    if category == "nocat": 
        # print("nocat mode!")
        prod_cat_cut =  ak.ones_like(region, dtype="bool")
    else: # VBF or ggH
        btagLoose_filter = ak.fill_none((events.nBtagLoose_nominal >= 2), value=False)
        btagMedium_filter = ak.fill_none((events.nBtagMedium_nominal >= 1), value=False) & ak.fill_none((events.njets_nominal >= 2), value=False)
        btag_cut = btagLoose_filter | btagMedium_filter
        # vbf_cut = ak.fill_none(events.vbf_cut, value=False) # in the future none values will be replaced with False
        vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
        vbf_cut = ak.fill_none(vbf_cut, value=False)
        if category == "vbf":
            # print("vbf mode!")
            prod_cat_cut =  vbf_cut
            prod_cat_cut = prod_cat_cut & ~btag_cut # btag cut is for VH and ttH categories
            if args.do_vbf_filter_study:
                # print("applying VBF filter gen cut!")
                if "dy_" in process:
                    if ("dy_VBF_filter" in process) or (process =="dy_m105_160_vbf_amc"):
                        print("dy_VBF_filter extra!")
                        vbf_filter = ak.fill_none((events.gjj_mass > 350), value=False)
                        prod_cat_cut =  (prod_cat_cut  
                                    & vbf_filter
                        )
                    elif process == "dy_m105_160_amc":
                        print("dy_M-100To200 extra!")
                        vbf_filter = ak.fill_none((events.gjj_mass > 350), value=False) 
                        prod_cat_cut =  (
                            prod_cat_cut  
                            & ~vbf_filter 
                        )
                    else:
                        print(f"no extra processing for {process}")
                        pass
        # else: # we're interested in ggH category
        elif category == "ggh":
            # print("ggH mode!")
            prod_cat_cut =  ~vbf_cut 
            prod_cat_cut = prod_cat_cut & ~btag_cut # btag cut is for VH and ttH categories
        else:
            print("Error: invalid category option!")
            raise ValueError
    
    category_selection = (
        prod_cat_cut & 
        region 
    )
    
    # filter events fro selected category
    
    # print(f"len(events) {process} b4 selection: {len(events)}")
    events = events[category_selection]
    return events


def getDaskHist2Compute(sample_hist_dictByVar2compute, events, sample_hist_empty, var, plot_settings):
    # for process in available_processes:
    if "_nominal" in var:
        plot_var = var.replace("_nominal", "")
    else:
        plot_var = var
    #-----------------------------------------------
    # intialize variables for filling histograms

    group_data_vals = []
    group_DY_vals = []
    group_Top_vals = []
    group_Ewk_vals = []
    group_VV_vals = []
    group_other_vals = []  # histograms not belonging to any other mc bkg group
    group_ggH_vals = [] # there should only be one ggH histogram, but making a list for consistency
    group_VBF_vals = [] # there should only be one VBF histogram, but making a list for consistency
    
    group_data_weights = []
    group_DY_weights = []
    group_Top_weights = []
    group_Ewk_weights = []
    group_VV_weights = []
    group_other_weights = []
    group_ggH_weights = []
    group_VBF_weights = []

    
    for process in available_processes:    
        sample_hist = copy.deepcopy(sample_hist_empty)
        for region_name in args.regions:
            # for each process make new hist
            # print(f"process: {process}")
            try:
                events = loaded_events[process]
            except:
                # print(f"skipping {process}")
                continue
            is_data = "data" in process.lower()
            # print(f"is_data: {is_data}")
            
            #-----------------------------------------------    
            # obtain the category selection

            

            # ------------------------------------------------
            # take the mass region and category cuts 
            # ------------------------------------------------

            
            # events = applyRegionCatCuts(events, args.category, region_name)
            events = dak.map_partitions(applyRegionCatCuts,events, args.category, region_name)
            
            # print(f"len(events) {process} after selection: {len(events)}")
            
            # category_selection = ak.to_numpy(category_selection) # this will be multiplied with weights
            # print(f"len(weights) {process} b4 selection: {len(weights)}")
            # weights = weights[category_selection]
            # print(f"len(weights) {process} after selection: {len(weights)}")

            # extract weights
            if is_data:
                weights = (ak.fill_none(events["wgt_nominal"], value=0.0))
                fraction_weight = 1/events.fraction
            else: # MC
                weights = ak.fill_none(events["wgt_nominal"], value=0.0)
                
                # weights = weights/events.wgt_nominal_muID/ events.wgt_nominal_muIso / events.wgt_nominal_muTrig #  quick test
                # temporary over write
                # print(f"events.fields: {events.fields}")
                # if "separate_wgt_zpt_wgt" in events.fields:
                #     print("removing Zpt rewgt!")
                #     weights = weights/events["separate_wgt_zpt_wgt"]

                
                # for some reason, some nan weights are still passes ak.fill_none() bc they're "nan", not None, this used to be not a problem
                # could be an issue of copying bunching of parquet files from one directory to another, but not exactly sure
                # weights = np.nan_to_num(weights, nan=0.0) 
                fraction_weight = ak.ones_like(events["wgt_nominal"])  # MC is already normalized by lumisonity, so no need for scaling by fraction
            
            # overwrite variable names with two bin ranges
            if ("_range2" in var):
                var_reduced = var.replace("_range2","")
                values = ak.fill_none(events[var_reduced], value=-999.0)
            elif ("_zpeak" in var):
                var_reduced = var.replace("_zpeak","")
                values = ak.fill_none(events[var_reduced], value=-999.0)
            else:
                values = ak.fill_none(events[var], value=-999.0)
                
            # MC samples are already normalized by their xsec*lumi, but data is not
            if process in group_data_processes:
                # print(f"weights: b4 {weights.compute()}")
                weights = weights*fraction_weight
                # print(f"fraction_weight: {fraction_weight.compute()}")
                # print(f"weights after: {weights.compute()}")
            group_name = find_group_name(process, group_dict)
            # print(f"group_name for {process}: {group_name}")
            to_fill_setting = {
            "region" : region_name,
            "channel" : args.category,
            "variation" : "nominal",
            "sample_group": group_name,
            }
            sample_hist = fillHist(sample_hist, to_fill_setting, values, weights)
            
        sample_hist_l.append(sample_hist)

    sample_hist_dictByVar2compute[var] = sample_hist_l
    return sample_hist_dictByVar2compute

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
    # parser.add_argument(
    # "-y",
    # "--year",
    # dest="year",
    # default=[],
    # nargs="*",
    # type=str,
    # action="store",
    # help="string value of year we are calculating",
    # )
    parser.add_argument(
    "-data",
    "--data",
    dest="data_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of data samples represented by alphabetical letters A-H",
    )
    parser.add_argument(
    "-bkg",
    "--background",
    dest="bkg_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of bkg samples represented by shorthands: DY, TT, ST, DB (diboson), EWK",
    )
    parser.add_argument(
    "-sig",
    "--signal",
    dest="sig_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of sig samples represented by shorthands: ggH, VBF",
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
    "--no_ratio",
    dest="no_ratio",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="doesn't plot Data/MC ratio",
    )
    parser.add_argument(
    "--ROOT_style",
    dest="ROOT_style",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses pyROOT functionality instead of mplhep",
    )
    parser.add_argument(
    "--linear_scale",
    dest="linear_scale",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, provide plots in linear scale",
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
    # parser.add_argument(
    # "--vbf",
    # dest="vbf_cat_mode",
    # default=False, 
    # action=argparse.BooleanOptionalAction,
    # help="If true, apply vbf cut for vbf category, else, ggH category cut",
    # )
    parser.add_argument(
    "-cat",
    "--category",
    dest="category",
    default="nocat",
    action="store",
    help="define production mode category. optionsare ggh, vbf and nocat (no category cut)",
    )
    parser.add_argument(
    "--vbf_filter_study",
    dest="do_vbf_filter_study",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, apply vbf filter cut for dy samples",
    )
    #---------------------------------------------------------
    # gather arguments
    args = parser.parse_args()
    available_processes = []
    # if doing VBF filter study, add the vbf filter sample to the DY group
    if args.do_vbf_filter_study:
        # vbf_filter_sample =  "dy_m105_160_vbf_amc"
        vbf_filter_sample =  "dy_VBF_filter_NewZWgt"
        # vbf_filter_sample =  "dy_VBF_filter_customJMEoff"
        # vbf_filter_sample =  "dy_VBF_filter_fromGridpack"
        available_processes.append(vbf_filter_sample)
    
    # take data
    data_samples = args.data_samples
    if len(data_samples) >0:
        for data_letter in data_samples:
            available_processes.append(f"data_{data_letter.upper()}")
        
        # # take data as one group to save load time 
        # available_processes.append(f"data_*")
    # take bkg
    bkg_samples = args.bkg_samples
    if len(bkg_samples) >0:
        for bkg_sample in bkg_samples:
            if bkg_sample.upper() == "DY": # enforce upper case to prevent confusion
                # available_processes.append("dy_M-50")
                # available_processes.append("dy_M-100To200")
                # available_processes.append("dy_m105_160_amc")
                # available_processes.append("dyTo2L_M-50_0j")
                # available_processes.append("dyTo2L_M-50_1j")
                # available_processes.append("dyTo2L_M-50_2j")
                # available_processes.append("dyTo2L_M-50_incl")
                # available_processes.append("dy_m105_160_vbf_amc")
                available_processes.append("dy_M-100To200_MiNNLO")
                available_processes.append("dy_M-50_MiNNLO")
                # available_processes.append("dy_M-100To200_aMCatNLO")
            
            elif bkg_sample.upper() == "TT": # enforce upper case to prevent confusion
                available_processes.append("ttjets_dl")
                available_processes.append("ttjets_sl")
                available_processes.append("tt_inclusive")
                available_processes.append("st_t_top")
                available_processes.append("st_t_antitop")
            elif bkg_sample.upper() == "ST": # enforce upper case to prevent confusion
                available_processes.append("st_tw_top")
                available_processes.append("st_tw_antitop")
            elif bkg_sample.upper() == "VV": # enforce upper case to prevent confusion
                available_processes.append("ww_2l2nu")
                available_processes.append("wz_3lnu")
                available_processes.append("wz_2l2q")
                # available_processes.append("wz_1l1nu2q")
                available_processes.append("zz")
            elif bkg_sample.upper() == "EWK": # enforce upper case to prevent confusion
                # available_processes.append("ewk_lljj_mll105_160_ptj0")
                available_processes.append("ewk_lljj_mll50_mjj120")
            elif bkg_sample.upper() == "OTHER": # enforce upper case to prevent confusion
                available_processes.append("www")
                available_processes.append("wwz")
                available_processes.append("wzz")
                available_processes.append("zzz")
            else:
                print(f"unknown background {bkg_sample} was given!")
        
    # take sig
    sig_samples = args.sig_samples
    if len(sig_samples) >0:
        for sig_sample in sig_samples:
            if sig_sample.upper() == "GGH": # enforce upper case to prevent confusion
                # available_processes.append("ggh_amcPS")
                available_processes.append("ggh_powhegPS")
            elif sig_sample.upper() == "VBF": # enforce upper case to prevent confusion
                available_processes.append("vbf_powheg_dipole")
            else:
                print(f"unknown signal {sig_sample} was given!")
    # gather variables to plot:
    kinematic_vars = ['pt', 'eta', 'phi']
    # kinematic_vars = ['pt']
    variables2plot = []
    if len(args.variables) == 0:
        print("no variables to plot!")
        raise ValueError
    for particle in args.variables:
        if "dimuon" in particle:
            variables2plot.append(f"{particle}_mass")
            variables2plot.append(f"{particle}_pt")
            variables2plot.append(f"{particle}_eta")
            variables2plot.append(f"{particle}_phi")
            variables2plot.append(f"{particle}_cos_theta_cs")
            variables2plot.append(f"{particle}_phi_cs")
            variables2plot.append(f"{particle}_cos_theta_eta")
            variables2plot.append(f"{particle}_phi_eta")
            variables2plot.append(f"mmj_min_dPhi_nominal")
            variables2plot.append(f"mmj_min_dEta_nominal")
            variables2plot.append(f"ll_zstar_log_nominal")
            variables2plot.append(f"dimuon_ebe_mass_res")
            variables2plot.append(f"dimuon_ebe_mass_res_rel")
            variables2plot.append(f"{particle}_rapidity")
        elif "dijet" in particle:
            variables2plot.append(f"jj_dEta_nominal")
            variables2plot.append(f"jj_mass_nominal")
            variables2plot.append(f"jj_pt_nominal")
            
            variables2plot.append(f"jj_dPhi_nominal")
            variables2plot.append(f"zeppenfeld_nominal")
            
            variables2plot.append(f"rpt_nominal")
            variables2plot.append(f"pt_centrality_nominal")
            variables2plot.append(f"nsoftjets2_nominal")
            variables2plot.append(f"htsoft2_nominal")
            variables2plot.append(f"nsoftjets5_nominal")
            variables2plot.append(f"htsoft5_nominal")

            # --------------------------------------------------
            # variables2plot.append(f"gjj_mass")
            
        elif ("mu" in particle) :
            for kinematic in kinematic_vars:
                # plot both leading and subleading muons/jets
                variables2plot.append(f"{particle}1_{kinematic}")
                variables2plot.append(f"{particle}2_{kinematic}")
            variables2plot.append(f"{particle}1_pt_over_mass")
            variables2plot.append(f"{particle}2_pt_over_mass")
        elif ("jet" in particle):
            variables2plot.append(f"njets_nominal")
            for kinematic in kinematic_vars:
                # plot both leading and subleading muons/jets
                variables2plot.append(f"{particle}1_{kinematic}_nominal")
                variables2plot.append(f"{particle}2_{kinematic}_nominal")
            variables2plot.append(f"jet1_qgl_nominal")
            variables2plot.append(f"jet2_qgl_nominal")
       
        else:
            print(f"Unsupported variable: {particle} is given!")


    variables2plot_orig = copy.deepcopy(variables2plot)
    if "jj_mass_nominal" in variables2plot:
        variables2plot += ["jj_mass_nominal_range2"] # add another range to plot
    if "dimuon_mass" in variables2plot:
        variables2plot = ["dimuon_mass_zpeak"] + variables2plot# add another range to plot
    print(f"variables2plot: {variables2plot}")
    # obtain plot settings from config file

    
    if args.category == "ggh":
        plot_setting_fname = "./src/lib/histogram/plot_settings_gghCat_BDT_input.json"
    else: # in no cat case, just use vbfCat plot settings
        plot_setting_fname = "./src/lib/histogram/plot_settings_vbfCat_MVA_input.json"

    print(f"plot_setting_fname: {plot_setting_fname}")
    
    with open(plot_setting_fname, "r") as file:
        plot_settings = json.load(file)
    status = args.status.replace("_", " ")

    # print(f"plot_settings.keys(): {plot_settings.keys()}")
    # raise ValueError
    
    # define client for parallelization 
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
    # record time
    time_step = time.time()

    # load saved parquet files. This increases memory use, but increases runtime significantly
    print(f"available_processes: {available_processes}")
    loaded_events = {} # intialize dictionary containing all the arrays
    for process in tqdm.tqdm(available_processes):
        print(f"loading process {process}..")
        # full_load_path = args.load_path+f"/{process}/*.parquet"
        full_load_path = args.load_path+f"/{process}/*/*.parquet"
        if len(glob.glob(full_load_path)) ==0: # check if there's files in the load path
            full_load_path = args.load_path+f"/{process}/*.parquet" # try coppperheadV1 path, if this also is empty, then skip
        print(f"full_load_path: {full_load_path}")
        try:
            events = dak.from_parquet(full_load_path)
            # target_chunksize = 150_000
            # target_chunksize = 500_000
            target_chunksize = 250_000
            # target_chunksize = 1_000_000
            events = events.repartition(rows_per_partition=target_chunksize)
        except:
            print(f"full_load_path: {full_load_path} Not available. Skipping")
            continue
        # print(f"events.fields: {events.fields}")

        # ------------------------------------------------------
        # select only needed variables to load to save run time
        # ------------------------------------------------------
        
        fields2load = variables2plot_orig + [
            "wgt_nominal",
            # "fraction", 
            # "h_sidebands", 
            # "h_peak", 
            # "z_peak", 
            # "vbf_cut",
            "nBtagLoose_nominal", 
            "nBtagMedium_nominal", 
            "njets_nominal", 
            "dimuon_mass",
            "zeppenfeld_nominal", 
            "jj_mass_nominal", 
            "jet1_pt_nominal", 
            "jj_dEta_nominal", 
            "dimuon_pt", 
            "jet2_pt_nominal",
            "jj_pt_nominal",
            "zeppenfeld_nominal",
        ]
        
            
        # # add in weights
        # for field in events.fields:
        #     if "wgt_nominal" in field:
        #         fields2load.append(field)
                
        is_data = "data" in process.lower()
        if not is_data: # MC sample
            fields2load += ["gjj_mass", "gjj_dR", "gjet1_pt", "gjet2_pt"]
            # temp addition
            # if "separate_wgt_zpt_wgt" in events.fields:
                # fields2load.append("separate_wgt_zpt_wgt")
            

        # filter out redundant fields by using the set object
        fields2load = list(set(fields2load))

        # # TOREMOVE
        # if "separate_wgt_qgl_wgt" in events.fields:
        #     print("removing separate_wgt_qgl_wgt!")
        #     events["wgt_nominal"] = events["wgt_nominal"] / events["separate_wgt_qgl_wgt"] # remove zpt wgt
        # if "separate_wgt_zpt_wgt" in events.fields:
        #     print("removing separate_wgt_zpt_wgt!")
        #     events["wgt_nominal"] = events["wgt_nominal"] / events["separate_wgt_zpt_wgt"] # remove zpt wgt
        
        # events = events[fields2load]
        # # load data to memory using compute()
        # events = ak.zip({
        #     field : events[field] for field in events.fields
        # }).compute()
        loaded_events[process] = events
    print("finished loading parquet files!")

    import mplhep as hep
    import matplotlib.pyplot as plt
    import matplotlib
    # hep.style.use("CMS")
    # Load CMS style including color-scheme (it's an editable dict)
    plt.style.use(hep.style.CMS)
    # this mplhep implementation assumes non-empty data; otherwise, it will crash
    # Dictionary for histograms and binnings

    # initialize histograms
    regions = ["z-peak", "signal", "h-peak", "h-sidebands"] # full list of possible regions to loop over
    channels = ["nocat", "vbf", "ggh"] # full list of possible channels to loop over
    variations = ["nominal"]
    sample_groups = list(group_dict.keys()) + ["other"]
    # years = ["2018", "2017", "2016postVFP", "2016preVFP"]
    sample_hist = (
            hda.Hist.new.StrCat(regions, name="region")
            .StrCat(channels, name="channel")
            .StrCat(["value", "sumw2"], name="val_sumw2")
            .StrCat(sample_groups, name="sample_group")
            # .StrCat(years, name="year")
    )
    # add axis for systematic variation
    sample_hist_dictByVar = {} 
    sample_hist = sample_hist.StrCat(variations, name="variation")
    for var in variables2plot:
        # for process in available_processes:
        if "_nominal" in var:
            plot_var = var.replace("_nominal", "")
        else:
            plot_var = var
        if plot_var not in plot_settings.keys():
            print(f"variable {var} not configured in plot settings!")
            continue
        binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
        print(f"var: {var}")
        sample_hist_dictByVar[var] = sample_hist.Var(binning, name=var).Double()
    # sample_hist_empty = sample_hist.Double()
    # sample_hist_l = []
    # fill the histograms
    sample_hist_dictByVar2compute = {}
    for var in tqdm.tqdm(variables2plot):
    
        sample_hist_empty = sample_hist_dictByVar[var]
        sample_hist_l = []
        var_step = time.time()
        if "_nominal" in var:
            plot_var = var.replace("_nominal", "")
        else:
            plot_var = var
        if plot_var not in plot_settings.keys():
            print(f"variable {var} not configured in plot settings!")
            continue
        
            
        sample_hist_dictByVar2compute = getDaskHist2Compute(sample_hist_dictByVar2compute, events, sample_hist_empty, var, plot_settings)

    
    # done with looping over process and variables we now compute
    sample_hist_dictByVarComputed = dask.compute(sample_hist_dictByVar2compute)[0]
    # print(f"sample_hist_dictByVarComputed: {sample_hist_dictByVarComputed}")
    # print(f"args.regions: {args.regions}")
    for region_name in args.regions:
        for var in tqdm.tqdm(variables2plot):
            if args.linear_scale:
                do_logscale = False
            else:
                do_logscale = True  
            full_save_path = args.save_path+f"/{args.year}/mplhep/Reg_{region_name}/Cat_{args.category}/{args.label}"
            # plotVars(sample_hist_dictByVarComputed, var, full_save_path, do_logscale=do_logscale)
            data_dict = {}
            bkg_MC_dict = {}
            sig_MC_dict = {}
            # for process in available_processes: 
            for group_name in sample_groups: 
                sample_hist_l = sample_hist_dictByVarComputed[var]
                sample_hist = sum(sample_hist_l)
                # print(f"sample_hist: {sample_hist}")
                to_project_setting = {
                    "region" : region_name,
                    "channel" : args.category,
                    "variation" : "nominal",
                    "sample_group": group_name,
                }
                
                to_project_setting_val = to_project_setting.copy()
                to_project_setting_val["val_sumw2"] = "value"
                hist_val = sample_hist[to_project_setting_val].project(var).values()
                #------------------------------------------------------
                to_project_setting_w2 = to_project_setting.copy()
                to_project_setting_w2["val_sumw2"] = "sumw2"
                hist_w2 = sample_hist[to_project_setting_w2].project(var).values()
                # print(f"to_project_setting: {to_project_setting}")
                # print(f"hist_val: {hist_val}")
                # print(f"hist_w2: {hist_w2}")
                if np.sum(hist_val)==0: # skip processes that doesn't have anything
                    continue
                hist_dict = {
                    "hist_arr" : hist_val,
                    "hist_w2_arr": hist_w2
                }
                
                
                if "data" in group_name: # data
                    data_dict = hist_dict
                elif "ggH" in group_name or "VBF" in group_name: # signal
                    sig_MC_dict[group_name] = hist_dict
                else: # bkg MC
                    bkg_MC_dict[group_name] = hist_dict
            # order bkg_MC_dict in a specific way for plotting, smallest yielding process first:
            bkg_MC_order = ["other", "VV", "Ewk", "Top", "DY"]
            bkg_MC_dict = {process: bkg_MC_dict[process] for process in bkg_MC_order if process in bkg_MC_dict}
            if len(data_dict) ==0:
                print(f"empty histograms for {var} skipping!")
                continue

            # -------------------------------------------------------
            # All data are prepped, now plot Data/MC histogram
            # -------------------------------------------------------
            full_save_path = args.save_path+f"/{args.year}/mplhep/Reg_{region_name}/Cat_{args.category}/{args.label}"
            # print(f"full_save_path: {full_save_path}")
            
            
            if not os.path.exists(full_save_path):
                os.makedirs(full_save_path)
            full_save_fname = f"{full_save_path}/{var}.pdf"


            plot_var = getPlotVar(var)
            if plot_var not in plot_settings.keys():
                print(f"variable {var} not configured in plot settings!")
                continue
            binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])
              
            plotDataMC_compare(
                binning, 
                data_dict, 
                bkg_MC_dict, 
                full_save_fname,
                sig_MC_dict=sig_MC_dict,
                title = "", 
                x_title = plot_settings[plot_var].get("xlabel"), 
                y_title = plot_settings[plot_var].get("ylabel"),
                lumi = args.lumi,
                status = status,
                log_scale = do_logscale,
            )
            


        

        # var_elapsed = round(time.time() - var_step, 3)
        # print(f"Finished processing {var} in {var_elapsed} s.")
    # ROOT style or mplhep style ends here --------------------------------------
    
    time_elapsed = round(time.time() - time_step, 3)
    print(f"Finished in {time_elapsed} s.")