import time
import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client
from omegaconf import OmegaConf
from typing import Tuple, List, Dict
import ROOT as rt
import ROOT
# from modules.fit_functions import MakeFEWZxBernDof3, plot_6_23, plot_6_26, getSigBkgPdf
# from modules.fit_functions import getBWZ_gamma, getBWZxBern, getLandxBern, getFEWZxBern
import argparse
import os
import copy
import pandas as pd
# from modules.utils import getGOF_KS
from src.corrections.jet import applyUpDown, getJecJerUncertainties



# def extract_values(cfg):
#     """
#     Recursively extract all values from an OmegaConf object into a flat list.
#     """
#     values = []
#     for v in cfg.values():
#         if isinstance(v, dict) or isinstance(v, OmegaConf):
#             values.extend(extract_values(v))
#         else:
#             # values.append(v)
#             values.extend(v)

#     values = list(set(values)) # make a list of unique vals
#     return values

# def getJecJerUncertainties(yaml_filename, year=None):
#     # Load YAML file
#     # cfg = OmegaConf.load("/work/users/yun79/Run3/copperheadV2/configs/parameters/jec.yaml")
#     cfg = OmegaConf.load(yaml_filename)
#     cfg = cfg["jec_parameters"]["jec_unc_to_consider"]
#     # Get list of all values
#     if year is None: # extract all years
#         jec_uncs = extract_values(cfg)
#     else:
#         jec_uncs = cfg[year]
#     jer_uncs = [f"jer{i}" for i in range(1,7)]
#     return jec_uncs + jer_uncs

def fillWgtVarations(df : pd.DataFrame, events, nSubcats : int):
    print(f"fillWgtVarations b4: \n {df}")
    wgt_fields = [col for col in df.columns if "wgt" in col]
    for subCat_ix in range(nSubcats):
        subcat_filter = events.subCategory_idx == subCat_ix
        row_data = {}
        for wgt_name in wgt_fields:
            wgt_values = events[wgt_name]
            wgt_values = wgt_values[subcat_filter]
            wgt_yield = ak.sum(wgt_values)
            # #debugging
            # if wgt_name == "nominal":
            #     print(f"{wgt_name} subCat{subCat_ix} wgt_yield: {wgt_yield}")
            
            row_data[wgt_name] = wgt_yield
        df.loc[f"subCat{subCat_ix}"] = row_data
    print(f"fillWgtVarations after: \n {df}")
    # raise ValueError
    
    return df


def getRelativeYield2Nominal(df):
    df_rel = df.div(df["wgt_nominal"], axis=0)
    return df_rel


# def fillJecJerVarations(df : pd.DataFrame, events, nSubcats : int, jec_unc_fields : list):
#     wgt_name = "wgt_nominal"
#     for jec_unc_name in jec_unc_fields:
#         col_data = []
#         for subCat_ix in range(nSubcats):
#             subCat_field = f"subCategory_idx_{jec_unc_name}"
#             subcat_filter = events[subCat_field] == subCat_ix
#             wgt_values = events[wgt_name]
#             wgt_values = wgt_values[subcat_filter]
#             wgt_yield = ak.sum(wgt_values)
#             col_data.append(wgt_yield)
#             # print(f"{jec_unc_name} subcat {subCat_ix} yield: {wgt_yield}")
#         df[jec_unc_name] = col_data
#     return df


def fillJecJerVarationsByYear(df : pd.DataFrame, load_path, years : list, nSubcats : int, jec_unc_fields : list):
    wgt_name = "wgt_nominal"
    jec_unc_fields = jec_unc_fields + ["nominal"] # add nominal
    for subCat_ix in range(nSubcats):
        for year in years:
            year_load_path = load_path.replace("year_value", year)
            # print(f"year_load_path: {year_load_path}")
            events = ak.from_parquet(year_load_path)
            # print(f"events.fields: {events.fields}")
            row_data = {}
            # row_data["year"] =  year
            for jec_unc_name in jec_unc_fields:
                if jec_unc_name == "nominal":
                    subCat_field = f"subCategory_idx"
                else:
                    subCat_field = f"subCategory_idx_{jec_unc_name}"
                if not subCat_field in events.fields: 
                    continue # ie. Absolute_2017 in events from 2018 stage2
                subcat_filter = events[subCat_field] == subCat_ix
                wgt_values = events[wgt_name]
                wgt_values = wgt_values[subcat_filter]
                wgt_yield = ak.sum(wgt_values)
                row_data[jec_unc_name] = wgt_yield
                
                # print(f"{jec_unc_name} subcat {subCat_ix} yield: {wgt_yield}")
            df.loc[f"subCat{subCat_ix}_{year}"] = row_data
    return df

def combine_dfByYear(df_JecByYear, jec_unc_fields : list, years : list, nSubCats : int):
    # jec_unc_fields = df_JecByYear.columns
    # print(f"combineDfByYear jec_unc_fields: {jec_unc_fields}")

    # define out df
    row_labels = [f"subCat{i}" for i in range(nSubcats)]
    out_df = pd.DataFrame(index=row_labels, columns=jec_unc_fields)
    # print(f"combineDfByYear out_df b4: {out_df}")
    
    
    for subCat_ix in range(nSubcats):
        row_data = {
            jec_unc_field : 0 for jec_unc_field in jec_unc_fields
        }
        print(row_data)
        for year in years:
            row_idx = f"subCat{subCat_ix}_{year}"
            for jec_unc_field in jec_unc_fields:
                value = df_JecByYear.loc[row_idx, jec_unc_field]
                # print(f"combineDfByYear {row_idx} {jec_unc_field} value: {value}")
                
                if pd.isna(value):
                    value = df_JecByYear.loc[row_idx, "nominal"]
                    row_data[jec_unc_field] += value
                else:
                    row_data[jec_unc_field] += value
        # print(f"subCat{subCat_ix} row_data: {row_data}")
        combined_row_idx = f"subCat{subCat_ix}"
        out_df.loc[combined_row_idx] = row_data
    # print(f"combineDfByYear out_df after: {out_df}")

    return out_df
    

def getProcessedEvents(events, fields2load, jec_unc_fields):
    bdt_fields = [
        "BDT_score",
        "subCategory_idx",
    ]
    bdt_fields_variation = [] 
    for jec_unc_field in jec_unc_fields:
        bdt_fields_variation = bdt_fields_variation + [f"{bdt_field}_{jec_unc_field}" for bdt_field in bdt_fields]
    wgt_fields = [field for field in events.fields if "wgt" in field]
    fields_total = fields2load + bdt_fields + bdt_fields_variation + wgt_fields
    print(fields_total)
    processed_events = ak.zip({field: events[field] for field in fields_total}).compute()
    return processed_events


def flipDfAddSample(df, sample : str):
    """
    flips the rows and columns of the given df and adds sample string value to each column
    """
    df_flipped = df.T  # Transpose (flip rows and columns)
    df_flipped.columns = df_flipped.columns.astype(str) + f"_{sample}"
    return df_flipped
    
def getDataCardLikeDf(samples, base_path):
    # collect dfs
    df_dict = {}
    for sample in samples:
        df_path = f""
        df = pd.read_csv(df_path)
        df = flipDfAddSample(df, sample)
        df_dict[sample] = df
    
    # initialize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-load",
    "--load_path",
    dest="load_path",
    default=None,
    action="store",
    help="save path to store stage1 output files",
    )
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="all",
    action="store",
    help="string value of year we are calculating",
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
    "-l",
    "--label",
    dest="label",
    default="",
    action="store",
    help="MVA model name to load",
    )
    args = parser.parse_args()
    # check for valid arguments
    if args.load_path == None:
        print("load path to load stage1 output is not specified!")
        raise ValueError

    category = args.category.lower()
    nSubcats = 5
    samples = [
        "ggh",
        "vbf"
    ]
    # sample = "ggh" #FIXME
    # year = "2017" #FIXME

    # make save directory
    base_path = f"./datacards/{args.year}/{args.label}"
    plot_save_path = base_path
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    
    for sample in samples:
        year = args.year
        fname = f"processed_events_sigMC_{sample}.parquet"
        if year=="all":
            load_path = f"{args.load_path}/*/{fname}"
        elif year=="2016only":
            load_path = f"{args.load_path}/2016*/{fname}"
        else:
            load_path = f"{args.load_path}/{year}/{fname}"
        print(f"load_path: {load_path}")
        # processed_events = ak.from_parquet(load_path)
        events = dak.from_parquet(load_path)
        # print(f"events.fields: {events.fields}")
        fields2load  = [
            "dimuon_mass",
        ]
        jec_unc_fields = ["Absolute", "FlavorQCD"]
        jec_unc_fields = applyUpDown(jec_unc_fields)
        processed_events = getProcessedEvents(events, fields2load, jec_unc_fields)
        print(f"processed_events.wgt_nominal: {processed_events.wgt_nominal}")
        # print(f"processed_events.wgt_nominal len: {ak.num(processed_events.wgt_nominal, axis=0)}")
        # print(f"processed_events.wgt_l1prefiring_up: {processed_events.wgt_l1prefiring_up}")
        
        print(f"processed_events length: {ak.num(processed_events.dimuon_mass, axis=0)}")
        print("events loaded!")
    
        events = processed_events
        wgt_fields = [field for field in events.fields if "wgt" in field]

        fields2process = wgt_fields + jec_unc_fields
        # fields2process = fields2process 
        print(fields2process)
        # raise ValueError
        print(events.fields)
        
    

        # --------------------------------------------------------------
        # fill in wgt unc
        # --------------------------------------------------------------

        # Define your row labels
        row_labels = [f"subCat{i}" for i in range(nSubcats)]
        
        # Create the empty DataFrame
        df_wgts = pd.DataFrame(index=row_labels, columns=wgt_fields)
        df_wgts = fillWgtVarations(df_wgts, events, nSubcats)
        # df_wgts = fillJecJerVarations(df_wgts, events, nSubcats, jec_unc_fields) # we don't use this function any more
        df_wgts.to_csv(f"{base_path}/{sample}_abs_yield.csv")
        df_wgts_rel = getRelativeYield2Nominal(df_wgts)
        df_wgts_rel.to_csv(f"{base_path}/{sample}_relative2nominal.csv")



        # --------------------------------------------------------------
        # fill in jec/jer unc
        # --------------------------------------------------------------

        years = ["2018", "2017", "2016postVFP", "2016preVFP"]
        # years = ["2017"] 
        row_labels = []
        for year in years:
            row_labels = row_labels + [f"subCat{i}_{year}" for i in range(nSubcats)]
            
        # df = pd.DataFrame(index=row_labels, columns=(jec_unc_fields+["year"))
        # jec_unc_fields = ["Absolute", "FlavorQCD", "Absolute_2018", "Absolute_2017"]
        jec_yml_path = "/work/users/yun79/Run3/copperheadV2/configs/parameters/jec.yaml"
        jec_unc_fields = getJecJerUncertainties(jec_yml_path)
        jec_unc_fields = applyUpDown(jec_unc_fields)
        print(f"jec_unc_fields: {jec_unc_fields}")
        # raise ValueError
        df_JecByYear = pd.DataFrame(index=row_labels, columns=(jec_unc_fields + ["nominal"]))
        fname = f"processed_events_sigMC_{sample}.parquet"
        load_path = f"{args.load_path}/year_value/{fname}"
        print(f"load_path: {load_path}")
        df_JecByYear = fillJecJerVarationsByYear(df_JecByYear, load_path, years, nSubcats, jec_unc_fields)
        print(f"df_JecByYear: {df_JecByYear}")
        df_JecCombined = combine_dfByYear(df_JecByYear, jec_unc_fields, years, nSubcats)
        df_JecCombined.to_csv(f"{base_path}/{sample}_jecUnc_absYield.csv")

        df_JecCombined_rel = df_JecCombined.div(df_wgts["wgt_nominal"], axis=0)
        df_JecCombined_rel.to_csv(f"{base_path}/{sample}_jecUnc_relYield.csv")

        
        df_total = pd.concat([df_wgts_rel, df_JecCombined_rel], axis=1)

        df_total.to_csv(f"{base_path}/{sample}_total_relYield.csv")

    # # --------------------------------------------------------------
    # # convert df to something more datacard-like
    # # --------------------------------------------------------------
    # df_datacardLike = getDataCardLikeDf(samples, base_path)
    # df_datacardLike.to_csv(f"{base_path}/datacardLikeDf.csv")
        

        