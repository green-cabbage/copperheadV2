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
from src.corrections.jet import applyUpDown

def fillWgtVarations(df : pd.DataFrame, events, nSubcats : int):
    wgt_fields = [col for col in df.columns if "wgt" in col]
    for subCat_ix in range(nSubcats):
        subcat_filter = events.subCategory_idx == subCat_ix
        row_data = {}
        for wgt_name in wgt_fields:
            wgt_values = events[wgt_name]
            wgt_values = wgt_values[subcat_filter]
            wgt_yield = ak.sum(wgt_values)
            row_data[wgt_name] = wgt_yield
        df.loc[f"subCat{subCat_ix}"] = row_data
    return df



def getRelativeYield2Nominal(df):
    df_rel = df.div(df["wgt_nominal"], axis=0)
    return df_rel


def fillJecJerVarations(df : pd.DataFrame, events, nSubcats : int, jec_unc_fields : list):
    wgt_name = "wgt_nominal"
    for jec_unc_name in jec_unc_fields:
        col_data = []
        for subCat_ix in range(nSubcats):
            subCat_field = f"subCategory_idx_{jec_unc_name}"
            subcat_filter = events[subCat_field] == subCat_ix
            wgt_values = events[wgt_name]
            wgt_values = wgt_values[subcat_filter]
            wgt_yield = ak.sum(wgt_values)
            col_data.append(wgt_yield)
            # print(f"{jec_unc_name} subcat {subCat_ix} yield: {wgt_yield}")
        df[jec_unc_name] = col_data
    return df


def fillJecJerVarationsByYear(df : pd.DataFrame, load_path, years : list, nSubcats : int, jec_unc_fields : list):
    wgt_name = "wgt_nominal"
    
    for subCat_ix in range(nSubcats):
        for year in years:
            year_load_path = load_path.replace("year_value", year)
            print(f"year_load_path: {year_load_path}")
            events = ak.from_parquet(year_load_path)
            print(f"events.fields: {events.fields}")
            row_data = {}
            for jec_unc_name in jec_unc_fields:
                subCat_field = f"subCategory_idx_{jec_unc_name}"
                subcat_filter = events[subCat_field] == subCat_ix
                wgt_values = events[wgt_name]
                wgt_values = wgt_values[subcat_filter]
                wgt_yield = ak.sum(wgt_values)
                row_data[jec_unc_name] = wgt_yield
                
                # print(f"{jec_unc_name} subcat {subCat_ix} yield: {wgt_yield}")
        df.loc[f"subCat{subCat_ix}"] = row_data
    print(df)
    raise ValueError
    return df

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
        # "vbf"
    ]
    # sample = "ggh" #FIXME
    year = args.year
    # year = "2018" #FIXME
    for sample in samples:
        fname = f"processed_events_sigMC_{sample}.parquet"
        if year=="all":
            load_path = f"{args.load_path}/*/{fname}"
        elif year=="2016only":
            load_path = f"{args.load_path}/2016*/{fname}"
        else:
            load_path = f"{args.load_path}/{year}/{fname}"
        print(f"load_path: {load_path}")
        # processed_events = ak.from_parquet(load_path)
        processed_events = dak.from_parquet(load_path)
        processed_events = processed_events.compute()
        print(f"processed_events.wgt_nominal: {processed_events.wgt_nominal}")
        print(f"processed_events.wgt_l1prefiring_up: {processed_events.wgt_l1prefiring_up}")
        
        print(f"processed_events length: {ak.num(processed_events.dimuon_mass, axis=0)}")
        print("events loaded!")
    
        events = processed_events
        fields2process = [field for field in events.fields if "wgt" in field]

        jec_unc_fields = ["Absolute", "FlavorQCD"]
        jec_unc_fields = applyUpDown(jec_unc_fields)
        # fields2process = fields2process + jec_unc_fields
        fields2process = fields2process #+ jec_unc_fields
        print(fields2process)
        # raise ValueError
        print(events.fields)
        # make plot directory
        base_path = f"./datacards/{args.year}/{args.label}"
        plot_save_path = base_path
        if not os.path.exists(plot_save_path):
            os.makedirs(plot_save_path)
    
        # Define your row labels
        row_labels = [f"subCat{i}" for i in range(nSubcats)]
        
        # Create the empty DataFrame
        df = pd.DataFrame(index=row_labels, columns=fields2process)
        df = fillWgtVarations(df, events, nSubcats)
        # df = fillJecJerVarations(df, events, nSubcats, jec_unc_fields)
        df.to_csv(f"{base_path}/{sample}_abs_yield.csv")
        df_rel = getRelativeYield2Nominal(df)
        df_rel.to_csv(f"{base_path}/{sample}_relative2nominal.csv")

        # fill in jec/jer samples
        years = ["2018", "2017"]
        df = pd.DataFrame(index=row_labels, columns=(jec_unc_fields+["year"]))
        fname = f"processed_events_sigMC_{sample}.parquet"
        load_path = f"{args.load_path}/year_value/{fname}"
        print(f"load_path: {load_path}")
        df = fillJecJerVarationsByYear(df, load_path, years, nSubcats, jec_unc_fields)
        
        print(df)
        

