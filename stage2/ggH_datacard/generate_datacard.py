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


def fillWgtVarations(df, events, nSubcats):
    wgt_fields = [col for col in df.columns if "wgt" in col]
    for cubCat_ix in range(nSubcats):
        subcat_filter = events.subCategory_idx == cubCat_ix
        row_data = {}
        for wgt_name in wgt_fields:
            wgt_values = events[wgt_name]
            wgt_values = wgt_values[subcat_filter]
            wgt_yield = ak.sum(wgt_values)
            row_data[wgt_name] = wgt_yield
        df.loc[f"subCat{cubCat_ix}"] = row_data
    return df

def getRelativeYield2Nominal(df):
    df_rel = df.div(df["wgt_nominal"], axis=0)
    return df_rel


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
    # sample = "ggh"
    for sample in samples:
        fname = f"processed_events_sigMC_{sample}.parquet"
        if args.year=="all":
            load_path = f"{args.load_path}/*/{fname}"
        elif args.year=="2016only":
            load_path = f"{args.load_path}/2016*/{fname}"
        else:
            load_path = f"{args.load_path}/{args.year}/{fname}"
        print(f"load_path: {load_path}")
        processed_events = ak.from_parquet(load_path)
        print(f"processed_events length: {ak.num(processed_events.dimuon_mass, axis=0)}")
        print("events loaded!")
    
        events = processed_events
        fields2process = [field for field in events.fields if "wgt" in field]
        # print(fields2process)
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
        print(df)
        df.to_csv(f"{base_path}/{sample}_abs_yield.csv")
        df_rel = getRelativeYield2Nominal(df)
        df_rel.to_csv(f"{base_path}/{sample}_relative2nominal.csv")

