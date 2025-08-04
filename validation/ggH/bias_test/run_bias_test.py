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
from modules.fit_functions import MakeFEWZxBernDof3, plot_6_23, plot_6_26, getSigBkgPdf
from modules.fit_functions import getBWZ_gamma, getBWZxBern, getLandxBern, getFEWZxBern
import argparse
import os
import copy
import pandas as pd
from modules.utils import getGOF_KS


    
def create_core_pdf(pdf_type, subcat_index, x, init_vals):
    """
    Create a RooAbsPdf of a given type with per-subcategory RooRealVars,
    initializing them with provided values and ranges.

    Args:
        pdf_type (str): 'BWZRedux' or 'sumExp'.
        subcat_index (int): Index for naming the PDF.
        x (RooRealVar): Observable (formerly 'mass').
        init_vals (dict): Dict with float values keyed by param name.

    Returns:
        tuple: (RooAbsPdf, [list of RooRealVar parameters])
    """
    prefix = f"subCat{subcat_index}_{pdf_type}"

    if pdf_type == "BWZRedux":
        a = rt.RooRealVar(f"{prefix}_a_coeff", f"{prefix}_a_coeff", 
                          init_vals["a_coeff"], -0.5, 0.5)
        b = rt.RooRealVar(f"{prefix}_b_coeff", f"{prefix}_b_coeff", 
                          init_vals["b_coeff"], -0.02, 0.02)
        c = rt.RooRealVar(f"{prefix}_c_coeff", f"{prefix}_c_coeff", 
                          init_vals["c_coeff"], -10.0, 10.0)
        pdf = rt.RooModZPdf(prefix, prefix, x, a, b, c)
        return pdf, [a, b, c]

    elif pdf_type == "sumExp":
        a1 = rt.RooRealVar(f"{prefix}_a1_coeff", f"{prefix}_a1_coeff", 
                           init_vals["a1_coeff"], -2.0, 1.0)
        a2 = rt.RooRealVar(f"{prefix}_a2_coeff", f"{prefix}_a2_coeff", 
                           init_vals["a2_coeff"], -2.0, 1.0)
        f  = rt.RooRealVar(f"{prefix}_f_coeff", f"{prefix}_f_coeff", 
                           init_vals["f_coeff"], 0.0, 1.0)
        pdf = rt.RooSumTwoExpPdf(prefix, prefix, x, a1, a2, f)
        return pdf, [a1, a2, f]

    else:
        raise ValueError(f"Unsupported PDF type: {pdf_type}")



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
    # load_path = "/work/users/yun79/stage2_output/ggH/test/processed_events_data.parquet"
    # load_path = f"{args.load_path}/{category}/{args.year}/processed_events_data.parquet"
    # if args.year=="all":
    #     load_path = f"{args.load_path}/{category}/*/processed_events_data.parquet"
    # elif args.year=="2016only":
    #     load_path = f"{args.load_path}/{category}/2016*/processed_events_data.parquet"
    # else:
    #     load_path = f"{args.load_path}/{category}/{args.year}/processed_events_data.parquet"

    # remove category we assume that the load_path already has category specified
    if args.year=="all":
        load_path = f"{args.load_path}/*/processed_events_data.parquet"
    elif args.year=="2016only":
        load_path = f"{args.load_path}/2016*/processed_events_data.parquet"
    else:
        load_path = f"{args.load_path}/{args.year}/processed_events_data.parquet"
    print(f"load_path: {load_path}")
    processed_eventsData = ak.from_parquet(load_path)
    print(f"processed_eventsData length: {ak.num(processed_eventsData.dimuon_mass, axis=0)}")
    print("events loaded!")

    # make plot directory
    base_path = f"./validation/stage3/{args.year}/{args.label}"
    plot_save_path = base_path
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    # Define your list of column names
    column_list = ["year", "category", "dataset", "yield"]
    yield_df = pd.DataFrame(columns=column_list)

    
    device = "cpu"
    # device = "cuda"
    # rt.RooAbsReal.setCudaMode(True)
    # Create model for physics sample
    # -------------------------------------------------------------
    # Create observables
    mass_name = "mh_ggh"
    mass = rt.RooRealVar(mass_name, mass_name, 120, 110, 150)
    nbins = 800
    mass.setBins(nbins)
    mass.setRange("hiSB", 135, 150 )
    mass.setRange("loSB", 110, 115 )
    mass.setRange("h_peak", 115, 135 )
    mass.setRange("full", 110, 150 )
    # fit_range = "loSB,hiSB" # we're fitting bkg only
    fit_range = "hiSB,loSB" # we're fitting bkg only
    
    subCatIdx_name = "subCategory_idx"
    # subCatIdx_name = "subCategory_idx_val"

    # Initialize BWZ Redux
    # --------------------------------------------------------------


    # # trying bigger range do that I don't get warning message from combine like: [WARNING] Found parameter BWZ_Redux_a_coeff at boundary (within ~1sigma)
    name = f"BWZ_Redux_a_coeff"
    a_coeff = rt.RooRealVar(name,name, 5.1288e-02,-0.5,0.5)
    name = f"BWZ_Redux_b_coeff"
    b_coeff = rt.RooRealVar(name,name, -1.3658e-04,-0.02,0.02)
    name = f"BWZ_Redux_c_coeff"
    c_coeff = rt.RooRealVar(name,name, 2.0602e+00,-10.0,10.0)
    # # old end --------------------------------------------------

    # sumexp subcat
    name = f"RooSumTwoExpPdf_a1_coeff"
    a1_coeff = rt.RooRealVar(name,name, -1.4756e-01,-2.0,1)
    name = f"RooSumTwoExpPdf_a2_coeff"
    a2_coeff = rt.RooRealVar(name,name, -3.4552e-02,-2.0,1)
    name = f"RooSumTwoExpPdf_f_coeff"
    f_coeff = rt.RooRealVar(name,name,  2.4864e-01,0.0,1.0)


    nSubCats = 5
    nSubCats = 1 # FIXME
    coreFunction_dict = {
        "BWZRedux" : [],
        # "BwzGamma" : [],
        # "BWZxBern" : [],
        "sumExp" : [],
        # "PowerLaw" : [],
        # "FEWZxBern" : [],
        # "LandauxBern" : [],
    }
    
    # subCat 0
    # for ix in range(nSubCats):
    #     # BWZRedux
    #     core_func_name = "BWZRedux" # match with one of the keys of coreFunction_dict
    #     # name = f"subCat{ix}_{core_func_name}"
    #     core_func, _ = create_core_pdf(core_func_name, ix, mass, init_vals)
        
    #     # core_func = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff)
    #     coreFunction_dict[core_func_name].append(core_func)

        
    #     # sumExp
    #     core_func_name = "sumExp" # match with one of the keys of coreFunction_dict
    #     name = f"subCat{ix}_{core_func_name}"
    #     core_func = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
    #     coreFunction_dict[core_func_name].append(core_func)

    bwz_init_vals = {
        "a_coeff": 5.1288e-02,
        "b_coeff": -1.3658e-04,
        "c_coeff": 2.0602e+00,
    }
    
    sumexp_init_vals = {
        "a1_coeff": -1.4756e-01,
        "a2_coeff": -3.4552e-02,
        "f_coeff": 2.4864e-01,
    }
    
    all_params = []
    
    for ix in range(nSubCats):
        # BWZRedux
        pdf_bwz, params_bwz = create_core_pdf("BWZRedux", ix, mass, bwz_init_vals)
        coreFunction_dict["BWZRedux"].append(pdf_bwz)
        all_params.extend(params_bwz)

        # sumExp
        pdf_sumexp, params_sumexp = create_core_pdf("sumExp", ix, mass, sumexp_init_vals)
        coreFunction_dict["sumExp"].append(pdf_sumexp)
        all_params.extend(params_sumexp)

        
    # ---------------------------------------------------------------
    # Extract Data over all sub cats
    # ---------------------------------------------------------------

    # also do for all subcats for later use
    allCat_mass_arr = processed_eventsData.dimuon_mass
    allCat_mass_arr  = ak.to_numpy(allCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData = rt.RooDataSet.from_numpy({mass_name: allCat_mass_arr}, [mass])
    roo_histData_allCat = rt.RooDataHist("allCat_rooHist","allCat_rooHist", rt.RooArgSet(mass), roo_datasetData)
    
    # ---------------------------------------------------------------
    # Initialize Data for Bkg models to fit to
    # ---------------------------------------------------------------
     
    # do for cat idx 0
    subCat_filter = (processed_eventsData[subCatIdx_name] == 0)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat0 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat0 = rt.RooDataHist("subCat0_rooHist","subCat0_rooHist", rt.RooArgSet(mass), roo_datasetData_subCat0)

    # do for cat idx 1
    subCat_filter = (processed_eventsData[subCatIdx_name] == 1)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat1 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat1 = rt.RooDataHist("subCat1_rooHist","subCat1_rooHist", rt.RooArgSet(mass), roo_datasetData_subCat1)

    # do for cat idx 2
    subCat_filter = (processed_eventsData[subCatIdx_name] == 2)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat2 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat2 = rt.RooDataHist("subCat2_rooHist","subCat2_rooHist", rt.RooArgSet(mass), roo_datasetData_subCat2)

    # do for cat idx 3
    subCat_filter = (processed_eventsData[subCatIdx_name] == 3)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat3 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat3 = rt.RooDataHist("subCat3_rooHist","subCat3_rooHist", rt.RooArgSet(mass), roo_datasetData_subCat3)

    # do for cat idx 4
    subCat_filter = (processed_eventsData[subCatIdx_name] == 4)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat4 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat4 = rt.RooDataHist("subCat4_rooHist","subCat4_rooHist", rt.RooArgSet(mass), roo_datasetData_subCat4)



    
    
    #----------------------------------------------------------------------------
    # Do fit to the core function
    # ---------------------------------------------------------------------------


    # fit FEWZxBern separately
    for core_func_name, core_func_l in coreFunction_dict.items():
        print(f"core_func_name: {core_func_name}")
        
        core_func = core_func_l[0]
        _ = core_func.fitTo(roo_histData_subCat0, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,SumW2Error=True)
        fitResult = core_func.fitTo(roo_histData_subCat0, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,SumW2Error=True)
        fitResult.Print()

    print("Success!")
    
    raise ValueError
    # ---------------------------------------------------
    # Make CORE-PDF
    # ---------------------------------------------------

    # subCat 0 
    cat_subCat0 = rt.RooCategory("pdf_index_ggh","Index of Pdf which is active"); # name of category index should stay same across subCategories
    
    # // Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # // 0 == BWZRedux
    # // 1 == BwzGamma
    # // 2 == BWZxBern
    # // 3 == sumExp
    # // 4 == PowerLaw
    # // 5 == FEWZxBern
    # // 6 == LandauxBern
    
    # FEWZxBern Sumexp is less dependent to dimuon mass as stated in line 1585 of RERECO AN
    # I suppose BWZredux is there bc it's the one function with overall least bias (which is why BWZredux is used if CORE-PDF is not used)
    pdf_list_subCat0 = rt.RooArgList(
        model_subCat0_sumExp,
        model_subCat0_BWZRedux,
        model_subCat0_FEWZxBern,
    )
    corePdf_subCat0 = rt.RooMultiPdf("CorePdf_subCat0","CorePdf_subCat0",cat_subCat0,pdf_list_subCat0)
    # penalty = 0 # as told in https://cms-talk.web.cern.ch/t/combine-fitting-not-working-with-roomultipdf-leading-to-bad-signal-significance/44238/
    penalty = 0.5
    corePdf_subCat0.setCorrectionFactor(penalty) 
    nevents = roo_datasetData_subCat0.sumEntries() # these are data, so all weights are one, thus no need to sum over the weights, though ofc you can just do that too
    print(f"roo_datasetData_subCat0 sumentries: {nevents}")
    bkg_subCat0_norm = rt.RooRealVar(corePdf_subCat0.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value
    
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat0"],
        "dataset": ["data"], 
        "yield": [nevents]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    


    # subCat 1 
    cat_subCat1 = rt.RooCategory("pdf_index_ggh","Index of Pdf which is active"); # name of category index should stay same across subCategories
    
    # // Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # // 0 == BWZ_Redux
    # // 1 == sumExp
    # // 2 == PowerSum
    
    # FEWZxBern Sumexp is less dependent to dimuon mass as stated in line 1585 of RERECO AN
    # I suppose BWZredux is there bc it's the one function with overall least bias (which is why BWZredux is used if CORE-PDF is not used)
    pdf_list_subCat1 = rt.RooArgList(
        model_subCat1_sumExp,
        model_subCat1_BWZRedux,
        model_subCat1_FEWZxBern,
    )
    corePdf_subCat1 = rt.RooMultiPdf("CorePdf_subCat1","CorePdf_subCat1",cat_subCat1,pdf_list_subCat1)
    penalty = 0 # as told in https://cms-talk.web.cern.ch/t/combine-fitting-not-working-with-roomultipdf-leading-to-bad-signal-significance/44238/
    corePdf_subCat1.setCorrectionFactor(penalty) 
    nevents = roo_datasetData_subCat1.sumEntries() # these are data, so all weights are one, thus no need to sum over the weights, though ofc you can just do that too
    print(f"roo_datasetData_subCat1 sumentries: {nevents}")
    bkg_subCat1_norm = rt.RooRealVar(corePdf_subCat1.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value
    
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat1"],
        "dataset": ["data"], 
        "yield": [nevents]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    
    
    # subCat 2 
    cat_subCat2 = rt.RooCategory("pdf_index_ggh","Index of Pdf which is active"); # name of category index should stay same across subCategories
    
    # // Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # // 0 == BWZ_Redux
    # // 1 == sumExp
    # // 2 == PowerSum
    
    # FEWZxBern Sumexp is less dependent to dimuon mass as stated in line 1585 of RERECO AN
    # I suppose BWZredux is there bc it's the one function with overall least bias (which is why BWZredux is used if CORE-PDF is not used)
    pdf_list_subCat2 = rt.RooArgList(
        model_subCat2_sumExp,
        model_subCat2_BWZRedux,
        model_subCat2_FEWZxBern,
    )
    corePdf_subCat2 = rt.RooMultiPdf("CorePdf_subCat2","CorePdf_subCat2",cat_subCat2,pdf_list_subCat2)
    penalty = 0 # as told in https://cms-talk.web.cern.ch/t/combine-fitting-not-working-with-roomultipdf-leading-to-bad-signal-significance/44238/
    corePdf_subCat2.setCorrectionFactor(penalty) 
    nevents = roo_datasetData_subCat2.sumEntries() # these are data, so all weights are one, thus no need to sum over the weights, though ofc you can just do that too
    print(f"roo_datasetData_subCat2 sumentries: {nevents}")
    bkg_subCat2_norm = rt.RooRealVar(corePdf_subCat2.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat2"],
        "dataset": ["data"], 
        "yield": [nevents]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    
        
    # subCat 3 
    cat_subCat3 = rt.RooCategory("pdf_index_ggh","Index of Pdf which is active"); # name of category index should stay same across subCategories
    
    # // Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # // 0 == BWZ_Redux
    # // 1 == sumExp
    # // 2 == PowerSum
    
    # FEWZxBern Sumexp is less dependent to dimuon mass as stated in line 1585 of RERECO AN
    # I suppose BWZredux is there bc it's the one function with overall least bias (which is why BWZredux is used if CORE-PDF is not used)
    pdf_list_subCat3 = rt.RooArgList(
        model_subCat3_sumExp,
        model_subCat3_BWZRedux,
        model_subCat3_FEWZxBern,
    )
    corePdf_subCat3 = rt.RooMultiPdf("CorePdf_subCat3","CorePdf_subCat3",cat_subCat3,pdf_list_subCat3)
    penalty = 0 # as told in https://cms-talk.web.cern.ch/t/combine-fitting-not-working-with-roomultipdf-leading-to-bad-signal-significance/44238/
    corePdf_subCat3.setCorrectionFactor(penalty) 
    nevents = roo_datasetData_subCat3.sumEntries() # these are data, so all weights are one, thus no need to sum over the weights, though ofc you can just do that too
    print(f"roo_datasetData_subCat3 sumentries: {nevents}")
    bkg_subCat3_norm = rt.RooRealVar(corePdf_subCat3.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat3"],
        "dataset": ["data"], 
        "yield": [nevents]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    

    # subCat 4
    cat_subCat4 = rt.RooCategory("pdf_index_ggh","Index of Pdf which is active"); # name of category index should stay same across subCategories
    
    # // Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # // 0 == sumExp
    # // 1 == BWZ_Redux
    # // 2 == FEWZxBern
    
    # FEWZxBern Sumexp is less dependent to dimuon mass as stated in line 1585 of RERECO AN
    # I suppose BWZredux is there bc it's the one function with overall least bias (which is why BWZredux is used if CORE-PDF is not used)
    pdf_list_subCat4 = rt.RooArgList(
        model_subCat4_sumExp,
        model_subCat4_BWZRedux,
        model_subCat4_FEWZxBern,
    )
    corePdf_subCat4 = rt.RooMultiPdf("CorePdf_subCat4","CorePdf_subCat4",cat_subCat4,pdf_list_subCat4)
    penalty = 0 # as told in https://cms-talk.web.cern.ch/t/combine-fitting-not-working-with-roomultipdf-leading-to-bad-signal-significance/44238/
    corePdf_subCat4.setCorrectionFactor(penalty) 
    nevents = roo_datasetData_subCat4.sumEntries() # these are data, so all weights are one, thus no need to sum over the weights, though ofc you can just do that too
    print(f"roo_datasetData_subCat4 sumentries: {nevents}")
    bkg_subCat4_norm = rt.RooRealVar(corePdf_subCat4.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat4"],
        "dataset": ["data"], 
        "yield": [nevents]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    
    print(f"yield_df after Data: \n {yield_df}")


    # #----------------------------------------------------------------------------
    # # Get GoF of CORE-PDF
    # # ---------------------------------------------------------------------------

    # KS_df = pd.DataFrame()
    corePDF_subCats = [
        corePdf_subCat0,
        corePdf_subCat1,
        corePdf_subCat2,
        corePdf_subCat3,
        corePdf_subCat4,
    ]
    hist_datas = [
        roo_histData_subCat0,
        roo_histData_subCat1,
        roo_histData_subCat2,
        roo_histData_subCat3,
        roo_histData_subCat4,
    ]
    multi_pdf_cats = [
        cat_subCat0,
        cat_subCat1,
        cat_subCat2,
        cat_subCat3,
        cat_subCat4,
    ]
    pdf_cat_name_dict = {
        0: "sumExp",
        1: "BWZRedux",
        2: "FEWZxBern",
    }
    gof_save_path = f"{plot_save_path}/gof"
    os.makedirs(gof_save_path, exist_ok=True)
    gof_df = pd.DataFrame(columns=["pdf category", "region", "KS statistic", "nevents", "alpha", "pass threshold", "test pass"])
    for i in range(len(corePDF_subCats)):
        hist_data = hist_datas[i]
        corePDF_subCat = corePDF_subCats[i]
        multi_pdf_cat = multi_pdf_cats[i]
        for cat_ix in range(len(pdf_cat_name_dict.keys())):
            multi_pdf_cat.setIndex(cat_ix)
            print(f"multi_pdf_cat.getIndex(): {multi_pdf_cat.getIndex()}")
            core_func_name = pdf_cat_name_dict[cat_ix]
            gof_test_name = f"ggh_cat{i}_{core_func_name}"
            KS_dict = getGOF_KS(mass, hist_data, corePDF_subCat, gof_test_name, gof_save_path)
            for region, ks_stat_dict in KS_dict.items():
                nevents = ks_stat_dict["nevents"]
                ks_stat = ks_stat_dict["ks_statistic"]
                # alpha = 0.05
                # pass_threshold = 1.358 / (nevents**(0.5))
                alpha = 0.1
                pass_threshold = 1.22385 / (nevents**(0.5))
                
                gof_df.loc[len(gof_df)] = {
                    "pdf category": gof_test_name,
                    "region": region,
                    "KS statistic": ks_stat,
                    "nevents": nevents,
                    "alpha": alpha,
                    "pass threshold": pass_threshold,
                    "test pass": ks_stat<pass_threshold,
                }
    gof_df.to_csv(f"{gof_save_path}/KS_stats.csv")

        
    raise ValueError

    # #----------------------------------------------------------------------------
    # # Now do multi-Pdf 
    # # ---------------------------------------------------------------------------
     
    # # Define category to distinguish physics and control samples events
    # sample = rt.RooCategory("sample", "sample")
    # sample.defineType("subCat0_BWZRedux")
    # sample.defineType("subCat1_BWZRedux")
    # sample.defineType("subCat2_BWZRedux")
    # sample.defineType("subCat3_BWZRedux")
    # sample.defineType("subCat4_BWZRedux")

     
    # # Construct combined dataset in (x,sample)
    # combData = rt.RooDataSet(
    #     "combData",
    #     "combined data",
    #     {mass},
    #     Index=sample,
    #     Import={
    #         "subCat0_BWZRedux": data_subCat0_BWZRedux, 
    #         "subCat1_BWZRedux": data_subCat1_BWZRedux,
    #         "subCat2_BWZRedux": data_subCat2_BWZRedux,
    #         "subCat3_BWZRedux": data_subCat3_BWZRedux,
    #         "subCat4_BWZRedux": data_subCat4_BWZRedux,
    #     },
    # )
    # # ---------------------------------------------------
    # # Construct a simultaneous pdf in (x, sample)
    # # -----------------------------------------------------------------------------------
     
    # simPdf = rt.RooSimultaneous(
    #                             "simPdf", 
    #                             "simultaneous pdf", 
    #                             {
    #                                 "subCat0_BWZRedux": corePdf_subCat0, 
    #                                 "subCat1_BWZRedux": corePdf_subCat1,
    #                                 "subCat2_BWZRedux": corePdf_subCat2,
    #                                 "subCat3_BWZRedux": corePdf_subCat3,
    #                                 "subCat4_BWZRedux": corePdf_subCat4,
    #                             }, 
    #                             sample,
    # )
    # # ---------------------------------------------------
    # # Perform a simultaneous fit
    # # ---------------------------------------------------
    # fitResult = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,SumW2Error=True)
    # fitResult.Print()
    # raise ValueError
    

    # ---------------------------------------------------
    # Obtain signal MC events
    # ---------------------------------------------------

    # load_path = f"{args.load_path}/{category}/{args.year}/processed_events_signalMC.parquet"
    if args.year=="all":
        load_path = f"{args.load_path}/{category}/*/processed_events_sigMC_ggh.parquet"
        # load_path = f"{args.load_path}/{category}/*/processed_events_sigMC_ggh_amcPS.parquet"
    elif args.year=="2016only":
        load_path = f"{args.load_path}/{category}/2016*/processed_events_sigMC_ggh.parquet"
    else:
        load_path = f"{args.load_path}/{category}/{args.year}/processed_events_sigMC_ggh.parquet" # Fig 6.15 was only with ggH process, though with all 2016, 2017 and 2018
    # load_path = f"{args.load_path}/{category}/{args.year}/processed_events_sigMC*.parquet"
    if args.year=="all":
        load_path = f"{args.load_path}/*/processed_events_sigMC_ggh.parquet"
        # load_path = f"{args.load_path}/{category}/*/processed_events_sigMC_ggh_amcPS.parquet"
    elif args.year=="2016only":
        load_path = f"{args.load_path}/2016*/processed_events_sigMC_ggh.parquet"
    else:
        load_path = f"{args.load_path}/{args.year}/processed_events_sigMC_ggh.parquet"
    processed_eventsSignalMC = ak.from_parquet(load_path)
    print(f"ggH yield: {np.sum(processed_eventsSignalMC.wgt_nominal)}")
    print("signal events loaded")
    
    # ---------------------------------------------------
    # Define signal model's Doubcl Crystal Ball PDF
    # ---------------------------------------------------
    
    # subCat 0
    # original start ------------------------------------------------------
    # MH_subCat0 = rt.RooRealVar("MH" , "MH", 125, 115,135)
    # MH_subCat0.setConstant(True) # this shouldn't change, I think
    # original end ------------------------------------------------------
    # MH_subCat0 = rt.RooRealVar("MH" , "MH", 124.805, 120,130) # matching AN
    # MH_subCat0 = rt.RooRealVar("MH" , "MH", 124.805, 124,126)
    MH_subCat0 = rt.RooRealVar("MH" , "MH", 125) # make this frozen
    
    # sigma_subCat0 = rt.RooRealVar("sigma_subCat0" , "sigma_subCat0", 2, .1, 4.0)
    # alpha1_subCat0 = rt.RooRealVar("alpha1_subCat0" , "alpha1_subCat0", 2, 0.01, 65)
    # n1_subCat0 = rt.RooRealVar("n1_subCat0" , "n1_subCat0", 10, 0.01, 100)
    # alpha2_subCat0 = rt.RooRealVar("alpha2_subCat0" , "alpha2_subCat0", 2.0, 0.01, 65)
    # n2_subCat0 = rt.RooRealVar("n2_subCat0" , "n2_subCat0", 25, 0.01, 100)

    # copying parameters from official AN workspace as starting params
    sigma_subCat0 = rt.RooRealVar("sigma_subCat0" , "sigma_subCat0", 1.8228, .1, 4.0)
    alpha1_subCat0 = rt.RooRealVar("alpha1_subCat0" , "alpha1_subCat0", 1.12842, 0.01, 65)
    n1_subCat0 = rt.RooRealVar("n1_subCat0" , "n1_subCat0", 4.019960, 0.01, 100)
    alpha2_subCat0 = rt.RooRealVar("alpha2_subCat0" , "alpha2_subCat0", 1.3132, 0.01, 65)
    n2_subCat0 = rt.RooRealVar("n2_subCat0" , "n2_subCat0", 9.97411, 0.01, 100)

    # # temporary test
    # sigma_subCat0.setConstant(True)
    # alpha1_subCat0.setConstant(True)
    # n1_subCat0.setConstant(True)
    # alpha2_subCat0.setConstant(True)
    # n2_subCat0.setConstant(True)
    
    
    CMS_hmm_sigma_cat0_ggh = rt.RooRealVar("CMS_hmm_sigma_cat0_ggh" , "CMS_hmm_sigma_cat0_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat0_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat0_ggh_fsigma = rt.RooFormulaVar("ggH_cat0_ggh_fsigma", "ggH_cat0_ggh_fsigma",'@0*(1+@1)',[sigma_subCat0, CMS_hmm_sigma_cat0_ggh])
    CMS_hmm_peak_cat0_ggh = rt.RooRealVar("CMS_hmm_peak_cat0_ggh" , "CMS_hmm_peak_cat0_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat0_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat0_ggh_fpeak = rt.RooFormulaVar("ggH_cat0_ggh_fpeak", "ggH_cat0_ggh_fpeak",'@0*(1+@1)',[MH_subCat0, CMS_hmm_peak_cat0_ggh])
    
    # n1_subCat0.setConstant(True) # freeze for stability
    # n2_subCat0.setConstant(True) # freeze for stability
    name = "signal_subCat0"
    signal_subCat0 = rt.RooCrystalBall(name,name,mass, ggH_cat0_ggh_fpeak, ggH_cat0_ggh_fsigma, alpha1_subCat0, n1_subCat0, alpha2_subCat0, n2_subCat0)

    # subCat 1
    # original start ------------------------------------------------------
    # MH_subCat1 = rt.RooRealVar("MH" , "MH", 125, 115,135)
    # MH_subCat1.setConstant(True) # this shouldn't change, I think
    # original end ------------------------------------------------------
    MH_subCat1 = MH_subCat0 
    
    # sigma_subCat1 = rt.RooRealVar("sigma_subCat1" , "sigma_subCat1", 2, .1, 4.0)
    # alpha1_subCat1 = rt.RooRealVar("alpha1_subCat1" , "alpha1_subCat1", 2, 0.01, 65)
    # n1_subCat1 = rt.RooRealVar("n1_subCat1" , "n1_subCat1", 10, 0.01, 100)
    # alpha2_subCat1 = rt.RooRealVar("alpha2_subCat1" , "alpha2_subCat1", 2.0, 0.01, 65)
    # n2_subCat1 = rt.RooRealVar("n2_subCat1" , "n2_subCat1", 25, 0.01, 100)

    # copying parameters from official AN workspace as starting params
    sigma_subCat1 = rt.RooRealVar("sigma_subCat1" , "sigma_subCat1", 1.503280, .1, 4.0)
    alpha1_subCat1 = rt.RooRealVar("alpha1_subCat1" , "alpha1_subCat1", 1.3364, 0.01, 65)
    n1_subCat1 = rt.RooRealVar("n1_subCat1" , "n1_subCat1", 2.815022, 0.01, 100)
    alpha2_subCat1 = rt.RooRealVar("alpha2_subCat1" , "alpha2_subCat1", 1.57127749, 0.01, 65)
    n2_subCat1 = rt.RooRealVar("n2_subCat1" , "n2_subCat1", 9.99687, 0.01, 100)

    # # temporary test
    # sigma_subCat1.setConstant(True)
    # alpha1_subCat1.setConstant(True)
    # n1_subCat1.setConstant(True)
    # alpha2_subCat1.setConstant(True)
    # n2_subCat1.setConstant(True)
    
    CMS_hmm_sigma_cat1_ggh = rt.RooRealVar("CMS_hmm_sigma_cat1_ggh" , "CMS_hmm_sigma_cat1_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat1_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat1_ggh_fsigma = rt.RooFormulaVar("ggH_cat1_ggh_fsigma", "ggH_cat1_ggh_fsigma",'@0*(1+@1)',[sigma_subCat1, CMS_hmm_sigma_cat1_ggh])
    CMS_hmm_peak_cat1_ggh = rt.RooRealVar("CMS_hmm_peak_cat1_ggh" , "CMS_hmm_peak_cat1_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat1_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat1_ggh_fpeak = rt.RooFormulaVar("ggH_cat1_ggh_fpeak", "ggH_cat1_ggh_fpeak",'@0*(1+@1)',[MH_subCat1, CMS_hmm_peak_cat1_ggh])
    
    # n1_subCat1.setConstant(True) # freeze for stability
    # n2_subCat1.setConstant(True) # freeze for stability
    name = "signal_subCat1"
    signal_subCat1 = rt.RooCrystalBall(name,name,mass, ggH_cat1_ggh_fpeak, ggH_cat1_ggh_fsigma, alpha1_subCat1, n1_subCat1, alpha2_subCat1, n2_subCat1)

    # subCat 2
    # original start ------------------------------------------------------
    # MH_subCat2 = rt.RooRealVar("MH" , "MH", 125, 115,135)
    # MH_subCat2.setConstant(True) # this shouldn't change, I think
    # original end ------------------------------------------------------
    MH_subCat2 = MH_subCat0 
    
    # sigma_subCat2 = rt.RooRealVar("sigma_subCat2" , "sigma_subCat2", 2, .1, 4.0)
    # alpha1_subCat2 = rt.RooRealVar("alpha1_subCat2" , "alpha1_subCat2", 2, 0.01, 65)
    # n1_subCat2 = rt.RooRealVar("n1_subCat2" , "n1_subCat2", 10, 0.01, 100)
    # alpha2_subCat2 = rt.RooRealVar("alpha2_subCat2" , "alpha2_subCat2", 2.0, 0.01, 65)
    # n2_subCat2 = rt.RooRealVar("n2_subCat2" , "n2_subCat2", 25, 0.01, 100)

    # copying parameters from official AN workspace as starting params
    sigma_subCat2 = rt.RooRealVar("sigma_subCat2" , "sigma_subCat2", 1.36025, .1, 4.0)
    alpha1_subCat2 = rt.RooRealVar("alpha1_subCat2" , "alpha1_subCat2", 1.4173626, 0.01, 65)
    n1_subCat2 = rt.RooRealVar("n1_subCat2" , "n1_subCat2", 2.42748, 0.01, 100)
    alpha2_subCat2 = rt.RooRealVar("alpha2_subCat2" , "alpha2_subCat2", 1.629120, 0.01, 65)
    n2_subCat2 = rt.RooRealVar("n2_subCat2" , "n2_subCat2", 9.983334, 0.01, 100)

    # # temporary test
    # sigma_subCat2.setConstant(True)
    # alpha1_subCat2.setConstant(True)
    # n1_subCat2.setConstant(True)
    # alpha2_subCat2.setConstant(True)
    # n2_subCat2.setConstant(True)

    CMS_hmm_sigma_cat2_ggh = rt.RooRealVar("CMS_hmm_sigma_cat2_ggh" , "CMS_hmm_sigma_cat2_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat2_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat2_ggh_fsigma = rt.RooFormulaVar("ggH_cat2_ggh_fsigma", "ggH_cat2_ggh_fsigma",'@0*(1+@1)',[sigma_subCat2, CMS_hmm_sigma_cat2_ggh])
    CMS_hmm_peak_cat2_ggh = rt.RooRealVar("CMS_hmm_peak_cat2_ggh" , "CMS_hmm_peak_cat2_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat2_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat2_ggh_fpeak = rt.RooFormulaVar("ggH_cat2_ggh_fpeak", "ggH_cat2_ggh_fpeak",'@0*(1+@1)',[MH_subCat2, CMS_hmm_peak_cat2_ggh])
    
    # n1_subCat2.setConstant(True) # freeze for stability
    # n2_subCat2.setConstant(True) # freeze for stability
    name = "signal_subCat2"
    signal_subCat2 = rt.RooCrystalBall(name,name,mass, ggH_cat2_ggh_fpeak, ggH_cat2_ggh_fsigma, alpha1_subCat2, n1_subCat2, alpha2_subCat2, n2_subCat2)

    # subCat 3
    # original start ------------------------------------------------------
    # MH_subCat3 = rt.RooRealVar("MH" , "MH", 125, 115,135)
    # MH_subCat3.setConstant(True) # this shouldn't change, I think
    # original end ------------------------------------------------------
    MH_subCat3 = MH_subCat0
    

    sigma_subCat3 = rt.RooRealVar("sigma_subCat3" , "sigma_subCat3", 0.1, .1, 10.0)
    alpha1_subCat3 = rt.RooRealVar("alpha1_subCat3" , "alpha1_subCat3", 2, 0.01, 200)
    n1_subCat3 = rt.RooRealVar("n1_subCat3" , "n1_subCat3", 25, 0.01, 200)
    alpha2_subCat3 = rt.RooRealVar("alpha2_subCat3" , "alpha2_subCat3", 2, 0.01, 65)
    n2_subCat3 = rt.RooRealVar("n2_subCat3" , "n2_subCat3", 25, 0.01, 200)

    # # copying parameters from official AN workspace as starting params
    # sigma_subCat3 = rt.RooRealVar("sigma_subCat3" , "sigma_subCat3", 1.25359, .1, 10.0)
    # alpha1_subCat3 = rt.RooRealVar("alpha1_subCat3" , "alpha1_subCat3", 1.4199, 0.01, 200)
    # n1_subCat3 = rt.RooRealVar("n1_subCat3" , "n1_subCat3", 2.409953, 0.01, 200)
    # alpha2_subCat3 = rt.RooRealVar("alpha2_subCat3" , "alpha2_subCat3", 1.64675, 0.01, 65)
    # n2_subCat3 = rt.RooRealVar("n2_subCat3" , "n2_subCat3", 9.670221, 0.01, 200)

    # # temporary test
    # sigma_subCat3.setConstant(True)
    # alpha1_subCat3.setConstant(True)
    # n1_subCat3.setConstant(True)
    # alpha2_subCat3.setConstant(True)
    # n2_subCat3.setConstant(True)

    CMS_hmm_sigma_cat3_ggh = rt.RooRealVar("CMS_hmm_sigma_cat3_ggh" , "CMS_hmm_sigma_cat3_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat3_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat3_ggh_fsigma = rt.RooFormulaVar("ggH_cat3_ggh_fsigma", "ggH_cat3_ggh_fsigma",'@0*(1+@1)',[sigma_subCat3, CMS_hmm_sigma_cat3_ggh])
    CMS_hmm_peak_cat3_ggh = rt.RooRealVar("CMS_hmm_peak_cat3_ggh" , "CMS_hmm_peak_cat3_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat3_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat3_ggh_fpeak = rt.RooFormulaVar("ggH_cat3_ggh_fpeak", "ggH_cat3_ggh_fpeak",'@0*(1+@1)',[MH_subCat3, CMS_hmm_peak_cat3_ggh])
    
    # n1_subCat3.setConstant(True) # freeze for stability
    # n2_subCat3.setConstant(True) # freeze for stability
    name = "signal_subCat3"
    signal_subCat3 = rt.RooCrystalBall(name,name,mass, ggH_cat3_ggh_fpeak, ggH_cat3_ggh_fsigma, alpha1_subCat3, n1_subCat3, alpha2_subCat3, n2_subCat3)

    # subCat 4
    # original start ------------------------------------------------------
    # MH_subCat4 = rt.RooRealVar("MH" , "MH", 125, 115,135)
    # MH_subCat4.setConstant(True) # this shouldn't change, I think
    # original end ------------------------------------------------------
    MH_subCat4 = MH_subCat0
    
    # sigma_subCat4 = rt.RooRealVar("sigma_subCat4" , "sigma_subCat4", 2, .1, 4.0)
    # alpha1_subCat4 = rt.RooRealVar("alpha1_subCat4" , "alpha1_subCat4", 2, 0.01, 65)
    # n1_subCat4 = rt.RooRealVar("n1_subCat4" , "n1_subCat4", 10, 0.01, 100)
    # alpha2_subCat4 = rt.RooRealVar("alpha2_subCat4" , "alpha2_subCat4", 2.0, 0.01, 65)
    # n2_subCat4 = rt.RooRealVar("n2_subCat4" , "n2_subCat4", 25, 0.01, 100)

    # copying parameters from official AN workspace as starting params
    sigma_subCat4 = rt.RooRealVar("sigma_subCat4" , "sigma_subCat4", 1.28250, .1, 4.0)
    alpha1_subCat4 = rt.RooRealVar("alpha1_subCat4" , "alpha1_subCat4", 1.47936, 0.01, 65)
    n1_subCat4 = rt.RooRealVar("n1_subCat4" , "n1_subCat4", 2.24104, 0.01, 100)
    alpha2_subCat4 = rt.RooRealVar("alpha2_subCat4" , "alpha2_subCat4", 1.67898, 0.01, 65)
    n2_subCat4 = rt.RooRealVar("n2_subCat4" , "n2_subCat4", 8.8719, 0.01, 100)

    # # temporary test
    # sigma_subCat4.setConstant(True)
    # alpha1_subCat4.setConstant(True)
    # n1_subCat4.setConstant(True)
    # alpha2_subCat4.setConstant(True)
    # n2_subCat4.setConstant(True)

    CMS_hmm_sigma_cat4_ggh = rt.RooRealVar("CMS_hmm_sigma_cat4_ggh" , "CMS_hmm_sigma_cat4_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat4_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat4_ggh_fsigma = rt.RooFormulaVar("ggH_cat4_ggh_fsigma", "ggH_cat4_ggh_fsigma",'@0*(1+@1)',[sigma_subCat4, CMS_hmm_sigma_cat4_ggh])
    CMS_hmm_peak_cat4_ggh = rt.RooRealVar("CMS_hmm_peak_cat4_ggh" , "CMS_hmm_peak_cat4_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat4_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat4_ggh_fpeak = rt.RooFormulaVar("ggH_cat4_ggh_fpeak", "ggH_cat4_ggh_fpeak",'@0*(1+@1)',[MH_subCat4, CMS_hmm_peak_cat4_ggh])
    
    # n1_subCat4.setConstant(True) # freeze for stability
    # n2_subCat4.setConstant(True) # freeze for stability
    name = "signal_subCat4"
    signal_subCat4 = rt.RooCrystalBall(name,name,mass, ggH_cat4_ggh_fpeak, ggH_cat4_ggh_fsigma, alpha1_subCat4, n1_subCat4, alpha2_subCat4, n2_subCat4)
    
    
    # ---------------------------------------------------
    # Define signal MC samples to fit to for ggH
    # ---------------------------------------------------

    # subCat 0
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 0)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat0_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal[subCat_filter]
    ) # weights

    # generate a weighted histogram 
    roo_histData_subCat0_signal = rt.TH1F("subCat0_rooHist_signal", "subCat0_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat0_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat0_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat0_signal = rt.RooDataHist("subCat0_rooHist_signal", "subCat0_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat0_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat0_signal = roo_histData_subCat0_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat0"],
        "dataset": ["ggH"], 
        "yield": [data_subCat0_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    flat_MC_SF = 1.00
    # flat_MC_SF = 0.92 # temporary flat SF to match my Data/MC agreement to that of AN's
    norm_val = np.sum(wgt_subCat0_SigMC)* flat_MC_SF 
    # norm_val = 254.528077 # quick test
    sig_norm_subCat0 = rt.RooRealVar(signal_subCat0.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat0 norm_val: {norm_val}")
    sig_norm_subCat0.setConstant(True)

    # subCat 1
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 1)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat1_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat1_signal = rt.TH1F("subCat1_rooHist_signal", "subCat1_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat1_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat1_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat1_signal = rt.RooDataHist("subCat1_rooHist_signal", "subCat1_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat1_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat1_signal = roo_histData_subCat1_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat1"],
        "dataset": ["ggH"], 
        "yield": [data_subCat1_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat1_SigMC)* flat_MC_SF
    # norm_val = 295.214 # quick test
    sig_norm_subCat1 = rt.RooRealVar(signal_subCat1.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat1 norm_val: {norm_val}")
    sig_norm_subCat1.setConstant(True)

    # subCat 2
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 2)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat2_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat2_signal = rt.TH1F("subCat2_rooHist_signal", "subCat2_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat2_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat2_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat2_signal = rt.RooDataHist("subCat2_rooHist_signal", "subCat2_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat2_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat2_signal = roo_histData_subCat2_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat2"],
        "dataset": ["ggH"], 
        "yield": [data_subCat2_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat2_SigMC) * flat_MC_SF
    # norm_val = 124.0364 # quick test
    sig_norm_subCat2 = rt.RooRealVar(signal_subCat2.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat2 norm_val: {norm_val}")
    sig_norm_subCat2.setConstant(True)

    # subCat 3
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 3)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat3_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat3_signal = rt.TH1F("subCat3_rooHist_signal", "subCat3_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat3_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat3_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat3_signal = rt.RooDataHist("subCat3_rooHist_signal", "subCat3_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat3_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat3_signal = roo_histData_subCat3_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat3"],
        "dataset": ["ggH"], 
        "yield": [data_subCat3_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat3_SigMC)* flat_MC_SF
    # norm_val = 116.4918 # quick test
    sig_norm_subCat3 = rt.RooRealVar(signal_subCat3.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat3 norm_val: {norm_val}")
    sig_norm_subCat3.setConstant(True)
    
    # subCat 4
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 4)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat4_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat4_signal = rt.TH1F("subCat4_rooHist_signal", "subCat4_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat4_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat4_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat4_signal = rt.RooDataHist("subCat4_rooHist_signal", "subCat4_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat4_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat4_signal = roo_histData_subCat4_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat4"],
        "dataset": ["ggH"], 
        "yield": [data_subCat4_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    print(f"yield_df after ggH: {yield_df}")

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat4_SigMC)* flat_MC_SF
    # norm_val = 45.423052 # quick test
    sig_norm_subCat4 = rt.RooRealVar(signal_subCat4.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat4 norm_val: {norm_val}")
    sig_norm_subCat4.setConstant(True)
    
    # ---------------------------------------------------
    # Fit signal model simultaneously. Sigma, and left and right tails are different for each category
    # ---------------------------------------------------

    

    # subCat 0
    _ = signal_subCat0.fitTo(data_subCat0_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat0.fitTo(data_subCat0_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat0.setConstant(True)
    alpha1_subCat0.setConstant(True)
    n1_subCat0.setConstant(True)
    alpha2_subCat0.setConstant(True)
    n2_subCat0.setConstant(True)

    

    # subCat 1
    _ = signal_subCat1.fitTo(data_subCat1_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat1.fitTo(data_subCat1_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat1.setConstant(True)
    alpha1_subCat1.setConstant(True)
    n1_subCat1.setConstant(True)
    alpha2_subCat1.setConstant(True)
    n2_subCat1.setConstant(True)

    

    # subCat 2
    _ = signal_subCat2.fitTo(data_subCat2_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat2.fitTo(data_subCat2_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat2.setConstant(True)
    alpha1_subCat2.setConstant(True)
    n1_subCat2.setConstant(True)
    alpha2_subCat2.setConstant(True)
    n2_subCat2.setConstant(True)

    
    
    # subCat 3
    _ = signal_subCat3.fitTo(data_subCat3_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat3.fitTo(data_subCat3_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat3.setConstant(True)
    alpha1_subCat3.setConstant(True)
    n1_subCat3.setConstant(True)
    alpha2_subCat3.setConstant(True)
    n2_subCat3.setConstant(True)
    # sigma_subCat3.setConstant(False)
    # alpha1_subCat3.setConstant(False)
    # n1_subCat3.setConstant(False)
    # alpha2_subCat3.setConstant(False)
    # n2_subCat3.setConstant(False)


    # subCat 4
    _ = signal_subCat4.fitTo(data_subCat4_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat4.fitTo(data_subCat4_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat4.setConstant(True)
    alpha1_subCat4.setConstant(True)
    n1_subCat4.setConstant(True)
    alpha2_subCat4.setConstant(True)
    n2_subCat4.setConstant(True)

    # ---------------------------------------------------
    # Obtain signal MC events for VBF
    # ---------------------------------------------------

    # load_path = f"{args.load_path}/{category}/{args.year}/processed_events_signalMC.parquet"
    if args.year=="all":
        load_path = f"{args.load_path}/*/processed_events_sigMC_vbf.parquet"
        # load_path = f"{args.load_path}/{category}/*/processed_events_sigMC_qqh_amcPS.parquet"
    elif args.year=="2016only":
        load_path = f"{args.load_path}/2016*/processed_events_sigMC_vbf.parquet"
    else:
        load_path = f"{args.load_path}/{args.year}/processed_events_sigMC_vbf.parquet" # Fig 6.15 was only with qqH process, though with all 2016, 2017 and 2018
    
    processed_eventsSignalMC_vbf = ak.from_parquet(load_path)
    print(f"qqH yield: {np.sum(processed_eventsSignalMC_vbf.wgt_nominal)}")
    print("signal events loaded")
    
    # ---------------------------------------------------
    # Define vbf signal model's Doubcl Crystal Ball PDF
    # ---------------------------------------------------
    
    # subCat 0
    
    sigma_subCat0_vbf = rt.RooRealVar("sigma_subCat0_vbf" , "sigma_subCat0_vbf", 2, .1, 4.0)
    alpha1_subCat0_vbf = rt.RooRealVar("alpha1_subCat0_vbf" , "alpha1_subCat0_vbf", 2, 0.01, 65)
    n1_subCat0_vbf = rt.RooRealVar("n1_subCat0_vbf" , "n1_subCat0_vbf", 10, 0.01, 100)
    alpha2_subCat0_vbf = rt.RooRealVar("alpha2_subCat0_vbf" , "alpha2_subCat0_vbf", 2.0, 0.01, 65)
    n2_subCat0_vbf = rt.RooRealVar("n2_subCat0_vbf" , "n2_subCat0_vbf", 25, 0.01, 100)

    # # temporary test
    # sigma_subCat0_vbf.setConstant(True)
    # alpha1_subCat0_vbf.setConstant(True)
    # n1_subCat0_vbf.setConstant(True)
    # alpha2_subCat0_vbf.setConstant(True)
    # n2_subCat0_vbf.setConstant(True)
    

    qqH_cat0_ggh_fsigma = rt.RooFormulaVar("qqH_cat0_ggh_fsigma", "qqH_cat0_ggh_fsigma",'@0*(1+@1)',[sigma_subCat0_vbf, CMS_hmm_sigma_cat0_ggh])
    qqH_cat0_ggh_fpeak = rt.RooFormulaVar("qqH_cat0_qqh_fpeak", "qqH_cat0_ggh_fpeak",'@0*(1+@1)',[MH_subCat0, CMS_hmm_peak_cat0_ggh])
    
    # n1_subCat0_vbf.setConstant(True) # freeze for stability
    # n2_subCat0_vbf.setConstant(True) # freeze for stability
    name = "signal_subCat0_vbf"
    signal_subCat0_vbf = rt.RooCrystalBall(name,name,mass, qqH_cat0_ggh_fpeak, qqH_cat0_ggh_fsigma, alpha1_subCat0_vbf, n1_subCat0_vbf, alpha2_subCat0_vbf, n2_subCat0_vbf)

    # subCat 1

    
    sigma_subCat1_vbf = rt.RooRealVar("sigma_subCat1_vbf" , "sigma_subCat1_vbf", 2, .1, 4.0)
    alpha1_subCat1_vbf = rt.RooRealVar("alpha1_subCat1_vbf" , "alpha1_subCat1_vbf", 2, 0.01, 65)
    n1_subCat1_vbf = rt.RooRealVar("n1_subCat1_vbf" , "n1_subCat1_vbf", 10, 0.01, 100)
    alpha2_subCat1_vbf = rt.RooRealVar("alpha2_subCat1_vbf" , "alpha2_subCat1_vbf", 2.0, 0.01, 65)
    n2_subCat1_vbf = rt.RooRealVar("n2_subCat1_vbf" , "n2_subCat1_vbf", 25, 0.01, 100)

    # # temporary test
    # sigma_subCat1_vbf.setConstant(True)
    # alpha1_subCat1_vbf.setConstant(True)
    # n1_subCat1_vbf.setConstant(True)
    # alpha2_subCat1_vbf.setConstant(True)
    # n2_subCat1_vbf.setConstant(True)
    
    qqH_cat1_ggh_fsigma = rt.RooFormulaVar("qqH_cat1_ggh_fsigma", "qqH_cat1_ggh_fsigma",'@0*(1+@1)',[sigma_subCat1_vbf, CMS_hmm_sigma_cat1_ggh])
    qqH_cat1_ggh_fpeak = rt.RooFormulaVar("qqH_cat1_ggh_fpeak", "qqH_cat1_ggh_fpeak",'@0*(1+@1)',[MH_subCat1, CMS_hmm_peak_cat1_ggh])
    
    # n1_subCat1_vbf.setConstant(True) # freeze for stability
    # n2_subCat1_vbf.setConstant(True) # freeze for stability
    name = "signal_subCat1_vbf"
    signal_subCat1_vbf = rt.RooCrystalBall(name,name,mass, qqH_cat1_ggh_fpeak, qqH_cat1_ggh_fsigma, alpha1_subCat1_vbf, n1_subCat1_vbf, alpha2_subCat1_vbf, n2_subCat1_vbf)

    # subCat 2
   
    sigma_subCat2_vbf = rt.RooRealVar("sigma_subCat2_vbf" , "sigma_subCat2_vbf", 2, .1, 4.0)
    alpha1_subCat2_vbf = rt.RooRealVar("alpha1_subCat2_vbf" , "alpha1_subCat2_vbf", 2, 0.01, 65)
    n1_subCat2_vbf = rt.RooRealVar("n1_subCat2_vbf" , "n1_subCat2_vbf", 10, 0.01, 100)
    alpha2_subCat2_vbf = rt.RooRealVar("alpha2_subCat2_vbf" , "alpha2_subCat2_vbf", 2.0, 0.01, 65)
    n2_subCat2_vbf = rt.RooRealVar("n2_subCat2_vbf" , "n2_subCat2_vbf", 25, 0.01, 100)

    # # temporary test
    # sigma_subCat2_vbf.setConstant(True)
    # alpha1_subCat2_vbf.setConstant(True)
    # n1_subCat2_vbf.setConstant(True)
    # alpha2_subCat2_vbf.setConstant(True)
    # n2_subCat2_vbf.setConstant(True)

    qqH_cat2_ggh_fsigma = rt.RooFormulaVar("qqH_cat2_ggh_fsigma", "qqH_cat2_ggh_fsigma",'@0*(1+@1)',[sigma_subCat2_vbf, CMS_hmm_sigma_cat2_ggh])
    qqH_cat2_ggh_fpeak = rt.RooFormulaVar("qqH_cat2_ggh_fpeak", "qqH_cat2_ggh_fpeak",'@0*(1+@1)',[MH_subCat2, CMS_hmm_peak_cat2_ggh])
    
    # n1_subCat2_vbf.setConstant(True) # freeze for stability
    # n2_subCat2_vbf.setConstant(True) # freeze for stability
    name = "signal_subCat2_vbf"
    signal_subCat2_vbf = rt.RooCrystalBall(name,name,mass, qqH_cat2_ggh_fpeak, qqH_cat2_ggh_fsigma, alpha1_subCat2_vbf, n1_subCat2_vbf, alpha2_subCat2_vbf, n2_subCat2_vbf)

    # subCat 3

    sigma_subCat3_vbf = rt.RooRealVar("sigma_subCat3_vbf" , "sigma_subCat3_vbf", 0.1, .1, 10.0)
    alpha1_subCat3_vbf = rt.RooRealVar("alpha1_subCat3_vbf" , "alpha1_subCat3_vbf", 2, 0.01, 200)
    n1_subCat3_vbf = rt.RooRealVar("n1_subCat3_vbf" , "n1_subCat3_vbf", 25, 0.01, 200)
    alpha2_subCat3_vbf = rt.RooRealVar("alpha2_subCat3_vbf" , "alpha2_subCat3_vbf", 2, 0.01, 65)
    n2_subCat3_vbf = rt.RooRealVar("n2_subCat3_vbf" , "n2_subCat3_vbf", 25, 0.01, 200)


    # # temporary test
    # sigma_subCat3_vbf.setConstant(True)
    # alpha1_subCat3_vbf.setConstant(True)
    # n1_subCat3_vbf.setConstant(True)
    # alpha2_subCat3_vbf.setConstant(True)
    # n2_subCat3_vbf.setConstant(True)

    qqH_cat3_ggh_fsigma = rt.RooFormulaVar("qqH_cat3_ggh_fsigma", "qqH_cat3_ggh_fsigma",'@0*(1+@1)',[sigma_subCat3_vbf, CMS_hmm_sigma_cat3_ggh])
    qqH_cat3_ggh_fpeak = rt.RooFormulaVar("qqH_cat3_ggh_fpeak", "qqH_cat3_ggh_fpeak",'@0*(1+@1)',[MH_subCat3, CMS_hmm_peak_cat3_ggh])
    
    # n1_subCat3_vbf.setConstant(True) # freeze for stability
    # n2_subCat3_vbf.setConstant(True) # freeze for stability
    name = "signal_subCat3_vbf"
    signal_subCat3_vbf = rt.RooCrystalBall(name,name,mass, qqH_cat3_ggh_fpeak, qqH_cat3_ggh_fsigma, alpha1_subCat3_vbf, n1_subCat3_vbf, alpha2_subCat3_vbf, n2_subCat3_vbf)

    # subCat 4
    
    sigma_subCat4_vbf = rt.RooRealVar("sigma_subCat4_vbf" , "sigma_subCat4_vbf", 2, .1, 4.0)
    alpha1_subCat4_vbf = rt.RooRealVar("alpha1_subCat4_vbf" , "alpha1_subCat4_vbf", 2, 0.01, 65)
    n1_subCat4_vbf = rt.RooRealVar("n1_subCat4_vbf" , "n1_subCat4_vbf", 10, 0.01, 100)
    alpha2_subCat4_vbf = rt.RooRealVar("alpha2_subCat4_vbf" , "alpha2_subCat4_vbf", 2.0, 0.01, 65)
    n2_subCat4_vbf = rt.RooRealVar("n2_subCat4_vbf" , "n2_subCat4_vbf", 25, 0.01, 100)


    # # temporary test
    # sigma_subCat4_vbf.setConstant(True)
    # alpha1_subCat4_vbf.setConstant(True)
    # n1_subCat4_vbf.setConstant(True)
    # alpha2_subCat4_vbf.setConstant(True)
    # n2_subCat4_vbf.setConstant(True)

    qqH_cat4_ggh_fsigma = rt.RooFormulaVar("qqH_cat4_ggh_fsigma", "qqH_cat4_ggh_fsigma",'@0*(1+@1)',[sigma_subCat4_vbf, CMS_hmm_sigma_cat4_ggh])
    qqH_cat4_ggh_fpeak = rt.RooFormulaVar("qqH_cat4_ggh_fpeak", "qqH_cat4_ggh_fpeak",'@0*(1+@1)',[MH_subCat4, CMS_hmm_peak_cat4_ggh])
    
    # n1_subCat4_vbf.setConstant(True) # freeze for stability
    # n2_subCat4_vbf.setConstant(True) # freeze for stability
    name = "signal_subCat4_vbf"
    signal_subCat4_vbf = rt.RooCrystalBall(name,name,mass, qqH_cat4_ggh_fpeak, qqH_cat4_ggh_fsigma, alpha1_subCat4_vbf, n1_subCat4_vbf, alpha2_subCat4_vbf, n2_subCat4_vbf)
    
    
    # ---------------------------------------------------
    # Define signal MC samples to fit to for qqH
    # ---------------------------------------------------

    # subCat 0
    subCat_filter = (processed_eventsSignalMC_vbf[subCatIdx_name] == 0)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC_vbf.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat0_vbf_SigMC = ak.to_numpy(
        processed_eventsSignalMC_vbf.wgt_nominal[subCat_filter]
    ) # weights

    # generate a weighted histogram 
    roo_histData_subCat0_vbf_signal = rt.TH1F("subCat0_vbf_rooHist_signal", "subCat0_vbf_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat0_vbf_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat0_vbf_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat0_vbf_signal = rt.RooDataHist("subCat0_vbf_rooHist_signal", "subCat0_vbf_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat0_vbf_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat0_vbf_signal = roo_histData_subCat0_vbf_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat0"],
        "dataset": ["VBF"], 
        "yield": [data_subCat0_vbf_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    flat_MC_SF = 1.00
    # flat_MC_SF = 0.92 # temporary flat SF to match my Data/MC agreement to that of AN's
    norm_val = np.sum(wgt_subCat0_vbf_SigMC)* flat_MC_SF 
    # norm_val = 254.528077 # quick test
    sig_norm_subCat0_vbf = rt.RooRealVar(signal_subCat0_vbf.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat0_vbf norm_val: {norm_val}")
    sig_norm_subCat0_vbf.setConstant(True)

    # subCat 1
    subCat_filter = (processed_eventsSignalMC_vbf[subCatIdx_name] == 1)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC_vbf.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat1_vbf_SigMC = ak.to_numpy(
        processed_eventsSignalMC_vbf.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat1_vbf_signal = rt.TH1F("subCat1_vbf_rooHist_signal", "subCat1_vbf_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat1_vbf_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat1_vbf_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat1_vbf_signal = rt.RooDataHist("subCat1_vbf_rooHist_signal", "subCat1_vbf_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat1_vbf_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat1_vbf_signal = roo_histData_subCat1_vbf_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat1"],
        "dataset": ["VBF"], 
        "yield": [data_subCat1_vbf_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat1_vbf_SigMC)* flat_MC_SF
    # norm_val = 295.214 # quick test
    sig_norm_subCat1_vbf = rt.RooRealVar(signal_subCat1_vbf.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat1_vbf norm_val: {norm_val}")
    sig_norm_subCat1_vbf.setConstant(True)

    # subCat 2
    subCat_filter = (processed_eventsSignalMC_vbf[subCatIdx_name] == 2)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC_vbf.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat2_vbf_SigMC = ak.to_numpy(
        processed_eventsSignalMC_vbf.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat2_vbf_signal = rt.TH1F("subCat2_vbf_rooHist_signal", "subCat2_vbf_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat2_vbf_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat2_vbf_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat2_vbf_signal = rt.RooDataHist("subCat2_vbf_rooHist_signal", "subCat2_vbf_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat2_vbf_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat2_vbf_signal = roo_histData_subCat2_vbf_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat2"],
        "dataset": ["VBF"], 
        "yield": [data_subCat2_vbf_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat2_vbf_SigMC) * flat_MC_SF
    # norm_val = 124.0364 # quick test
    sig_norm_subCat2_vbf = rt.RooRealVar(signal_subCat2_vbf.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat2_vbf norm_val: {norm_val}")
    sig_norm_subCat2_vbf.setConstant(True)

    # subCat 3
    subCat_filter = (processed_eventsSignalMC_vbf[subCatIdx_name] == 3)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC_vbf.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat3_vbf_SigMC = ak.to_numpy(
        processed_eventsSignalMC_vbf.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat3_vbf_signal = rt.TH1F("subCat3_vbf_rooHist_signal", "subCat3_vbf_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat3_vbf_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat3_vbf_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat3_vbf_signal = rt.RooDataHist("subCat3_vbf_rooHist_signal", "subCat3_vbf_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat3_vbf_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat3_vbf_signal = roo_histData_subCat3_vbf_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat3"],
        "dataset": ["VBF"], 
        "yield": [data_subCat3_vbf_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat3_vbf_SigMC)* flat_MC_SF
    # norm_val = 116.4918 # quick test
    sig_norm_subCat3_vbf = rt.RooRealVar(signal_subCat3_vbf.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat3_vbf norm_val: {norm_val}")
    sig_norm_subCat3_vbf.setConstant(True)
    
    # subCat 4
    subCat_filter = (processed_eventsSignalMC_vbf[subCatIdx_name] == 4)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC_vbf.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat4_vbf_SigMC = ak.to_numpy(
        processed_eventsSignalMC_vbf.wgt_nominal[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat4_vbf_signal = rt.TH1F("subCat4_vbf_rooHist_signal", "subCat4_vbf_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat4_vbf_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat4_vbf_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat4_vbf_signal = rt.RooDataHist("subCat4_vbf_rooHist_signal", "subCat4_vbf_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat4_vbf_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat4_vbf_signal = roo_histData_subCat4_vbf_signal
    # add yield
    new_row = {
        "year": [args.year],
        "category": ["cat4"],
        "dataset": ["VBF"], 
        "yield": [data_subCat4_vbf_signal.sumEntries()]
    }
    new_row = pd.DataFrame(new_row)
    yield_df = pd.concat([yield_df, new_row], ignore_index=True)
    print(f"yield_df after VBF: \n {yield_df}")

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat4_vbf_SigMC)* flat_MC_SF
    sig_norm_subCat4_vbf = rt.RooRealVar(signal_subCat4_vbf.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat4_vbf norm_val: {norm_val}")
    sig_norm_subCat4_vbf.setConstant(True)
    
    # ---------------------------------------------------
    # Fit signal model individually, not simultaneous. Sigma, and left and right tails are different for each category
    # ---------------------------------------------------

    # subCat 0
    # _ = signal_subCat0_vbf.fitTo(data_subCat0_vbf_signal,  EvalBackend=device, Save=True, )
    # fit_result = signal_subCat0_vbf.fitTo(data_subCat0_vbf_signal,  EvalBackend=device, Save=True, )
    _ = signal_subCat0_vbf.fitTo(data_subCat0_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat0_vbf.fitTo(data_subCat0_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()


    # Freeze the MH parameters. Source: "Crucially, we need to freeze the fit parameters of the signal mode" https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/tutorial2023/parametric_exercise/#signal-modelling
    MH_subCat0.setConstant(True)
    

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat0_vbf.setConstant(True)
    alpha1_subCat0_vbf.setConstant(True)
    n1_subCat0_vbf.setConstant(True)
    alpha2_subCat0_vbf.setConstant(True)
    n2_subCat0_vbf.setConstant(True)


    # subCat 1
    # _ = signal_subCat1_vbf.fitTo(data_subCat1_vbf_signal,  EvalBackend=device, Save=True, )
    # fit_result = signal_subCat1_vbf.fitTo(data_subCat1_vbf_signal,  EvalBackend=device, Save=True, )
    _ = signal_subCat1_vbf.fitTo(data_subCat1_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat1_vbf.fitTo(data_subCat1_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat1_vbf.setConstant(True)
    alpha1_subCat1_vbf.setConstant(True)
    n1_subCat1_vbf.setConstant(True)
    alpha2_subCat1_vbf.setConstant(True)
    n2_subCat1_vbf.setConstant(True)



    # subCat 2
    # _ = signal_subCat2_vbf.fitTo(data_subCat2_vbf_signal,  EvalBackend=device, Save=True, )
    # fit_result = signal_subCat2_vbf.fitTo(data_subCat2_vbf_signal,  EvalBackend=device, Save=True, )
    _ = signal_subCat2_vbf.fitTo(data_subCat2_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat2_vbf.fitTo(data_subCat2_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat2_vbf.setConstant(True)
    alpha1_subCat2_vbf.setConstant(True)
    n1_subCat2_vbf.setConstant(True)
    alpha2_subCat2_vbf.setConstant(True)
    n2_subCat2_vbf.setConstant(True)


    
    # subCat 3
    # _ = signal_subCat3_vbf.fitTo(data_subCat3_vbf_signal,  EvalBackend=device, Save=True, )
    # fit_result = signal_subCat3_vbf.fitTo(data_subCat3_vbf_signal,  EvalBackend=device, Save=True, )
    _ = signal_subCat3_vbf.fitTo(data_subCat3_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat3_vbf.fitTo(data_subCat3_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat3_vbf.setConstant(True)
    alpha1_subCat3_vbf.setConstant(True)
    n1_subCat3_vbf.setConstant(True)
    alpha2_subCat3_vbf.setConstant(True)
    n2_subCat3_vbf.setConstant(True)
    # sigma_subCat3_vbf.setConstant(False)
    # alpha1_subCat3_vbf.setConstant(False)
    # n1_subCat3_vbf.setConstant(False)
    # alpha2_subCat3_vbf.setConstant(False)
    # n2_subCat3_vbf.setConstant(False)


    # subCat 4
    # _ = signal_subCat4_vbf.fitTo(data_subCat4_vbf_signal,  EvalBackend=device, Save=True, )
    # fit_result = signal_subCat4_vbf.fitTo(data_subCat4_vbf_signal,  EvalBackend=device, Save=True, )
    _ = signal_subCat4_vbf.fitTo(data_subCat4_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = signal_subCat4_vbf.fitTo(data_subCat4_vbf_signal,  EvalBackend=device, Save=True, SumW2Error=True)
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat4_vbf.setConstant(True)
    alpha1_subCat4_vbf.setConstant(True)
    n1_subCat4_vbf.setConstant(True)
    alpha2_subCat4_vbf.setConstant(True)
    n2_subCat4_vbf.setConstant(True)


    
        
    # -------------------------------------------------------------------------
    # Save yield_df
    # -------------------------------------------------------------------------
    summed_values = yield_df.groupby("dataset", as_index=False)["yield"].sum()
    summed_values["year"] = args.year
    summed_values["category"] = "combined"
    yield_df = pd.concat([yield_df, summed_values], ignore_index=True)
    # print(f"yield_df after all: \n {yield_df}")
    yield_df = yield_df.sort_values(by=["dataset", "category"], ascending=[False, True])
    yield_df.to_csv(f"{base_path}/yield_df.csv")



    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # do signal ggH plotting with fit and data
    # -------------------------------------------------------------------------
    
    # subCat 0
    print(f"data_subCat0_signal.sumEntries(): {data_subCat0_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat0_signal.GetName()
    data_subCat0_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat0.GetName()
    signal_subCat0.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
        
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat0.pdf")

    # subCat 1
    print(f"data_subCat1_signal.sumEntries(): {data_subCat1_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat1_signal.GetName()
    data_subCat1_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat1.GetName()
    signal_subCat1.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat1.pdf")

    # subCat 2
    print(f"data_subCat2_signal.sumEntries(): {data_subCat2_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat2_signal.GetName()
    data_subCat2_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat2.GetName()
    signal_subCat2.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat2.pdf")

    # subCat 3
    print(f"data_subCat3_signal.sumEntries(): {data_subCat3_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat3_signal.GetName()
    data_subCat3_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat3.GetName()
    signal_subCat3.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat3.pdf")

    # subCat 4
    print(f"data_subCat4_signal.sumEntries(): {data_subCat4_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat4_signal.GetName()
    data_subCat4_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat4.GetName()
    signal_subCat4.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat4.pdf")

    # -------------------------------------------------------------------------
    # do signal VBF plotting with fit and data
    # -------------------------------------------------------------------------
    
    # subCat 0
    print(f"data_subCat0_vbf_signal.sumEntries(): {data_subCat0_vbf_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat0_vbf_signal.GetName()
    data_subCat0_vbf_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat0_vbf.GetName()
    signal_subCat0_vbf.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat0_vbf.pdf")

    # subCat 1
    print(f"data_subCat1_vbf_signal.sumEntries(): {data_subCat1_vbf_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat1_vbf_signal.GetName()
    data_subCat1_vbf_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat1_vbf.GetName()
    signal_subCat1_vbf.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat1_vbf.pdf")

    # subCat 2
    print(f"data_subCat2_vbf_signal.sumEntries(): {data_subCat2_vbf_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat2_vbf_signal.GetName()
    data_subCat2_vbf_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat2_vbf.GetName()
    signal_subCat2_vbf.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat2_vbf.pdf")

    # subCat 3
    print(f"data_subCat3_vbf_signal.sumEntries(): {data_subCat3_vbf_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat3_vbf_signal.GetName()
    data_subCat3_vbf_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat3_vbf.GetName()
    signal_subCat3_vbf.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat3_vbf.pdf")

    # subCat 4
    print(f"data_subCat4_vbf_signal.sumEntries(): {data_subCat4_vbf_signal.sumEntries()}")
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat4_vbf_signal.GetName()
    data_subCat4_vbf_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat4_vbf.GetName()
    signal_subCat4_vbf.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat4_vbf.pdf")

    # ---------------------------------------------------
    # Save to Signal, Background and Data to Workspace
    # ---------------------------------------------------
    # workspace_path = "./workspaces"
    workspace_path = f"{base_path}/workspaces"
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)


    # unfreeze the hmm sigma and peak b4 saving
    CMS_hmm_sigma_cat0_ggh.setConstant(False)
    CMS_hmm_peak_cat0_ggh.setConstant(False)

    CMS_hmm_sigma_cat1_ggh.setConstant(False)
    CMS_hmm_peak_cat1_ggh.setConstant(False)
    
    CMS_hmm_sigma_cat2_ggh.setConstant(False)
    CMS_hmm_peak_cat2_ggh.setConstant(False)
    
    CMS_hmm_sigma_cat3_ggh.setConstant(False)
    CMS_hmm_peak_cat3_ggh.setConstant(False)
    
    CMS_hmm_sigma_cat4_ggh.setConstant(False)
    CMS_hmm_peak_cat4_ggh.setConstant(False)

    # ----------------------------------------------------------------
    # # freeze back the core function before saving to workspace
    # # BWZ redux
    # a_coeff.setConstant(True)
    # b_coeff.setConstant(True)
    # c_coeff.setConstant(True)
    
    # # sumExp
    # a1_coeff.setConstant(True)
    # a2_coeff.setConstant(True)
    # f_coeff.setConstant(True)

    # # FEWZxBern
    # c1.setConstant(True)
    # c2.setConstant(True)
    # c3.setConstant(True)
    # c4.setConstant(True)
    # ----------------------------------------------------------------

    # subCat 0 
    fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat0_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat0.SetName("data_cat0_ggh");
    corePdf_subCat0.SetName("bkg_cat0_ggh_pdf");
    bkg_subCat0_norm.SetName(corePdf_subCat0.GetName()+"_norm"); 
    # make norm for data
    nevents = roo_histData_subCat0.sumEntries()
    roo_histData_subCat0_norm = rt.RooRealVar(roo_histData_subCat0.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat0_norm);
    wout.Import(roo_histData_subCat0);
    wout.Import(cat_subCat0);
    wout.Import(bkg_subCat0_norm);
    wout.Import(corePdf_subCat0);
    # wout.Print();
    wout.Write();

    fout = rt.TFile(f"{workspace_path}/workspace_sig_cat0_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat0.SetName("ggH_cat0_ggh_pdf");
    roo_histData_subCat0_signal.SetName("data_ggH_cat0_ggh");
    sig_norm_subCat0.SetName(signal_subCat0.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat0);
    wout.Import(signal_subCat0); 
    wout.Import(roo_histData_subCat0_signal); 
    
    signal_subCat0_vbf.SetName("qqH_cat0_ggh_pdf");
    roo_histData_subCat0_vbf_signal.SetName("data_qqH_cat0_ggh");
    sig_norm_subCat0_vbf.SetName(signal_subCat0_vbf.GetName()+"_norm"); 
    wout.Import(signal_subCat0_vbf);
    wout.Import(roo_histData_subCat0_vbf_signal); 
    wout.Import(sig_norm_subCat0_vbf); 
    
    # wout.Print();
    wout.Write();
    

    # subCat 1 
    fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat1_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat1.SetName("data_cat1_ggh");
    corePdf_subCat1.SetName("bkg_cat1_ggh_pdf");
    bkg_subCat1_norm.SetName(corePdf_subCat1.GetName()+"_norm");
    # make norm for data
    nevents = roo_histData_subCat1.sumEntries()
    roo_histData_subCat1_norm = rt.RooRealVar(roo_histData_subCat1.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat1_norm);
    wout.Import(roo_histData_subCat1);
    wout.Import(cat_subCat1);
    wout.Import(bkg_subCat1_norm);
    wout.Import(corePdf_subCat1);
    # wout.Print();
    wout.Write();

    fout = rt.TFile(f"{workspace_path}/workspace_sig_cat1_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat1.SetName("ggH_cat1_ggh_pdf"); 
    roo_histData_subCat1_signal.SetName("data_ggH_cat1_ggh");
    sig_norm_subCat1.SetName(signal_subCat1.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat1);
    wout.Import(signal_subCat1); 
    wout.Import(roo_histData_subCat1_signal); 

    signal_subCat1_vbf.SetName("qqH_cat1_ggh_pdf"); 
    roo_histData_subCat1_vbf_signal.SetName("data_qqH_cat1_ggh");
    sig_norm_subCat1_vbf.SetName(signal_subCat1_vbf.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat1_vbf);
    wout.Import(signal_subCat1_vbf); 
    wout.Import(roo_histData_subCat1_vbf_signal); 
    # wout.Print();
    wout.Write();

    # subCat 2
    fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat2_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat2.SetName("data_cat2_ggh");
    corePdf_subCat2.SetName("bkg_cat2_ggh_pdf");
    bkg_subCat2_norm.SetName(corePdf_subCat2.GetName()+"_norm");
    # make norm for data
    nevents = roo_histData_subCat2.sumEntries()
    roo_histData_subCat2_norm = rt.RooRealVar(roo_histData_subCat2.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat2_norm);
    wout.Import(roo_histData_subCat2);
    wout.Import(cat_subCat2);
    wout.Import(bkg_subCat2_norm);
    wout.Import(corePdf_subCat2);
    # wout.Print();
    wout.Write();

    fout = rt.TFile(f"{workspace_path}/workspace_sig_cat2_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat2.SetName("ggH_cat2_ggh_pdf"); 
    roo_histData_subCat2_signal.SetName("data_ggH_cat2_ggh");
    sig_norm_subCat2.SetName(signal_subCat2.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat2);
    wout.Import(signal_subCat2); 
    wout.Import(roo_histData_subCat2_signal); 

    signal_subCat2_vbf.SetName("qqH_cat2_ggh_pdf"); 
    roo_histData_subCat2_vbf_signal.SetName("data_qqH_cat2_ggh");
    sig_norm_subCat2_vbf.SetName(signal_subCat2_vbf.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat2_vbf);
    wout.Import(signal_subCat2_vbf); 
    wout.Import(roo_histData_subCat2_vbf_signal); 
    # wout.Print();
    wout.Write();


    # subCat 3
    fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat3_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat3.SetName("data_cat3_ggh");
    corePdf_subCat3.SetName("bkg_cat3_ggh_pdf");
    bkg_subCat3_norm.SetName(corePdf_subCat3.GetName()+"_norm");
    # make norm for data
    nevents = roo_histData_subCat3.sumEntries()
    roo_histData_subCat3_norm = rt.RooRealVar(roo_histData_subCat3.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat3_norm);
    wout.Import(roo_histData_subCat3);
    wout.Import(cat_subCat3);
    wout.Import(bkg_subCat3_norm);
    wout.Import(corePdf_subCat3);
    # wout.Print();
    wout.Write();

    fout = rt.TFile(f"{workspace_path}/workspace_sig_cat3_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat3.SetName("ggH_cat3_ggh_pdf"); 
    roo_histData_subCat3_signal.SetName("data_ggH_cat3_ggh");
    sig_norm_subCat3.SetName(signal_subCat3.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat3);
    wout.Import(signal_subCat3); 
    wout.Import(roo_histData_subCat3_signal); 

    signal_subCat3_vbf.SetName("qqH_cat3_ggh_pdf"); 
    roo_histData_subCat3_vbf_signal.SetName("data_qqH_cat3_ggh");
    sig_norm_subCat3_vbf.SetName(signal_subCat3_vbf.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat3_vbf);
    wout.Import(signal_subCat3_vbf); 
    wout.Import(roo_histData_subCat3_vbf_signal); 
    # wout.Print();
    wout.Write();

    # subCat 4
    fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat4_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat4.SetName("data_cat4_ggh");
    corePdf_subCat4.SetName("bkg_cat4_ggh_pdf");
    bkg_subCat4_norm.SetName(corePdf_subCat4.GetName()+"_norm");
    # make norm for data
    nevents = roo_histData_subCat4.sumEntries()
    roo_histData_subCat4_norm = rt.RooRealVar(roo_histData_subCat4.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat4_norm);
    wout.Import(roo_histData_subCat4);
    wout.Import(cat_subCat4);
    wout.Import(bkg_subCat4_norm);
    wout.Import(corePdf_subCat4);
    # wout.Print();
    wout.Write();

    fout = rt.TFile(f"{workspace_path}/workspace_sig_cat4_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat4.SetName("ggH_cat4_ggh_pdf"); 
    roo_histData_subCat4_signal.SetName("data_ggH_cat4_ggh");
    sig_norm_subCat4.SetName(signal_subCat4.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat4);
    wout.Import(signal_subCat4); 
    wout.Import(roo_histData_subCat4_signal); 

    signal_subCat4_vbf.SetName("qqH_cat4_ggh_pdf"); 
    roo_histData_subCat4_vbf_signal.SetName("data_qqH_cat4_ggh");
    sig_norm_subCat4_vbf.SetName(signal_subCat4_vbf.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat4_vbf);
    wout.Import(signal_subCat4_vbf); 
    wout.Import(roo_histData_subCat4_vbf_signal);
    # wout.Print();
    wout.Write();

    
    # ---------------------------------------------------
    # Group plotting start here
    # ---------------------------------------------------

    # ---------------------------------------------------
    # Plot 6.23 blinded
    # ---------------------------------------------------
    save_fname = f"{plot_save_path}/fig6_23"
    subCat_dataHists = [
        roo_histData_subCat0,
        roo_histData_subCat1,
        roo_histData_subCat2,
        roo_histData_subCat3,
        roo_histData_subCat4,
    ]
    SMF_func_l = [
        subCat0_SMF,
        subCat1_SMF,
        subCat2_SMF,
        subCat3_SMF,
        subCat4_SMF,
    ]
    y_range_l = [
        (7e-3, 13e-3),
        (7e-3, 13e-3),
        (7e-3, 14e-3),
        (5.6e-3, 18.5e-3),
        (5.6e-3, 18.5e-3),
    ]
    plot_6_23(mass, roo_histData_allCat, subCat_dataHists, SMF_func_l, fitResult, save_fname, y_range_l=y_range_l)

    # ---------------------------------------------------
    # plot Fig 6.26 blinded
    # ---------------------------------------------------
    # define the sim pdfs
    bkg_pdf_dict = {
        "subCat0_BWZRedux": model_subCat0_BWZRedux, 
        "subCat1_BWZRedux": model_subCat1_BWZRedux,
        "subCat2_BWZRedux": model_subCat2_BWZRedux,
        "subCat3_BWZRedux": model_subCat3_BWZRedux,
        "subCat4_BWZRedux": model_subCat4_BWZRedux,
        "subCat0_sumExp": model_subCat0_sumExp, 
        "subCat1_sumExp": model_subCat1_sumExp,
        "subCat2_sumExp": model_subCat2_sumExp,
        "subCat3_sumExp": model_subCat3_sumExp,
        "subCat4_sumExp": model_subCat4_sumExp,
        "subCat0_FEWZxBern": model_subCat0_FEWZxBern, 
        "subCat1_FEWZxBern": model_subCat1_FEWZxBern,
        "subCat2_FEWZxBern": model_subCat2_FEWZxBern,
        "subCat3_FEWZxBern": model_subCat3_FEWZxBern,
        "subCat4_FEWZxBern": model_subCat4_FEWZxBern,
    }
    sig_pdf_dict = {
        "signal_subCat0" : signal_subCat0,
        "signal_subCat1" : signal_subCat1,
        "signal_subCat2" : signal_subCat2,
        "signal_subCat3" : signal_subCat3,
        "signal_subCat4" : signal_subCat4,
    }
    sim_sigBkg_pdf, parameters_sigBkg = getSigBkgPdf(bkg_pdf_dict, sig_pdf_dict)
    # ------------------------
    sim_sigBkg_pdf = {
        "subCat0_BWZRedux": sim_sigBkg_pdf["subCat0_BWZRedux"], 
        "subCat1_BWZRedux": sim_sigBkg_pdf["subCat1_BWZRedux"],
        "subCat2_BWZRedux": sim_sigBkg_pdf["subCat2_BWZRedux"],
        "subCat3_BWZRedux": sim_sigBkg_pdf["subCat3_BWZRedux"],
        "subCat4_BWZRedux": sim_sigBkg_pdf["subCat4_BWZRedux"],
        # ----------------------------
        "subCat0_sumExp": sim_sigBkg_pdf["subCat0_sumExp"], 
        "subCat1_sumExp": sim_sigBkg_pdf["subCat1_sumExp"],
        "subCat2_sumExp": sim_sigBkg_pdf["subCat2_sumExp"],
        "subCat3_sumExp": sim_sigBkg_pdf["subCat3_sumExp"],
        "subCat4_sumExp": sim_sigBkg_pdf["subCat4_sumExp"],
        #----------------------------
        "subCat0_FEWZxBern": sim_sigBkg_pdf["subCat0_FEWZxBern"], 
        "subCat1_FEWZxBern": sim_sigBkg_pdf["subCat1_FEWZxBern"],
        "subCat2_FEWZxBern": sim_sigBkg_pdf["subCat2_FEWZxBern"],
        "subCat3_FEWZxBern": sim_sigBkg_pdf["subCat3_FEWZxBern"],
        "subCat4_FEWZxBern": sim_sigBkg_pdf["subCat4_FEWZxBern"],
        
    }
    # -----------------------------------------------------
    
    # save_fname = f"{plot_save_path}/fig6_26"
    subCat_dataHists = [
        roo_histData_subCat0,
        roo_histData_subCat1,
        roo_histData_subCat2,
        roo_histData_subCat3,
        roo_histData_subCat4,
    ]

    core_funcs = {
        "BWZRedux" : "BWZRedux",
        "sumExp" : "SumExp",
        "FEWZxBern" : "FEWZxBern",
    }
    for core_func, coreFuncName in core_funcs.items():
        save_fname = f"{plot_save_path}/fig6_26_{coreFuncName}"
        
        multi_pdf_l = [
            sim_sigBkg_pdf[f"subCat0_{core_func}"],
            sim_sigBkg_pdf[f"subCat1_{core_func}"],
            sim_sigBkg_pdf[f"subCat2_{core_func}"],
            sim_sigBkg_pdf[f"subCat3_{core_func}"],
            sim_sigBkg_pdf[f"subCat4_{core_func}"],
        ]
        plot_6_26(mass, subCat_dataHists, multi_pdf_l, fitResult, save_fname, coreFuncName=coreFuncName, unblind=False)
        

    # ---------------------------------------------------
    # Unblinded fitting
    # ---------------------------------------------------

    # perform fit over full 110, 150
    # CAUTION: make the parameters in the workspace is saved and closed

    # freeze back the core function s
    # # BWZ redux
    # a_coeff.setConstant(True)
    # b_coeff.setConstant(True)
    # c_coeff.setConstant(True)
    
    # # sumExp
    # a1_coeff.setConstant(True)
    # a2_coeff.setConstant(True)
    # f_coeff.setConstant(True)

    # # FEWZxBern
    # c1.setConstant(True)
    # c2.setConstant(True)
    # c3.setConstant(True)
    # c4.setConstant(True)


    # # SMF coeffs
    # a0_subCat0.setConstant(True)
    # a1_subCat0.setConstant(True)
    # a3_subCat0.setConstant(True)

    # a0_subCat1.setConstant(True)
    # a1_subCat1.setConstant(True)
    # a0_subCat2.setConstant(True)
    # a1_subCat2.setConstant(True)
    # a0_subCat3.setConstant(True)
    # a1_subCat3.setConstant(True)
    # a0_subCat4.setConstant(True)
    # a1_subCat4.setConstant(True)

    # MH_subCat0.setConstant(False)
    MH_subCat0.setConstant(True) # all other MH subcat refers to MH_subCat0
    # MH_subCat0.Print("v")
    # print(f"MH_subCat0: {MH_subCat0.getVal()}")
    # raise ValueError
  
    CMS_hmm_sigma_cat0_ggh.setConstant(True)
    CMS_hmm_peak_cat0_ggh.setConstant(True)
    
    CMS_hmm_sigma_cat1_ggh.setConstant(True)
    CMS_hmm_peak_cat1_ggh.setConstant(True)
    
    CMS_hmm_sigma_cat2_ggh.setConstant(True)
    CMS_hmm_peak_cat2_ggh.setConstant(True)
    
    CMS_hmm_sigma_cat3_ggh.setConstant(True)
    CMS_hmm_peak_cat3_ggh.setConstant(True)
    
    CMS_hmm_sigma_cat4_ggh.setConstant(True)
    CMS_hmm_peak_cat4_ggh.setConstant(True)
    
    # ------------------------
    simPdf = rt.RooSimultaneous(
                                "simPdf", 
                                "simultaneous pdf", 
                                sim_sigBkg_pdf,
                                sample,
    )
    fitResult = simPdf.fitTo(combData, EvalBackend=device, PrintLevel=0 ,Save=True,SumW2Error=True)
    # fitResult.Print()
    # raise ValueError
    # ---------------------------------------------------
    # Plot 6.23 unblinded
    # ---------------------------------------------------
    save_fname = f"{plot_save_path}/fig6_23_unblinded"
    subCat_dataHists = [
        roo_histData_subCat0,
        roo_histData_subCat1,
        roo_histData_subCat2,
        roo_histData_subCat3,
        roo_histData_subCat4,
    ]
    SMF_func_l = [
        subCat0_SMF,
        subCat1_SMF,
        subCat2_SMF,
        subCat3_SMF,
        subCat4_SMF,
    ]
    y_range_l = [
        (7e-3, 13e-3),
        (7e-3, 13e-3),
        (7e-3, 14e-3),
        (5.6e-3, 18.5e-3),
        (5.6e-3, 18.5e-3),
    ]
    # plot_6_23(mass, roo_histData_allCat, subCat_dataHists, SMF_func_l, fitResult, save_fname, y_range_l=y_range_l)
    

    # ---------------------------------------------------
    # plot Fig 6.26
    # ---------------------------------------------------
    
    save_fname = f"{plot_save_path}/fig6_26"
    subCat_dataHists = [
        roo_histData_subCat0,
        roo_histData_subCat1,
        roo_histData_subCat2,
        roo_histData_subCat3,
        roo_histData_subCat4,
    ]
    # multi_pdf_l = [
    #     corePdf_subCat0,
    #     corePdf_subCat1,
    #     corePdf_subCat2,
    #     corePdf_subCat3,
    #     corePdf_subCat4,
    # ]
    # multi_pdf_l = [
    #     model_subCat0_BWZRedux,
    #     model_subCat1_BWZRedux,
    #     model_subCat2_BWZRedux,
    #     model_subCat3_BWZRedux,
    #     model_subCat4_BWZRedux,
    # ]
    # multi_pdf_l = [
    #     sim_sigBkg_pdf["subCat0_BWZRedux"],
    #     sim_sigBkg_pdf["subCat1_BWZRedux"],
    #     sim_sigBkg_pdf["subCat2_BWZRedux"],
    #     sim_sigBkg_pdf["subCat3_BWZRedux"],
    #     sim_sigBkg_pdf["subCat4_BWZRedux"],
    # ]
    # plot_6_26(mass, subCat_dataHists, multi_pdf_l, fitResult, save_fname, coreFuncName="BWZRedux")
    # multi_pdf_l = [
    #     sim_sigBkg_pdf["subCat0_sumExp"],
    #     sim_sigBkg_pdf["subCat1_sumExp"],
    #     sim_sigBkg_pdf["subCat2_sumExp"],
    #     sim_sigBkg_pdf["subCat3_sumExp"],
    #     sim_sigBkg_pdf["subCat4_sumExp"],
    # ]
    # plot_6_26(mass, subCat_dataHists, multi_pdf_l, fitResult, save_fname, coreFuncName="SumExp")
    multi_pdf_l = [
        sim_sigBkg_pdf["subCat0_FEWZxBern"],
        sim_sigBkg_pdf["subCat1_FEWZxBern"],
        sim_sigBkg_pdf["subCat2_FEWZxBern"],
        sim_sigBkg_pdf["subCat3_FEWZxBern"],
        sim_sigBkg_pdf["subCat4_FEWZxBern"],
    ]
    plot_6_26(mass, subCat_dataHists, multi_pdf_l, fitResult, save_fname, coreFuncName="FEWZxBern", unblind=True)
    
    # print(f"data_subCat0_signal sumentries: {data_subCat0_signal.sumEntries()}")
    # print(f"data_subCat1_signal sumentries: {data_subCat1_signal.sumEntries()}")
    # print(f"data_subCat2_signal sumentries: {data_subCat2_signal.sumEntries()}")
    # print(f"data_subCat3_signal sumentries: {data_subCat3_signal.sumEntries()}")
    # print(f"data_subCat4_signal sumentries: {data_subCat4_signal.sumEntries()}")
    # # raise ValueError

    # -------------------------------------------------------------------------
    # do signal plotting for all sub-Cats in one plot
    # -------------------------------------------------------------------------
    sig_dict_by_sample = {
        "ggh_signal" : [
            signal_subCat0, 
            signal_subCat1,
            signal_subCat2,
            signal_subCat3,
            signal_subCat4,
        ],
    }
    sigHist_list = [ # for signal function normalization
        roo_histData_subCat0_signal,
        roo_histData_subCat1_signal,
        roo_histData_subCat2_signal,
        roo_histData_subCat3_signal,
        roo_histData_subCat4_signal
    ]
    plotSigBySample(mass, sig_dict_by_sample, sigHist_list, plot_save_path)

    sig_dict_by_sample = {
        "vbf_signal" : [
            signal_subCat0_vbf, 
            signal_subCat1_vbf,
            signal_subCat2_vbf,
            signal_subCat3_vbf,
            signal_subCat4_vbf,
        ]
    }
    sigHist_list = [ # for signal function normalization
        roo_histData_subCat0_vbf_signal,
        roo_histData_subCat1_vbf_signal,
        roo_histData_subCat2_vbf_signal,
        roo_histData_subCat3_vbf_signal,
        roo_histData_subCat4_vbf_signal
    ]
    plotSigBySample(mass, sig_dict_by_sample, sigHist_list, plot_save_path)
        

    # -------------------------------------------------------------------------
    # do Bkg plotting loop divided into core-function
    # -------------------------------------------------------------------------
    
    model_dict_by_coreFunction = {
        "BWZRedux" : [
            model_subCat0_BWZRedux, 
            model_subCat1_BWZRedux,
            model_subCat2_BWZRedux,
            model_subCat3_BWZRedux,
            model_subCat4_BWZRedux,
        ],
        "sumExp" : [
            model_subCat0_sumExp, 
            model_subCat1_sumExp,
            model_subCat2_sumExp,
            model_subCat3_sumExp,
            model_subCat4_sumExp,
        ],
        "FEWZxBern" : [
            model_subCat0_FEWZxBern, 
            model_subCat1_FEWZxBern,
            model_subCat2_FEWZxBern,
            model_subCat3_FEWZxBern,
            model_subCat4_FEWZxBern,
        ],
        # "FEWZxBern" : [
        #     coreFEWZxBern_SubCat0, 
        #     coreFEWZxBern_SubCat1,
        #     coreFEWZxBern_SubCat2,
        #     coreFEWZxBern_SubCat3,
        #     coreFEWZxBern_SubCat4,
        # ],
        "SMF" : [
            subCat0_SMF, 
            subCat1_SMF,
            subCat2_SMF,
            subCat3_SMF,
            subCat4_SMF,
        ],
    }
    rooHist_list = [ # for normalization histogram reference
        roo_histData_subCat0,
        roo_histData_subCat1,
        roo_histData_subCat2,
        roo_histData_subCat3,
        roo_histData_subCat4
    ]
    plotBkgByCoreFunc(mass, model_dict_by_coreFunction, rooHist_list, plot_save_path)
    

    # -------------------------------------------------------------------------
    # do Bkg plotting loop divided into Sub Categories
    # -------------------------------------------------------------------------

    model_dict_by_subCat = {
        0 : [
            model_subCat0_BWZRedux, 
            model_subCat0_sumExp,
            model_subCat0_FEWZxBern,
        ],
        1 : [
            model_subCat1_BWZRedux, 
            model_subCat1_sumExp,
            model_subCat1_FEWZxBern,
        ],
        2 : [
            model_subCat2_BWZRedux, 
            model_subCat2_sumExp,
            model_subCat2_FEWZxBern,
        ],
        3 : [
            model_subCat3_BWZRedux, 
            model_subCat3_sumExp,
            model_subCat3_FEWZxBern,
        ],
        4 : [
            model_subCat4_BWZRedux, 
            model_subCat4_sumExp,
            model_subCat4_FEWZxBern,
        ],
    }
    data_dict_by_subCat = {
        0 : roo_histData_subCat0,
        1 : roo_histData_subCat1,
        2 : roo_histData_subCat2,
        3 : roo_histData_subCat3,
        4 : roo_histData_subCat4,
    }
    plotBkgBySubCat(mass, model_dict_by_subCat, data_dict_by_subCat, plot_save_path)

    


