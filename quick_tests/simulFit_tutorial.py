import time
import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client
from omegaconf import OmegaConf

from typing import Tuple, List, Dict
import ROOT as rt
from quickSMFtest_functions import MakeFEWZxBernDof3

if __name__ == "__main__":
    load_path = "./processed_events_data.parquet"
    processed_eventsData = ak.from_parquet(load_path)
    print("events loaded!")
    
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
    
    # Initialize BWZ Redux
    # --------------------------------------------------------------
    name = f"BWZ_Redux_a_coeff"
    a_coeff = rt.RooRealVar(name,name, -0.0146,-0.02,0.03)
    name = f"BWZ_Redux_b_coeff"
    b_coeff = rt.RooRealVar(name,name, -0.000111,-0.001,0.001)
    name = f"BWZ_Redux_c_coeff"
    c_coeff = rt.RooRealVar(name,name, 0.462,-5.0,5.0)


    
    # subCat 0
    name = "subCat0_BWZ_Redux"
    coreBWZredux_SubCat0 = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
     
    # Construct background pdf
    a0_subCat0 = rt.RooRealVar("a0_subCat0", "a0_subCat0", -0.1, -1, 1)
    a1_subCat0 = rt.RooRealVar("a1_subCat0", "a1_subCat0", 0.5, -1, 1)
    a3_subCat0 = rt.RooRealVar("a3_subCat0", "a3_subCat0", 0.5, -1, 1)

    name = "subCat0_SMF"
    subCat0_SMF = rt.RooChebychev(name, name, mass, [a0_subCat0, a1_subCat0, a3_subCat0])


    
    # Construct composite pdf
    name = "subCat0_SMFxBWZredux"
    model_subCat0_BWZredux = rt.RooProdPdf(name, name, [coreBWZredux_SubCat0, subCat0_SMF])


    
    # subCat 1
    name = "subCat1_BWZ_Redux"
    # coreBWZredux_SubCat1 = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
    coreBWZredux_SubCat1 = coreBWZredux_SubCat0
    
    # Construct the background pdf
    a0_subCat1 = rt.RooRealVar("a0_subCat1", "a0_subCat1", -0.1, -1, 1)
    a1_subCat1 = rt.RooRealVar("a1_subCat1", "a1_subCat1", 0.5, -1, 1)
    a3_subCat1 = rt.RooRealVar("a3_subCat1", "a3_subCat1", 0.5, -1, 1)
    name =  "subCat1_SMF"
    subCat1_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat1, 
                              a1_subCat1, 
                              a3_subCat1
                             ])
     
    # Construct the composite model
    name = "model_SubCat1_SMFxBWZredux"
    model_subCat1_BWZredux = rt.RooProdPdf(name, name, [coreBWZredux_SubCat1, subCat1_SMF])

    # subCat 2
    name = "subCat2_BWZ_Redux"
    # coreBWZredux_SubCat2 = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
    coreBWZredux_SubCat2 = coreBWZredux_SubCat0
    
    # Construct the background pdf
    a0_subCat2 = rt.RooRealVar("a0_subCat2", "a0_subCat2", -0.1, -1, 1)
    a1_subCat2 = rt.RooRealVar("a1_subCat2", "a1_subCat2", 0.5, -1, 1)
    name = "subCat2_SMF"
    subCat2_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat2, 
                              a1_subCat2, 
                             ])
    name = "model_SubCat2_SMFxBWZRedux"
    model_subCat2_BWZredux = rt.RooProdPdf(name, name, [coreBWZredux_SubCat2, subCat2_SMF])    

    # subCat 3
    name = "subCat3_BWZ_Redux"
    # coreBWZredux_SubCat3 = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
    coreBWZredux_SubCat3 = coreBWZredux_SubCat0
    
    # Construct the background pdf
    a0_subCat3 = rt.RooRealVar("a0_subCat3", "a0_subCat3", -0.1, -1, 1)
    a1_subCat3 = rt.RooRealVar("a1_subCat3", "a1_subCat3", 0.5, -1, 1)
    name = "subCat3_SMF"
    subCat3_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat3, 
                              a1_subCat3, 
                             ])
    name = "model_SubCat3_SMFxBWZRedux"
    model_subCat3_BWZredux = rt.RooProdPdf(name, name, [coreBWZredux_SubCat3, subCat3_SMF])  

    # subCat 4
    name = "subCat4_BWZ_Redux"
    # coreBWZredux_SubCat4 = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
    coreBWZredux_SubCat4 = coreBWZredux_SubCat0
    
    # Construct the background pdf
    a0_subCat4 = rt.RooRealVar("a0_subCat4", "a0_subCat4", -0.1, -1, 1)
    a1_subCat4 = rt.RooRealVar("a1_subCat4", "a1_subCat4", 0.5, -1, 1)
    name = "subCat4_SMF"
    subCat4_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat4, 
                              a1_subCat4, 
                             ])
    name = "model_SubCat4_SMFxBWZRedux"
    model_subCat4_BWZredux = rt.RooProdPdf(name, name, [coreBWZredux_SubCat4, subCat4_SMF])  


    # ---------------------------------------------------------------
    # Generate events for both samples
    # ---------------------------------------------------------------
     
    # do for cat idx 0
    subCat_filter = (processed_eventsData["subCategory_idx"] == 0)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat0 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat0 = rt.RooDataHist("subCat0_rooHist_BWZredux","subCat0_rooHist_BWZredux", rt.RooArgSet(mass), roo_datasetData_subCat0)
    data_subCat0_BWZredux = roo_histData_subCat0

    # do for cat idx 1
    subCat_filter = (processed_eventsData["subCategory_idx"] == 1)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat1 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat1 = rt.RooDataHist("subCat1_rooHist_BWZredux","subCat1_rooHist_BWZredux", rt.RooArgSet(mass), roo_datasetData_subCat1)
    data_subCat1_BWZredux = roo_histData_subCat1

    # do for cat idx 2
    subCat_filter = (processed_eventsData["subCategory_idx"] == 2)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat2 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat2 = rt.RooDataHist("subCat2_rooHist_BWZredux","subCat2_rooHist_BWZredux", rt.RooArgSet(mass), roo_datasetData_subCat2)
    data_subCat2_BWZredux = roo_histData_subCat2

    # do for cat idx 3
    subCat_filter = (processed_eventsData["subCategory_idx"] == 3)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat3 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat3 = rt.RooDataHist("subCat3_rooHist_BWZredux","subCat3_rooHist_BWZredux", rt.RooArgSet(mass), roo_datasetData_subCat3)
    data_subCat3_BWZredux = roo_histData_subCat3

    # do for cat idx 4
    subCat_filter = (processed_eventsData["subCategory_idx"] == 4)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat4 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat4 = rt.RooDataHist("subCat4_rooHist_BWZredux","subCat4_rooHist_BWZredux", rt.RooArgSet(mass), roo_datasetData_subCat4)
    data_subCat4_BWZredux = roo_histData_subCat4




    # --------------------------------------------------------------
    # Initialize Sum Exponential
    # --------------------------------------------------------------
    # name = f"RooSumTwoExpPdf_a1_coeff"
    # a1_coeff = rt.RooRealVar(name,name, -0.0603,-2.0,1)
    # name = f"RooSumTwoExpPdf_a2_coeff"
    # a2_coeff = rt.RooRealVar(name,name, -0.0450,-2.0,1)
    # name = f"RooSumTwoExpPdf_f_coeff"
    # f_coeff = rt.RooRealVar(name,name, 0.742,0.0,1.0)

    # name = f"RooSumTwoExpPdf_a1_coeff"
    # a1_coeff = rt.RooRealVar(name,name, -0.2,-2.0,1)
    # name = f"RooSumTwoExpPdf_a2_coeff"
    # a2_coeff = rt.RooRealVar(name,name, -0.09,-2.0,1)
    # name = f"RooSumTwoExpPdf_f_coeff"
    # f_coeff = rt.RooRealVar(name,name, 0.02,0.0,1.0)

    # name = f"RooSumTwoExpPdf_a1_coeff"
    # a1_coeff = rt.RooRealVar(name,name, -0.059609,-2.0,1)
    # name = f"RooSumTwoExpPdf_a2_coeff"
    # a2_coeff = rt.RooRealVar(name,name, -0.0625122,-2.0,1)
    # name = f"RooSumTwoExpPdf_f_coeff"
    # f_coeff = rt.RooRealVar(name,name, 0.9,0.0,1.0)

    name = f"RooSumTwoExpPdf_a1_coeff"
    a1_coeff = rt.RooRealVar(name,name, -0.043657,-2.0,1)
    name = f"RooSumTwoExpPdf_a2_coeff"
    a2_coeff = rt.RooRealVar(name,name, -0.23726,-2.0,1)
    name = f"RooSumTwoExpPdf_f_coeff"
    f_coeff = rt.RooRealVar(name,name, 0.9,0.0,1.0)
    
    name = "subCat0_sumExp"
    coreSumExp_SubCat0 = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
     
    name = "subCat0_SMF_sumExp"
    subCat0_SumExp_SMF = rt.RooChebychev(name, name, mass, [a0_subCat0, a1_subCat0, a3_subCat0])


    
    # Construct composite pdf
    name = "model_SubCat0_SMFxSumExp"
    model_subCat0_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat0, subCat0_SumExp_SMF])
     
    # subCat 1
    name = "subCat1_sumExp"
    # coreSumExp_SubCat1 = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
    coreSumExp_SubCat1 = coreSumExp_SubCat0
    

    name = "subCat1_SMF_sumExp"
    subCat1_SumExp_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat1, 
                              a1_subCat1, 
                              a3_subCat1
                             ])
     
    # Construct the composite model
    name = "model_SubCat1_SMFxSumExp"
    model_subCat1_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat1, subCat1_SumExp_SMF])

    # subCat 2
    name = "subCat2_sumExp"
    # coreSumExp_SubCat2 = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
    coreSumExp_SubCat2 = coreSumExp_SubCat0
    
    name = "subCat2_SMF_sumExp"
    subCat2_SumExp_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat2, 
                              a1_subCat2, 
                             ])
    name = "model_SubCat2_SMFxSumExp"
    model_subCat2_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat2, subCat2_SumExp_SMF])    

    # subCat 3
    name = "subCat3_sumExp"
    # coreSumExp_SubCat3 = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
    coreSumExp_SubCat3 = coreSumExp_SubCat0
    
    name = "subCat3_SMF_sumExp"
    subCat3_SumExp_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat3, 
                              a1_subCat3, 
                             ])
    name = "model_SubCat3_SMFxSumExp"
    model_subCat3_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat3, subCat3_SumExp_SMF])    

    # subCat 4
    name = "subCat4_sumExp"
    # coreSumExp_SubCat4 = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
    coreSumExp_SubCat4 = coreSumExp_SubCat0
    
    name = "subCat4_SMF_sumExp"
    subCat4_SumExp_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat4, 
                              a1_subCat4, 
                             ])
    name = "model_SubCat4_SMFxSumExp"
    model_subCat4_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat4, subCat4_SumExp_SMF])    
     
    # Generate events for both samples
    # ---------------------------------------------------------------
     
    # do for cat idx 0
    subCat_filter = (processed_eventsData["subCategory_idx"] == 0)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat0_sumExp = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat0_sumExp = rt.RooDataHist("subCat0_rooHist_sumExp","subCat0_rooHist_sumExp", rt.RooArgSet(mass), roo_datasetData_subCat0_sumExp)
    data_subCat0_sumExp = roo_histData_subCat0_sumExp

    # do for cat idx 1
    subCat_filter = (processed_eventsData["subCategory_idx"] == 1)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat1_sumExp = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat1_sumExp = rt.RooDataHist("subCat1_rooHist_sumExp","subCat1_rooHist_sumExp", rt.RooArgSet(mass), roo_datasetData_subCat1_sumExp)
    data_subCat1_sumExp = roo_histData_subCat1_sumExp

    # do for cat idx 2
    subCat_filter = (processed_eventsData["subCategory_idx"] == 2)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat2_sumExp = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat2_sumExp = rt.RooDataHist("subCat2_rooHist_sumExp","subCat2_rooHist_sumExp", rt.RooArgSet(mass), roo_datasetData_subCat2_sumExp)
    data_subCat2_sumExp = roo_histData_subCat2_sumExp

    # do for cat idx 3
    subCat_filter = (processed_eventsData["subCategory_idx"] == 3)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat3_sumExp = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat3_sumExp = rt.RooDataHist("subCat3_rooHist_sumExp","subCat3_rooHist_sumExp", rt.RooArgSet(mass), roo_datasetData_subCat3_sumExp)
    data_subCat3_sumExp = roo_histData_subCat3_sumExp


    # do for cat idx 4
    subCat_filter = (processed_eventsData["subCategory_idx"] == 4)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat4_sumExp = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat4_sumExp = rt.RooDataHist("subCat4_rooHist_sumExp","subCat4_rooHist_sumExp", rt.RooArgSet(mass), roo_datasetData_subCat4_sumExp)
    data_subCat4_sumExp = roo_histData_subCat4_sumExp


    # --------------------------------------------------------------
    # Initialize FEWZxBernstein
    # --------------------------------------------------------------

    name = f"FEWZxBern_c1"
    c1 = rt.RooRealVar(name,name, 0.2,-2,2)
    name = f"FEWZxBern_c2"
    c2 = rt.RooRealVar(name,name, 1.0,-2,2)
    name = f"FEWZxBern_c3"
    c3 = rt.RooRealVar(name,name, 0.5,-2,2)
    
    name = "subCat0_FEWZxBern"
    coreFEWZxBern_SubCat0, params_FEWZxBern_SubCat0 = MakeFEWZxBernDof3(name, name, mass, c1, c2, c3) 
     
    name = "subCat0_SMF_FEWZxBern"
    subCat0_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, [a0_subCat0, a1_subCat0, a3_subCat0])


    
    # Construct composite pdf
    name = "model_SubCat0_SMFxFEWZxBern"
    model_subCat0_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat0, subCat0_FEWZxBern_SMF])
     
    # subCat 1
    name = "subCat1_FEWZxBern"
    # coreFEWZxBern_SubCat1, params_FEWZxBern_SubCat1 = MakeFEWZxBernDof3(name, name, mass, c1, c2, c3) 
    coreFEWZxBern_SubCat1 = coreFEWZxBern_SubCat0
    

    name = "subCat1_SMF_FEWZxBern"
    subCat1_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat1, 
                              a1_subCat1, 
                              a3_subCat1
                             ])
     
    # Construct the composite model
    name = "model_SubCat1_SMFxFEWZxBern"
    model_subCat1_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat1, subCat1_FEWZxBern_SMF])

    # subCat 2
    name = "subCat2_FEWZxBern"
    # coreFEWZxBern_SubCat2, params_FEWZxBern_SubCat2 = MakeFEWZxBernDof3(name, name, mass, c1, c2, c3) 
    coreFEWZxBern_SubCat2 = coreFEWZxBern_SubCat0
    
    name = "subCat2_SMF_FEWZxBern"
    subCat2_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat2, 
                              a1_subCat2, 
                             ])
    name = "model_SubCat2_SMFxFEWZxBern"
    model_subCat2_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat2, subCat2_FEWZxBern_SMF])    

    # subCat 3
    name = "subCat3_FEWZxBern"
    # coreFEWZxBern_SubCat3, params_FEWZxBern_SubCat3 = MakeFEWZxBernDof3(name, name, mass, c1, c2, c3)  
    coreFEWZxBern_SubCat3 = coreFEWZxBern_SubCat0
    
    name = "subCat3_SMF_FEWZxBern"
    subCat3_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat3, 
                              a1_subCat3, 
                             ])
    name = "model_SubCat3_SMFxFEWZxBern"
    model_subCat3_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat3, subCat3_FEWZxBern_SMF])    

    # subCat 4
    name = "subCat4_FEWZxBern"
    # coreFEWZxBern_SubCat4, params_FEWZxBern_SubCat4 = MakeFEWZxBernDof3(name, name, mass, c1, c2, c3)  
    coreFEWZxBern_SubCat4 = coreFEWZxBern_SubCat0
    
    name = "subCat4_SMF_FEWZxBern"
    subCat4_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat4, 
                              a1_subCat4, 
                             ])
    name = "model_SubCat4_SMFxFEWZxBern"
    model_subCat4_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat4, subCat4_FEWZxBern_SMF])        
     
    # Generate events for both samples
    # ---------------------------------------------------------------
     
    # do for cat idx 0
    subCat_filter = (processed_eventsData["subCategory_idx"] == 0)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat0_FEWZxBern = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat0_FEWZxBern = rt.RooDataHist("subCat0_rooHist_FEWZxBern","subCat0_rooHist_FEWZxBern", rt.RooArgSet(mass), roo_datasetData_subCat0_FEWZxBern)
    data_subCat0_FEWZxBern = roo_histData_subCat0_FEWZxBern

    # do for cat idx 1
    subCat_filter = (processed_eventsData["subCategory_idx"] == 1)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat1_FEWZxBern = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat1_FEWZxBern = rt.RooDataHist("subCat1_rooHist_FEWZxBern","subCat1_rooHist_FEWZxBern", rt.RooArgSet(mass), roo_datasetData_subCat1_FEWZxBern)
    data_subCat1_FEWZxBern = roo_histData_subCat1_FEWZxBern

    # do for cat idx 2
    subCat_filter = (processed_eventsData["subCategory_idx"] == 2)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat2_FEWZxBern = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat2_FEWZxBern = rt.RooDataHist("subCat2_rooHist_FEWZxBern","subCat2_rooHist_FEWZxBern", rt.RooArgSet(mass), roo_datasetData_subCat2_FEWZxBern)
    data_subCat2_FEWZxBern = roo_histData_subCat2_FEWZxBern

    # do for cat idx 3
    subCat_filter = (processed_eventsData["subCategory_idx"] == 3)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat3_FEWZxBern = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat3_FEWZxBern = rt.RooDataHist("subCat3_rooHist_FEWZxBern","subCat3_rooHist_FEWZxBern", rt.RooArgSet(mass), roo_datasetData_subCat3_FEWZxBern)
    data_subCat3_FEWZxBern = roo_histData_subCat3_FEWZxBern


    # do for cat idx 4
    subCat_filter = (processed_eventsData["subCategory_idx"] == 4)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat4_FEWZxBern = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat4_FEWZxBern = rt.RooDataHist("subCat4_rooHist_FEWZxBern","subCat4_rooHist_FEWZxBern", rt.RooArgSet(mass), roo_datasetData_subCat4_FEWZxBern)
    data_subCat4_FEWZxBern = roo_histData_subCat4_FEWZxBern

    
    #----------------------------------------------------------------------------
    # Create index category and join samples
    # ---------------------------------------------------------------------------
     
    # Define category to distinguish physics and control samples events
    sample = rt.RooCategory("sample", "sample")
    sample.defineType("subCat0_BWZredux")
    sample.defineType("subCat1_BWZredux")
    sample.defineType("subCat2_BWZredux")
    sample.defineType("subCat3_BWZredux")
    sample.defineType("subCat4_BWZredux")
    sample.defineType("subCat0_sumExp")
    sample.defineType("subCat1_sumExp")
    sample.defineType("subCat2_sumExp")
    sample.defineType("subCat3_sumExp")
    sample.defineType("subCat4_sumExp")
    sample.defineType("subCat0_FEWZxBern")
    sample.defineType("subCat1_FEWZxBern")
    sample.defineType("subCat2_FEWZxBern")
    sample.defineType("subCat3_FEWZxBern")
    sample.defineType("subCat4_FEWZxBern")
     
    # Construct combined dataset in (x,sample)
    combData = rt.RooDataSet(
        "combData",
        "combined data",
        {mass},
        Index=sample,
        Import={
            "subCat0_BWZredux": data_subCat0_BWZredux, 
            "subCat1_BWZredux": data_subCat1_BWZredux,
            "subCat2_BWZredux": data_subCat2_BWZredux,
            "subCat3_BWZredux": data_subCat3_BWZredux,
            "subCat4_BWZredux": data_subCat4_BWZredux,
            "subCat0_sumExp": data_subCat0_sumExp, 
            "subCat1_sumExp": data_subCat1_sumExp,
            "subCat2_sumExp": data_subCat2_sumExp,
            "subCat3_sumExp": data_subCat3_sumExp,
            "subCat4_sumExp": data_subCat4_sumExp,
            "subCat0_FEWZxBern": data_subCat0_FEWZxBern, 
            "subCat1_FEWZxBern": data_subCat1_FEWZxBern,
            "subCat2_FEWZxBern": data_subCat2_FEWZxBern,
            "subCat3_FEWZxBern": data_subCat3_FEWZxBern,
            "subCat4_FEWZxBern": data_subCat4_FEWZxBern,
        },
    )
    # ---------------------------------------------------
    # Construct a simultaneous pdf in (x, sample)
    # -----------------------------------------------------------------------------------
     
    simPdf = rt.RooSimultaneous(
                                "simPdf", 
                                "simultaneous pdf", 
                                {
                                    "subCat0_BWZredux": model_subCat0_BWZredux, 
                                    "subCat1_BWZredux": model_subCat1_BWZredux,
                                    "subCat2_BWZredux": model_subCat2_BWZredux,
                                    "subCat3_BWZredux": model_subCat3_BWZredux,
                                    "subCat4_BWZredux": model_subCat4_BWZredux,
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
                                }, 
                                sample,
    )
    # ---------------------------------------------------
    # Perform a simultaneous fit
    # ---------------------------------------------------
     
    start = time.time()

    # _ = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend="cpu", PrintLevel=-1, Save=True)
    # fitResult = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend="cpu", PrintLevel=-1, Save=True)
    # _ = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend="cpu", PrintLevel=-1, Save=True, Strategy=0)
    # fitResult = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend="cpu", PrintLevel=-1, Save=True,)
    _ = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend="cpu",  PrintLevel=0 ,Save=True, Strategy=0)
    fitResult = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend="cpu", PrintLevel=0 ,Save=True,)
    end = time.time()
    
    fitResult.Print()
    print(f"runtime: {end-start} seconds")

    # # do plotting
    # name = "Canvas"
    # canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    
    # frame = mass.frame()
    # legend = rt.TLegend(0.65,0.55,0.9,0.7)


    # # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
    # roo_datasetData_subCat1.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0) )
    # model_subCat0.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=model_subCat0.GetName(), LineColor=rt.kGreen)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),model_subCat0.GetName(), "L")
    # model_subCat1.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=model_subCat1.GetName(), LineColor=rt.kBlue)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),model_subCat1.GetName(), "L")
    # model_subCat2.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=model_subCat2.GetName(), LineColor=rt.kRed)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),model_subCat2.GetName(), "L")

    # frame.Draw()
    # legend.Draw()        
    # canvas.Update()
    # canvas.Draw()
    # canvas.SaveAs(f"./quick_plots/simultaneousPlotTestFromTutorial.pdf")
