import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client

"""
time to fit : define the core functions with their parameters
"""
from typing import Tuple, List, Dict
import ROOT as rt

def MakeBWZxBern(mass: rt.RooRealVar, order: int) ->Tuple[rt.RooAddPdf, Dict]:
    """
    params:
    mass = rt.RooRealVar that we will fitTo
    order = order of the sum of exponential, that we assume to be >= 2
    """
    # collect all variables that we don't want destroyed by Python once function ends
    out_dict = {}
    
    # make BernStein
    bern_order = order-1
    BernCoeff_list = []
    for ix in range(bern_order):
        name = f"Bernstein_c_{ix}"
        coeff = rt.RooRealVar(name,name, 1.0,-1.0,5.0)
        out_dict[name] = coeff # add variable to make python remember 
        BernCoeff_list.append(coeff)
    name = f"Bernstein_model_order_{bern_order}"
    bern_model = rt.RooBernstein(name, name, mass, BernCoeff_list)
    out_dict[name] = bern_model # add variable to make python remember

    
    # make BWZ
    bwWidth = rt.RooRealVar("bwWidth", "bwWidth", 2.5, 0, 30)
    bwmZ = rt.RooRealVar("bwmZ", "bwmZ", 91.2, 90, 92)
    bwWidth.setConstant(True)
    # bwmZ.setConstant(True)
    out_dict[bwWidth.GetName()] = bwWidth 
    out_dict[bwmZ.GetName()] = bwmZ 
    
    name = "VanillaBW_model"
    BWZ = rt.RooBreitWigner(name, name, mass, bwmZ,bwWidth)
    # our BWZ model is also multiplied by exp(a* mass) as defined in the AN
    name = "BWZ_exp_coeff"
    expCoeff = rt.RooRealVar(name, name, -0.5, -1.0, 1.0)
    name = "BWZ_exp_model"
    exp_model = rt.RooExponential(name, name, mass, expCoeff)
    name = "BWZxExp"
    full_BWZ = rt.RooProdPdf(name, name, [BWZ, exp_model]) 

    # add variables
    out_dict[BWZ.GetName()] = BWZ 
    out_dict[expCoeff.GetName()] = expCoeff 
    out_dict[exp_model.GetName()] = exp_model 
    out_dict[full_BWZ.GetName()] = full_BWZ 
    
    # multiply BWZ and Bernstein
    name = f"BWZxBern_order_{order}"
    final_model = rt.RooProdPdf(name, name, [bern_model, full_BWZ]) 
   
    return (final_model, out_dict)
    

def MakeSumExponential(mass: rt.RooRealVar, order: int) ->Tuple[rt.RooAddPdf, Dict]:
    """
    params:
    mass = rt.RooRealVar that we will fitTo
    order = order of the sum of exponential, that we assume to be >= 2
    returns:
    rt.RooAddPdf
    dictionary of variables with {variable name : rt.RooRealVar or rt.RooExponential} format mainly for keep python from
    destroying these variables, but also useful in debugging
    """
    model_list = [] # list of RooExp models for RooAddPdf
    a_i_list = [] # list of RooExp coeffs for RooAddPdf
    rest_list = [] # list of rest of variables to save it from being destroyed
    for ix in range(order):
        name = f"S_exp_b_{ix}"
        b_i = rt.RooRealVar(name, name, -0.5, -5.0, 1.0)
        rest_list.append(b_i)
        
        name = f"S_exp_model_{ix}"
        model = rt.RooExponential(name, name, mass, b_i)
        model_list.append(model)
        
        if ix >0:
            name = f"S_exp_a_{ix}"
            a_i = rt.RooRealVar(name, name, 0.5, 0, 1.0)
            a_i_list.append(a_i)
            
    name = f"S_exp_order_{order}"
    final_model = rt.RooAddPdf(name, name, model_list, a_i_list)
    # collect all variables that we don't want destroyed by Python once function ends
    out_dict = {}
    for model in model_list:
        out_dict[model.GetName()] = model
    for a_i in a_i_list:
        out_dict[a_i.GetName()] = a_i
    for var in rest_list:
        out_dict[var.GetName()] = var
    return (final_model, out_dict)


if __name__ == "__main__":
    client =  Client(n_workers=31,  threads_per_worker=1, processes=True, memory_limit='4 GiB') 
    load_path = "/depot/cms/users/yun79/results/stage1/test_VBF-filter_JECon_07June2024/2018/f1_0/"
    full_load_path = load_path+f"/data_C/*/*.parquet"
    events = dak.from_parquet(full_load_path)
    # figure out the discontinuous fit range later ---------------------
    
    mass_arr = ak.to_numpy(events.dimuon_mass.compute())
    # for debugging purposes -----------------
    binning = np.linspace(110, 140, 100)
    np_hist, _ = np.histogram(mass_arr, bins=binning)
    # print(f"np_hist: {np_hist}")
    # -------------------------------------------

    # start Root fit 
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    mass_name = "dimuon_mass"
    mass =  rt.RooRealVar(mass_name,"mass (GeV)",120,110,150)
    nbins = 100
    mass.setBins(nbins)
    # set sideband mass range
    mass.setRange("loSB", 110, 115 )
    mass.setRange("hiSB", 135, 150 )
    mass.setRange("h_peak", 115, 135 )
    mass.setRange("full", 110, 150 )
    fit_range = "loSB,hiSB" # we're fitting bkg only
    
    roo_dataset = rt.RooDataSet.from_numpy({mass_name: mass_arr}, [mass])
    order = 3
    BWZxBern, params_bern = MakeBWZxBern(mass, order)
    sumExp, params_exp = MakeSumExponential(mass, order)
   
    # print(f"params: {params}")
    # roo_dataset.Print()
    roo_hist = rt.RooDataHist("data_hist","binned version of roo_dataset", rt.RooArgSet(mass), roo_dataset)  # copies binning from mass variable
    # roo_hist.Print()

    # begin multi-pdf
    cat = rt.RooCategory("pdf_index","Index of Pdf which is active")
    pdflist = rt.RooArgList()
    # Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # 0 == BWZxBern
    # 1 == sumExp
    pdflist.add(BWZxBern)
    pdflist.add(sumExp)
    
    # multipdf = rt.RooMultiPdf(
    #     # f"multipdf_{self.channel}_{category}", 
    #     "multipdf",
    #     "multipdf", 
    #     cat, 
    #     pdflist
    # )
    

    rt.EnableImplicitMT()
    # _ = BWZxBern.fitTo(roo_hist, rt.RooFit.Range(fit_range), Save=True,  EvalBackend ="cpu")
    # fit_result = BWZxBern.fitTo(roo_hist, rt.RooFit.Range(fit_range), Save=True,  EvalBackend ="cpu")
    _ = BWZxBern.fitTo(roo_hist, Save=True,  EvalBackend ="cpu")
    fit_result = BWZxBern.fitTo(roo_hist, Save=True,  EvalBackend ="cpu")

    # draw on canvas
    frame = mass.frame()
    roo_dataset.plotOn(frame, rt.RooFit.CutRange(fit_range), DataError="SumW2", Name="data_hist")
    BWZxBern.plotOn(frame, Name="BWZxBern", LineColor=rt.kGreen)

    frame.Draw()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"./quick_plots/stage3_plot_test.pdf")
    