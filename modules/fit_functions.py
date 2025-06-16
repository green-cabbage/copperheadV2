import ROOT
import ROOT as rt
from typing import Tuple, List, Dict

def getFEWZ_roospline(x, root_path):
    """
    Extract RooSpline1D instance that we assume has been saved in ucsd_workspace/fewz.root
    with the name "fewz_1j_spl_order1_cat_ggh" (which we will keep)
    replace the variable that the RooSpline1D was constructed with, with our own variable "x"
    so fitTo could work with the rest of the roofit pdfs
    """
    # ucsd_spline = rt.TFile("modules/ucsd_workspace/fewz.root")["fewz_1j_spl_order1_cat_ggh"]
    ucsd_spline = rt.TFile(f"{root_path}/fewz.root")["fewz_1j_spl_order1_cat_ggh"]
    ucsd_var = ucsd_spline.getVariables()[0]
    # replace the variable with our variable
    customizer = rt.RooCustomizer(ucsd_spline, "")
    customizer.replaceArg(ucsd_var, x)
    roo_spline_func = customizer.build()
    name = "fewz_1j_spl_order1_cat_ggh"
    roo_spline_func.SetName(name)
    return roo_spline_func



def MakeFEWZxBernDof3(
        name_final:str, 
        title:str, 
        mass: rt.RooRealVar, 
        BernCoeff_list,
    ) ->Tuple[rt.RooProdPdf, Dict]:
    """
    params:
    mass = rt.RooRealVar that we will fitTo
    dof = degrees of freedom given to this model. Since the spline
    has no dof, all the dof is inserted to the Bernstein
    """
    # collect all variables that we don't want destroyed by Python once function ends
    out_dict = {}

    # name = f"BernsteinFast"
    # n_coeffs = len(BernCoeff_list)
    # bern_model = rt.RooBernsteinFast(n_coeffs)(name, name, mass, BernCoeff_list)
    name = f"Bernstein_FEWZxBern"
    bern_model = rt.RooBernstein(name, name, mass, BernCoeff_list) # we assume that we have one extra frozen parameter
    out_dict[name] = bern_model # add model to make python remember


    
    # make the spline portion
    roo_spline_func = getFEWZ_roospline(mass, "ucsd_workspace/")# extract from ucsd 's fews root file
    out_dict[roo_spline_func.GetName()] = roo_spline_func
    name = "fewz_1j_spl_pdf"
    roo_spline_pdf = rt.RooWrapperPdf(name, name, roo_spline_func)
    out_dict[name] = roo_spline_pdf # add model to make python remember  

    final_model = rt.RooProdPdf(name_final, name_final, [bern_model, roo_spline_pdf]) 
   
    return (final_model, out_dict)

def getShapeModifierHist(x, x_rebinned, allCat_hist, subCat_hist, normalize=False, nbins=nbins):
    x_name = x.GetName()
    nbins_old = x.getBins()
    nbins_new = x_rebinned.getBins()
    # nbins_new = nbins
    reBinFactor = int(nbins_old/nbins_new)
    print(f"nbins_old : {nbins_old}")
    print(f"nbins_new : {nbins_new}")
    print(f"reBinFactor : {reBinFactor}")
    allCat_th1 = allCat_hist.createHistogram(x_name).Clone("allCat_clone").Rebin(reBinFactor) # clone it just in case
    subCat_th1 = subCat_hist.createHistogram(x_name).Clone("subCat_clone").Rebin(reBinFactor) # clone it just in case
    subCat_th1.Divide(allCat_th1)
    if normalize:
        subCat_th1.Scale(1/subCat_th1.Integral()) # normalize to one
    rooHist_name = "shapModifier_hist"
    roo_hist_shapModifier = rt.RooDataHist(rooHist_name, rooHist_name, rt.RooArgSet(x_rebinned), subCat_th1) 
    return roo_hist_shapModifier

def plot_6_23(x, roo_histData_allCat, subCat_dataHists, save_fname, normalize=False, nbins=100):
    x_name = x.GetName()
    x_rebinned = rt.RooRealVar(x.GetName(), x.GetName(), x.getVal(), x.getMin(), x.getMax())
    x_rebinned.setBins(nbins)
    for ix in range(len(subCat_dataHists)):
    # for ix in range(1):
        canvas = rt.TCanvas("canvas","canvas",800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        
        # frame = x.frame(110, 150, nbins)
        frame = x.frame(Bins=nbins)
        roo_hist_shapModifier = getShapeModifierHist(x, x_rebinned, roo_histData_allCat, subCat_dataHists[ix], normalize=normalize)
        roo_hist_shapModifier.plotOn(frame)
        frame.Draw()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{save_fname}_subCat{ix}.pdf")
    raise ValueError