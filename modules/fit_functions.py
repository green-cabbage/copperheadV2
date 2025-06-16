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

def getShapeModifierHist(x, allCat_hist, subCat_hist, normalize=False, nbins=100):
    x_name = x.GetName()
    nbins_old = x.getBins()
    nbins_new = nbins
    reBinFactor = int(nbins_old/nbins_new)
    # print(f"nbins_old : {nbins_old}")
    # print(f"nbins_new : {nbins_new}")
    # print(f"reBinFactor : {reBinFactor}")
    allCat_th1 = allCat_hist.createHistogram(x_name).Clone("allCat_clone").Rebin(reBinFactor) # clone it just in case
    subCat_th1 = subCat_hist.createHistogram(x_name).Clone("subCat_clone").Rebin(reBinFactor) # clone it just in case
    subCat_th1.Divide(allCat_th1)
    if normalize:
        subCat_th1.Scale(1/subCat_th1.Integral()) # normalize to one
    rooHist_name = "shapModifier_hist"
    roo_hist_shapModifier = rt.RooDataHist(rooHist_name, rooHist_name, rt.RooArgSet(x), subCat_th1) 
    return roo_hist_shapModifier

def plot_6_23(x, roo_histData_allCat, subCat_dataHists, SMF_func_l, fitResult, save_fname, normalize=True, nbins=100):
    x_name = x.GetName()
    for ix in range(len(subCat_dataHists)):
    # for ix in range(1):
        canvas = rt.TCanvas("canvas","canvas",800, 800) # giving a specific name for each canvas prevents segfault
        canvas.cd()
        # Define upper and lower pads
        pad1 = ROOT.TPad("pad1", "Distribution", 0, 0.3, 1, 1.0)
        pad2 = ROOT.TPad("pad2", "Ratio", 0, 0.0, 1, 0.3)
        
        # Adjust margins
        pad1.SetBottomMargin(0)  # Upper plot does not need bottom margin
        pad2.SetTopMargin(0)     # Lower plot does not need top margin
        pad2.SetBottomMargin(0.3)

        pad1.SetTicks(2, 2)
        pad2.SetTicks(2, 2)
        pad1.Draw() # value plot
        pad2.Draw() # ratio plot

        # Top pad start
        pad1.cd()
        frame = x.frame()
        roo_hist_shapModifier = getShapeModifierHist(x, roo_histData_allCat, subCat_dataHists[ix], normalize=normalize, nbins=nbins)
        # plot the SMF fit function first
        roo_hist_shapModifier.plotOn(frame, Invisible=True) # Invisible plot for SMF functions to plot over
        SMF_func = SMF_func_l[ix]
        SMF_func.plotOn(frame, VisualizeError=(fitResult, 1), FillColor=rt.kCyan, Components=SMF_func.GetName()) # don't need the specify component name, but I guess it's good practice
        SMF_func.plotOn(frame, LineColor=rt.kRed)
        
        # plot the shape modifier data

        roo_hist_shapModifier.plotOn(frame)
        

        # plot settings
        frame.Draw()
        # frame.GetYaxis().SetLabelSize(0.08)
        # frame.GetYaxis().SetRangeUser(0.98, 1.02)
        if normalize:
            frame.GetYaxis().SetTitle("A.U.")
        else:
            frame.GetYaxis().SetTitle("Events")
        frame.SetTitle("")
        
        # Bottom pad start
        pad2.cd()
        ratio_frame= x.frame()
        
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{save_fname}_subCat{ix}.pdf")
    raise ValueError


