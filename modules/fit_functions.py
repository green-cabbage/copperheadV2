import ROOT
import ROOT as rt
from typing import Tuple, List, Dict
ROOT.gStyle.SetOptStat(0) # remove stats box

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

def getPdfToHist(x, pdf, hist2copy):
    pdf_hist = hist2copy.Clone("pdf_hist")
    for i in range(1, pdf_hist.GetNbinsX()+1):
        xval = pdf_hist.GetXaxis().GetBinCenter(i)
        x.setVal(xval)
        # get uncertainty on PDF at this point from fit result
        pdf_val = pdf.getVal(ROOT.RooArgSet(x))
        pdf_hist.SetBinContent(i, pdf_val)  # ratio = 1
        pdf_hist.SetBinError(i, 0) # remove errors
    return pdf_hist

# def getRatioHist(x, hist, pdf):
#     ratio_hist = hist.Clone("ratio_hist")
#     for i in range(1, hist.GetNbinsX()+1):
#         xval = ratio_hist.GetXaxis().GetBinCenter(i)
#         x.setVal(xval)
    
#         # get uncertainty on PDF at this point from fit result
#         hist_val = ratio_hist.GetBinContent(i)
#         pdf_val = pdf.getVal(ROOT.RooArgSet(x))
#         print(f"bin {i} hist_val: {hist_val}")
#         print(f"bin {i} pdf_val: {pdf_val}")
#         ratio_hist.SetBinContent(i, hist_val/pdf_val)  # ratio = 1
#         ratio_hist.SetBinError(i, 0) # remove errors
#     return ratio_hist

def getRatioHist(x, hist, pdf):
    hist_clone = hist.Clone("ratio_hist")
    hist_pdf = getPdfToHist(x, pdf, hist)
    # normalize
    hist_clone.Scale(1/hist_clone.Integral())
    hist_pdf.Scale(1/hist_pdf.Integral())
    # for i in range(1, hist_clone.GetNbinsX()+1):
    #        # get uncertainty on PDF at this point from fit result
    #     hist_val = hist_clone.GetBinContent(i)
    #     pdf_val = hist_pdf.GetBinContent(i)
    #     print(f"bin {i} hist_val: {hist_val}")
    #     print(f"bin {i} pdf_val: {pdf_val}")
    ratio_hist = hist_clone
    ratio_hist.Divide(hist_pdf)

    # Style: black dots with error bars
    ratio_hist.SetMarkerStyle(20)        # Filled circle
    ratio_hist.SetMarkerSize(1.0)
    ratio_hist.SetMarkerColor(ROOT.kBlack)
    ratio_hist.SetLineColor(ROOT.kBlack)  # Error bars in black
    return ratio_hist


def getUnityHistBand(x, pdf, fitResult, hist2copy):
    """
    from roofit histogram, generate a histogram with value one with relative fit errors from pdf and paste them in the same TH1 format as hist2copy
    """
    h_band = hist2copy.Clone("h_band")
    for i in range(1, h_band.GetNbinsX()+1):
        xval = h_band.GetXaxis().GetBinCenter(i)
        x.setVal(xval)
    
        # get uncertainty on PDF at this point from fit result
        val = pdf.getVal(ROOT.RooArgSet(x))
        err = pdf.getPropagatedError(fitResult)
    
        # rel_err = err / val if val != 0 else 0
        rel_err = err
        h_band.SetBinContent(i, 1.0)  # ratio = 1
        h_band.SetBinError(i, rel_err)
        # print(f"bin {i} rel_err: {rel_err}")
        # print(f"bin {i} val: {val}")
        # print(f"bin {i} err: {err}")
        

    # Style
    h_band.SetFillColor(ROOT.kBlue - 9)
    h_band.SetMarkerSize(0)
    h_band.SetLineWidth(0)
    return h_band
    
def plot_6_23(x, roo_histData_allCat, subCat_dataHists, SMF_pdf_l, fitResult, save_fname, normalize=True, nbins=100, y_range_l=None):
    # normalize=False
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
        legend = rt.TLegend(0.65,0.75,0.9,0.9)
        frame = x.frame()
        roo_hist_shapModifier = getShapeModifierHist(x, roo_histData_allCat, subCat_dataHists[ix], normalize=normalize, nbins=nbins)
        # plot the SMF fit function first
        roo_hist_shapModifier.plotOn(frame, Invisible=True) # Invisible plot for SMF functions to plot over
        SMF_pdf = SMF_pdf_l[ix]
        SMF_pdf.plotOn(frame, VisualizeError=(fitResult, 1), FillColor=(ROOT.kBlue - 9), Components=SMF_pdf.GetName()) # don't need the specify component name, but I guess it's good practice
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),"Uncertainty", "F")
        
        SMF_pdf.plotOn(frame, LineColor=rt.kRed)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),"Polynomial fit", "L")
        
        
        # plot the shape modifier data

        roo_hist_shapModifier.plotOn(frame)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),"Shape Modifier", "PE")
        
        

        # plot settings
        frame.Draw()
        legend.Draw()
        # frame.GetYaxis().SetLabelSize(0.08)
        # frame.GetYaxis().SetRangeUser(0.98, 1.02)
        if normalize:
            frame.GetYaxis().SetTitle("A.U.")
        else:
            frame.GetYaxis().SetTitle("Events")
        frame.SetTitle("")
        if y_range_l is not None:
            x_min, x_max = y_range_l[ix]
            frame.GetYaxis().SetRangeUser(x_min, x_max)
        
        # Bottom pad start
        pad2.cd()
        ratio_frame= x.frame()

        # SMF_pdf_hist = SMF_pdf.createHistogram(x.GetName()).Clone("SMF_pdf_clone")
        # ratio_hist = roo_hist_shapModifier.createHistogram(x.GetName()).Clone("ratio_hist")
        # ratio_hist.Scale(SMF_pdf_hist.Integral() / ratio_hist.Integral())  # normalize
        # ratio_hist.Divide(SMF_pdf_hist)

        shapeModifier_hist = roo_hist_shapModifier.createHistogram(x.GetName())
        ratio_hist = getRatioHist(x, shapeModifier_hist, SMF_pdf)
        ratio_hist.GetYaxis().SetRangeUser(0.9, 1.1)
        ratio_hist.SetTitle("")
        ratio_hist.GetYaxis().SetTitle("Data/Pred")
        ratio_hist.GetXaxis().SetTitle("m_{\mu\mu} [GeV]")
        
            
        h_band = getUnityHistBand(x, SMF_pdf, fitResult, ratio_hist)

        # start draw
        h_band.Draw("E2") # draw h band first
        # change style to add a straight red line
        h_band_line = h_band.Clone("h_bandClone")
        h_band_line.SetLineColor(ROOT.kRed)
        h_band_line.SetLineWidth(2)
        h_band_line.SetFillStyle(0)  # No fill
        h_band_line.Draw("HIST L SAME")
        ratio_hist.Draw("E1 SAME")
        

        
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{save_fname}_subCat{ix}.pdf")
    # raise ValueError



def getSigBkgPdf(bkg_pdf_dict, sig_pdf_dict, nSubCats=5):
    parameters = []
    sim_sigBkg_pdf = {}
    for ix in range(nSubCats):
        name = f"frac_subCat{ix}"
        frac = rt.RooRealVar(name,name,0.01, 0.0, 1.0) 

        bwz_redux = bkg_pdf_dict[f"subCat{ix}_BWZRedux"]
        sum_exp = bkg_pdf_dict[f"subCat{ix}_sumExp"]
        fewzXbern = bkg_pdf_dict[f"subCat{ix}_FEWZxBern"]
        signal_ggh = sig_pdf_dict[f"signal_subCat{ix}"]
        
        name = f"sigBkg_subCat{ix}_BWZRedux"
        sigBkg_BWZRedux = rt.RooAddPdf(name, name, [signal_ggh, bwz_redux], [frac])
        name = f"sigBkg_subCat{ix}_sumExp"
        sigBkg_sumExp = rt.RooAddPdf(name, name, [signal_ggh, sum_exp], [frac])
        name = f"sigBkg_subCat{ix}_FEWZxBern"
        sigBkg_FEWZxBern = rt.RooAddPdf(name, name, [signal_ggh, fewzXbern], [frac])
        
        parameters.append(frac)
        
        sim_sigBkg_pdf[f"subCat{ix}_BWZRedux"] = sigBkg_BWZRedux
        sim_sigBkg_pdf[f"subCat{ix}_sumExp"] = sigBkg_sumExp
        sim_sigBkg_pdf[f"subCat{ix}_FEWZxBern"] = sigBkg_FEWZxBern

    return sim_sigBkg_pdf, parameters
        

def rebinHist(x, roofitHist,  nbins, normalize=False,):
    x_name = x.GetName()
    nbins_old = x.getBins()
    nbins_new = nbins
    reBinFactor = int(nbins_old/nbins_new)
    # print(f"nbins_old : {nbins_old}")
    # print(f"nbins_new : {nbins_new}")
    # print(f"reBinFactor : {reBinFactor}")
    roofit_th1 = roofitHist.createHistogram(x_name).Clone("subCat_clone").Rebin(reBinFactor) # clone it just in case
    if normalize:
        roofit_th1.Scale(1/roofit_th1.Integral()) # normalize to one
    rooHist_name = roofitHist.GetName() + f"rebinned_{nbins}"
    roofitHist_rebinned = rt.RooDataHist(rooHist_name, rooHist_name, rt.RooArgSet(x), roofit_th1) 
    return roofitHist_rebinned

def get_pdf_by_name(add_pdf, name):
    """
    Extracts a sub-PDF from a RooAddPdf by its name.

    Args:
        add_pdf (ROOT.RooAddPdf): The RooAddPdf instance.
        name (str): The name of the sub-PDF to extract.

    Returns:
        ROOT.RooAbsPdf or None: The extracted RooAbsPdf if found, otherwise None.
    """
    if not isinstance(add_pdf, ROOT.RooAddPdf):
        print("Error: Input is not a RooAddPdf instance.")
        return None

    # Get the list of component PDFs
    pdf_list = add_pdf.pdfList()

    # Iterate through the list and find the PDF by name
    # In PyROOT, you can iterate directly over RooArgList
    for i in range(pdf_list.getSize()):
        current_pdf = pdf_list.at(i) # Use at(i) to get the element
        if current_pdf and current_pdf.GetName() == name:
            return current_pdf

    print(f"Warning: PDF with name '{name}' not found in RooAddPdf '{add_pdf.GetName()}'.")
    return None

def get_fracFromAddPdf(add_pdf, frac_name):
    """
    Extracts the fraction (yield) RooAbsReal for a specific sub-PDF by its name
    from a RooAddPdf.
    """
    if not isinstance(add_pdf, ROOT.RooAddPdf):
        print("Error: Input is not a RooAddPdf instance.")
        return None

    coeff_list = add_pdf.coefList()

    for i in range(coeff_list.getSize()):
        current_frac = coeff_list.at(i) # Use at(i) to get the element
        if current_frac.GetName() == frac_name:
            return current_frac

    else:
        print(f"Warning: coeff with name '{frac_name}' not found in RooAddPdf '{add_pdf.GetName()}'.")
        return None


# def getResidHistBand(x, pdf, fitResult, hist2copy):
#     """
#     from roofit histogram, generate a histogram with value one with relative fit errors from pdf and paste them in the same TH1 format as hist2copy
#     """
#     h_band = hist2copy.Clone("h_band")
#     for i in range(1, h_band.GetNbinsX()+1):
#         xval = h_band.GetXaxis().GetBinCenter(i)
#         x.setVal(xval)
    
#         # get uncertainty on PDF at this point from fit result
#         val = pdf.getVal(ROOT.RooArgSet(x))
#         err = pdf.getPropagatedError(fitResult)
        
#         rel_err = err * val 
#         hist_val = hist2copy.GetBinContent(i) 
#         h_band.SetBinContent(i, 0.0)  # ratio = 1
#         h_band.SetBinError(i, rel_err*2*hist_val)
#         print(f"bin {i} rel_err: {rel_err}")
#         print(f"bin {i} val: {val}")
#         print(f"bin {i} err: {err}")
#         print(f"bin {i} hist_val: {hist_val}")
        

#     # Style
#     h_band.SetFillColor(ROOT.kOrange)
#     h_band.SetMarkerSize(0)
#     h_band.SetLineWidth(0)
#     # raise ValueError
#     return h_band


def getResidHistBand(x, pdf, fitResult, dataHist, n_sigma=1, color=rt.kGreen):
    """
    Source: https://root-forum.cern.ch/t/problems-with-errors-for-residhist/51455/5
    """
    nbins=dataHist.numEntries() # match the nbins from dataHist
    old_nbins = x.getBins()
    x.setBins(nbins) 
    # h_band = hist2copy.Clone("h_band")
    nBkg = rt.RooRealVar("nBkg", "nBkg", 5000, 0, 10000)
    binning = x.getBinning()
    h_band = dataHist.createHistogram(x.GetName()).Clone("h_band")
    for i in range(dataHist.numEntries()):
        # xval = h_band.GetXaxis().GetBinCenter(i)
        # x.setVal(xval)
        x.setRange("range_for_bin", binning.binLow(i), binning.binHigh(i))
        bkgPdfIntegral = pdf.createIntegral(x, rt.RooFit.NormSet(x), rt.RooFit.Range("range_for_bin"))
        bkgYield = rt.RooProduct("bkgYield", "bkgYield", [bkgPdfIntegral, nBkg])
        one_sigma_err = bkgYield.getPropagatedError(fitResult)
        # print(f"bin {i} dataHist->weight(): {dataHist.weight()}")
        # print(f"bin {i} dataHist->weightError(): {dataHist.weightError()}")
        # print(f"bin {i} bkgYield.getPropagatedError(fitResult): {one_sigma_err}")
        # print(f"bin {i} binning.binLow(i): {binning.binLow(i)}")
        # print(f"bin {i} binning.binHigh(i): {binning.binHigh(i)}")

        h_band.SetBinContent(i+1, 0.0) 
        h_band.SetBinError(i+1, one_sigma_err*n_sigma)

    # Style
    h_band.SetFillColor(color)
    h_band.SetMarkerSize(0)
    h_band.SetLineWidth(0)

    # convert to RooDataHist
    # x_name = x.GetName()
    # h_band = rt.RooDataHist(x_name, x_name, rt.RooArgSet(x), h_band) 

    # for i in range(h_band.numEntries()):
    #     coord = h_band.get(i)  # returns RooArgSet
    #     yval = h_band.weight(i)  # bin content
    #     yerr = h_band.weightError(i)  # bin error
    #     print(f"Bin {i}:val = {yval:.2f} ± {yerr:.2f}")


    x.setBins(old_nbins) 
    # print(f"old_nbins: {old_nbins}")
    # raise ValueError
    return h_band


def plot_6_26(x, subCat_dataHists, multi_pdf_l, fitResult, save_fname, target_nbins=50):
    x_name = x.GetName()
    sig_yield_multiply_l = [
        50,
        50,
        30,
        30,
        20
    ]
    
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
        legend = rt.TLegend(0.55,0.65,0.9,0.9)
        frame = x.frame()
        subCat_dataHist = subCat_dataHists[ix]
        subCat_dataHist = rebinHist(x, subCat_dataHist, target_nbins) # rebin
         
        
        multi_pdf = multi_pdf_l[ix]
        
        subCat_dataHist.plotOn(frame, Invisible=True)
        bkg_pdf_name = f"model_SubCat{ix}_SMFxBWZRedux"
        multi_pdf.plotOn(frame, Components=bkg_pdf_name, Invisible=True) 
        hresid_bkg_only = frame.residHist() # obtain residual for later

        multi_pdf.plotOn(frame, VisualizeError=(fitResult, 2), FillColor=(ROOT.kOrange), Components=bkg_pdf_name) 
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),"+/ 2\sigma", "F")
        
        multi_pdf.plotOn(frame, VisualizeError=(fitResult, 1), FillColor=(ROOT.kGreen), Components=bkg_pdf_name) 
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),"+/ 1\sigma", "F")
        
        multi_pdf.plotOn(frame, LineColor=rt.kRed, LineWidth=2, Components=bkg_pdf_name, LineStyle=rt.kDashed)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),"Fitted background", "L")
        
        
        
        multi_pdf.plotOn(frame, LineColor=rt.kRed, LineWidth=2)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),"S+B fit", "L")
        subCat_dataHist.plotOn(frame)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),"Data", "PE")
        
        
        add_pdf = multi_pdf
        sig_frac = get_fracFromAddPdf(add_pdf, f"frac_subCat{ix}")
        sig_frac.Print("v")
        original_frac_val = sig_frac.getVal()
        multipy_val = sig_yield_multiply_l[ix]
        sig_frac.setVal(original_frac_val*multipy_val)
        multi_pdf.plotOn(frame, LineColor=rt.kBlue, LineWidth=2, Components=f"ggH_cat{ix}_ggh_pdf")
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),f"Post-fit signal x {multipy_val}, m_H = 125 GeV", "L")

        frame.Draw()
        legend.Draw()

        
        print(f"original_frac_val: {original_frac_val}")
        print(f"subCat {ix} dataHist sumentries: {subCat_dataHist.sumEntries()}")
        print(f"subCat {ix} signal yield: {subCat_dataHist.sumEntries()*original_frac_val}")
        # done with pad1
        
        # Bottom pad start
        pad2.cd()
        frame_resid = x.frame()
        frame_resid.addPlotable(hresid_bkg_only, "P", invisible=True)
        frame_resid.Draw() # draw invisible residual to set y range in pad2
        

        # set fraction back to normal
        sig_frac.setVal(original_frac_val)

        bkg_pdf = get_pdf_by_name(multi_pdf, bkg_pdf_name)
        # print(f"bkg_pdf.GetName(): {bkg_pdf.GetName()}")
        # raise ValueError
        bkgOnly_resid_pdf = rt.RooGenericPdf("bkg_resid_pdf", "@0-@0", rt.RooArgList(bkg_pdf))
        # bkg_resid_pdf.plotOn(frame, VisualizeError=(fitResult, 2), FillColor=(ROOT.kOrange)) 
        # bkg_resid_pdf.plotOn(frame, VisualizeError=(fitResult, 1), FillColor=(ROOT.kGreen)) 
        
        bkgOnly_resid_pdf.plotOn(frame_resid, LineColor=rt.kRed, LineStyle=rt.kDashed, LineWidth=2)

        # multi_pdf.Print("V")
        # sigBkg_resid_pdf = rt.RooGenericPdf("sigBkg_resid_pdf", "@0-@1", rt.RooArgList(multi_pdf,bkg_pdf))

        sig_pdf_name = f"ggH_cat{ix}_ggh_pdf"
        sigBkg_resid_pdf = get_pdf_by_name(multi_pdf, sig_pdf_name)
        
        sigBkg_resid_pdf.plotOn(frame_resid, LineColor=rt.kRed, LineStyle=rt.kSolid, LineWidth=2)

        # Get the Erro bands
        # h_band_sig2 = getResidHistBand(x, bkg_pdf, fitResult, subCat_dataHist, n_sigma=2, color=rt.kOrange)
        # h_band_sig1 = getResidHistBand(x, bkg_pdf, fitResult, subCat_dataHist, n_sigma=1, color=rt.kGreen)
        h_band_sig2 = getResidHistBand(x, multi_pdf, fitResult, subCat_dataHist, n_sigma=2, color=rt.kOrange)
        h_band_sig1 = getResidHistBand(x, multi_pdf, fitResult, subCat_dataHist, n_sigma=1, color=rt.kGreen)
        
        # plot the residual data points again, but visible this time
        frame_resid.addPlotable(hresid_bkg_only, "P")
        
        # draw 
        h_band_sig2.Draw("E2 SAME")
        h_band_sig1.Draw("E2 SAME")
        frame_resid.Draw("SAME")
        # frame_resid.Draw()
        
        # continue
        
        # done with pad2
        
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{save_fname}_subCat{ix}.pdf")
    
    fitResult.Print()
    # raise ValueError