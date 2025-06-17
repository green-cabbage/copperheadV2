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
    
def plot_6_23(x, roo_histData_allCat, subCat_dataHists, SMF_pdf_l, fitResult, save_fname, normalize=True, nbins=100):
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
        frame = x.frame()
        roo_hist_shapModifier = getShapeModifierHist(x, roo_histData_allCat, subCat_dataHists[ix], normalize=normalize, nbins=nbins)
        # plot the SMF fit function first
        roo_hist_shapModifier.plotOn(frame, Invisible=True) # Invisible plot for SMF functions to plot over
        SMF_pdf = SMF_pdf_l[ix]
        SMF_pdf.plotOn(frame, VisualizeError=(fitResult, 1), FillColor=(ROOT.kBlue - 9), Components=SMF_pdf.GetName()) # don't need the specify component name, but I guess it's good practice
        pull_hist = frame.pullHist() # to be used later
        SMF_pdf.plotOn(frame, LineColor=rt.kRed)
        
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

        # SMF_pdf_hist = SMF_pdf.createHistogram(x.GetName()).Clone("SMF_pdf_clone")
        # ratio_hist = roo_hist_shapModifier.createHistogram(x.GetName()).Clone("ratio_hist")
        # ratio_hist.Scale(SMF_pdf_hist.Integral() / ratio_hist.Integral())  # normalize
        # ratio_hist.Divide(SMF_pdf_hist)

        shapeModifier_hist = roo_hist_shapModifier.createHistogram(x.GetName())
        ratio_hist = getRatioHist(x, shapeModifier_hist, SMF_pdf)
        ratio_hist.GetYaxis().SetRangeUser(0.9, 1.1)
        ratio_hist.SetTitle("")

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
        

        # ratio_hist = rt.RooDataHist("ratio_hist", "ratio_hist", rt.RooArgSet(x), ratio_hist) 
        # ratio_hist.plotOn(ratio_frame)
        
        # ratio_frame.addPlotable(pull_hist, "P")
        # ratio_smf_pdf = rt.RooGenericPdf("ratio_smf_pdf", "0.5*@0-0.5*@0 +1", rt.RooArgList(SMF_pdf))
        # ratio_smf_pdf = rt.RooGenericPdf("ratio_smf_pdf", "@0/@0", rt.RooArgList(SMF_pdf))

        
        # ratio_smf_pdf.plotOn(ratio_frame, VisualizeError=(fitResult, 1), FillColor=rt.kCyan, Components=SMF_pdf.GetName()) # don't need the specify component name, but I guess it's good practice
        # # ratio_smf_pdf.plotOn(ratio_frame, VisualizeError=(fitResult, 1), FillColor=rt.kCyan) # don't need the specify component name, but I guess it's good practice
        # ratio_smf_pdf.plotOn(ratio_frame, LineColor=rt.kRed)
        
        # ratio_frame.GetYaxis().SetTitle("Data/Pred")
        # ratio_frame.GetXaxis().SetTitle("m_{\mu\mu} [GeV]")
        # ratio_frame.Draw()
        
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{save_fname}_subCat{ix}.pdf")
    raise ValueError


