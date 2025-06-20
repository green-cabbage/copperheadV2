import dask_awkward as dak
import awkward as ak
from distributed import LocalCluster, Client, progress
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import mplhep as hep
import glob
import pandas as pd
import ROOT
from array import array
import dask
ROOT.gStyle.SetOptStat(0) # remove stats box

plt.style.use(hep.style.CMS)

"""
This code prints ggH/VBF channel yields after applying category cuts
"""

def getPlotVar(var: str):
    """
    Helper function that removes the variations in variable name if they exist
    """
    if "_nominal" in var:
        plot_var = var.replace("_nominal", "")
    else:
        plot_var = var
    return plot_var

def applyDijetMassCut(events):
    mass_cut = (events.jj_mass_nominal > 400) & (events.gjj_mass > 350)
    mass_cut = ak.fill_none(mass_cut, value=False)
    return events[mass_cut]

def applyVBF_phaseCut(events):
    gjj_mass_cut = (events.gjj_mass > 350)
    gjj_mass_cut = ak.fill_none(gjj_mass_cut, value=False)
    return events[gjj_mass_cut]

def applyVBF_invPhaseCut(events):
    # gjj_mass_cut = (events.gjj_mass <= 350)
    gjj_mass_cut = (events.gjj_mass > 350)
    gjj_mass_cut = ak.fill_none(gjj_mass_cut, value=False)
    gjj_mass_cut = ~gjj_mass_cut
    return events[gjj_mass_cut]


def stitch_twoSamples(events_vbf, events_100To200, keepTwo=False):
    events_vbf = applyVBF_phaseCut(events_vbf)
    events_100To200 = applyVBF_invPhaseCut(events_100To200)
    if keepTwo: # have a field that indicates whether which is VBF-filtered MC
        events_vbf["is_VBFfilter_mc"] = ak.ones_like(events_vbf.wgt_nominal, dtype="bool")
        events_100To200["is_VBFfilter_mc"] = ak.zeros_like(events_100To200.wgt_nominal, dtype="bool")
    return ak.concatenate([events_vbf, events_100To200], axis=0)

def applyVBF_cutV1(events):
    btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
    vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
    vbf_cut = ak.fill_none(vbf_cut, value=False)
    VBF_filter = (
        vbf_cut & 
        ~btag_cut # btag cut is for VH and ttH categories
    )
    # dimuon_mass = events.dimuon_mass
    # trues = ak.ones_like(dimuon_mass, dtype="bool")
    # falses = ak.zeros_like(dimuon_mass, dtype="bool")
    # events["vbf_filter"] = ak.where(VBF_filter, trues,falses)
    return events[VBF_filter]

def applyGGH_cutV1(events):
    btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
    vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
    vbf_cut = ak.fill_none(vbf_cut, value=False)
    ggH_filter = (
        ~vbf_cut & 
        ~btag_cut # btag cut is for VH and ttH categories
    )
    return events[ggH_filter]


def applyGGH_NoBtagNjet1(events):
    btagLoose_filter = ak.fill_none((events.nBtagLoose_nominal >= 2), value=False)
    btagMedium_filter = ak.fill_none((events.nBtagMedium_nominal >= 1), value=False) & ak.fill_none((events.njets_nominal >= 2), value=False)
    btag_cut = (btagLoose_filter | btagMedium_filter)
    vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
    vbf_cut = ak.fill_none(vbf_cut, value=False)
    ggH_filter = (
        ~vbf_cut & 
        ~btag_cut # btag cut is for VH and ttH categories
    )
    return events[ggH_filter]

def applyGGH_30(events):
    btagLoose_filter = ak.fill_none((events.nBtagLoose_nominal >= 2), value=False)
    btagMedium_filter = ak.fill_none((events.nBtagMedium_nominal >= 1), value=False) & ak.fill_none((events.njets_nominal >= 2), value=False)
    btag_cut = (btagLoose_filter | btagMedium_filter)
    vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35)   & (events.jet2_pt_nominal > 30) 
    vbf_cut = ak.fill_none(vbf_cut, value=False)
    jet_30_cut = ak.fill_none((events.jet1_pt_nominal > 30), value=False)
    ggH_filter = (
        ~vbf_cut 
        & ~btag_cut # btag cut is for VH and ttH categories
    )
    return events[ggH_filter]

def applyVBF_30(events):
    btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
    vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35)   & (events.jet2_pt_nominal > 30)
    vbf_cut = ak.fill_none(vbf_cut, value=False)
    dimuon_mass = events.dimuon_mass
    VBF_filter = (
        vbf_cut & 
        ~btag_cut # btag cut is for VH and ttH categories
    )
    trues = ak.ones_like(dimuon_mass, dtype="bool")
    falses = ak.zeros_like(dimuon_mass, dtype="bool")
    events["vbf_filter"] = ak.where(VBF_filter, trues,falses)
    return events[VBF_filter]


def applyGGH_cutflow(events):
    btagLoose_filter = ak.fill_none((events.nBtagLoose_nominal >= 2), value=False)
    btagMedium_filter = ak.fill_none((events.nBtagMedium_nominal >= 1), value=False) & ak.fill_none((events.njets_nominal >= 2), value=False)
    btag_cut = btagLoose_filter | btagMedium_filter
    vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
    vbf_cut = ak.fill_none(vbf_cut, value=False)
    ggH_filter = (
        ~vbf_cut & 
        ~btag_cut # btag cut is for VH and ttH categories
    )
    return events[ggH_filter]

def applyGGH_noJetPt(events):
    btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
    vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5)
    vbf_cut = ak.fill_none(vbf_cut, value=False)
    ggH_filter = (
        ~vbf_cut & 
        ~btag_cut # btag cut is for VH and ttH categories
    )
    return events[ggH_filter]

def veto_ttH_VH(events):
    btagLoose_filter = ak.fill_none((events.nBtagLoose_nominal >= 2), value=False)
    btagMedium_filter = ak.fill_none((events.nBtagMedium_nominal >= 1), value=False) & ak.fill_none((events.njets_nominal >= 2), value=False)
    btag_cut = btagLoose_filter | btagMedium_filter
    
    bool_filter = (
        ~btag_cut # btag cut is for VH and ttH categories
    )
    return events[bool_filter]


def veto_nJetGeq3(events):
    njet_filter = ak.fill_none((events.njets_nominal <= 2), value=True)
    bool_filter = (
        njet_filter # btag cut is for VH and ttH categories
    )
    return events[bool_filter]

def filterRegion(events, region="h-peak"):
    dimuon_mass = events.dimuon_mass
    if region =="h-peak":
        region = (dimuon_mass > 115.03) & (dimuon_mass < 135.03)
    elif region =="h-sidebands":
        region = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))
    elif region =="signal":
        region = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
    elif region =="z-peak":
        region = (dimuon_mass >= 70) & (dimuon_mass <= 110.0)
    elif region =="combined":
        region = (dimuon_mass >= 70) & (dimuon_mass <= 150.0)
    events = events[region]
    return events


def plot_histograms_pyroot(dy100To200, dy_vbf, dy100To200_wgt, dy_vbf_wgt, nbins=50, xmin=None, xmax=None, title="2018 UL", xlabel="Observable", save_fname = "normalized_hist_signWgt", normalize=True):

    # Auto range if not given
    all_data = np.concatenate([dy100To200, dy_vbf])
    if xmin is None:
        xmin = float(np.min(all_data))
    if xmax is None:
        xmax = float(np.max(all_data))

    
    # Create histograms
    h1 = ROOT.TH1F("dy100To200", "DY 100-200", nbins, xmin, xmax)
    h2 = ROOT.TH1F("dy_vbf",     "DY VBF",     nbins, xmin, xmax)

    # Fill histograms
    dy100To200 = array('d', dy100To200) # make the array double
    weights = array('d', dy100To200_wgt) # make the array double
    # print(f"dy100To200: {dy100To200[:10]}")
    # print(f"weights: {weights[:10]}")
    h1.FillN(len(dy100To200), dy100To200, weights)
    
    dy_vbf = array('d', dy_vbf) # make the array double
    weights = array('d', dy_vbf_wgt) # make the array double
    h2.FillN(len(dy_vbf), dy_vbf, weights)
    
    # Normalize
    if normalize:
        h1.Scale(1/h1.Integral())
        h2.Scale(1/h2.Integral())

    # Set styles
    h1.SetLineColor(ROOT.kBlue)
    h2.SetLineColor(ROOT.kRed)
    h1.SetLineWidth(2)
    h2.SetLineWidth(2)

    h1.GetXaxis().SetTitle(xlabel)
    if normalize:
        h1.GetYaxis().SetTitle("A.U.")
    h1.SetTitle(title)

    # Draw
    c = ROOT.TCanvas("c", "c", 800, 600)
    pad1 = ROOT.TPad("pad1", "Top pad", 0, 0.3, 1, 1.0)
    pad2 = ROOT.TPad("pad2", "Bottom pad", 0, 0.05, 1, 0.3)
    
    pad1.SetBottomMargin(0.02)
    pad2.SetTopMargin(0.02)
    pad2.SetBottomMargin(0.3)
    
    pad1.Draw()
    pad2.Draw()

    pad1.cd()

    h1.Draw("E")
    h2.Draw("E SAME")

    # Legend
    legend = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    legend.AddEntry(h1, "DY 100-200", "l")
    legend.AddEntry(h2, "DY VBF", "l")
    legend.Draw()

    pad2.cd()
    residual = h1.Clone("residual")
    residual.Add(h2, -1)
    residual.SetLineColor(ROOT.kBlack)
    residual.SetMarkerColor(ROOT.kBlack)
    residual.Draw("E")
    # residual.SetTitle(f";{xlabel}, nbins:{nbins};Residual")
    residual.SetTitle(f";{xlabel};Residual")
    # Similarly for the residual plot
    residual.GetXaxis().SetTitleSize(0.12)  # Bigger because bottom pad is small
    residual.GetXaxis().SetLabelSize(0.10)
    # residual.GetYaxis().SetTitleSize(0.80)
    residual.GetYaxis().SetLabelSize(0.10)

    pad1.SetTicks(2, 2)
    pad2.SetTicks(2, 2)
    c.SetGrid()
    c.Update()
    c.Draw()
    c.SaveAs(f"plots/{save_fname}.pdf")

    return c, h1, h2  # Useful if you want to save or manipulate further


def plotStitchedHStack(dy100To200_val, dy_vbf_val, dy100To200_wgt, dy_vbf_wgt, dy_vbf_isVBF_sample, nbins=50, xmin=None, xmax=None, title="2018 UL", xlabel="Observable", save_fname = "normalized_hist_signWgt"):

    # Auto range if not given
    all_data = np.concatenate([dy100To200_val, dy_vbf_val])
    if xmin is None:
        xmin = float(np.min(all_data))
    if xmax is None:
        xmax = float(np.max(all_data))

    
    # Create histograms
    print(f"xlabel: {xlabel}")
    
    if "gen" in xlabel:
        # bin_edges = np.linspace(0,600,61).tolist()
        bin_edges = np.linspace(0,600,61)
        bin_edges = array('d', bin_edges)
        # bin_edges = [float(bin_edge) for bin_edge in bin_edges] # convert to float values
        h100_nc = ROOT.TH1F("dy100To200 nc", "DY 100-200 no cut", len(bin_edges)-1, bin_edges)
        h100 = ROOT.TH1F("dy100To200", "DY 100-200", len(bin_edges)-1, bin_edges)
        hVBF = ROOT.TH1F("dy_vbf",     "DY VBF",     len(bin_edges)-1, bin_edges)
    else:
        h100_nc = ROOT.TH1F("dy100To200 nc", "DY 100-200 no cut", nbins, xmin, xmax)
        h100 = ROOT.TH1F("dy100To200", "DY 100-200", nbins, xmin, xmax)
        hVBF = ROOT.TH1F("dy_vbf",     "DY VBF",     nbins, xmin, xmax)
    # raise ValueError
    # Fill histograms
    # h100_nc
    dy100To200_val = array('d', dy100To200_val) # make the array double
    weights = array('d', dy100To200_wgt) # make the array double
    # print(f"dy100To200_val: {dy100To200_val[:10]}")
    # print(f"weights: {weights[:10]}")
    h100_nc.FillN(len(dy100To200_val), dy100To200_val, weights)

    # h100
    is_VBFfilter_mc = dy_vbf_isVBF_sample
    dy_vbf_val_fromDy100To200 = dy_vbf_val[~is_VBFfilter_mc]
    dy_vbf_val_fromDy100To200_wgt = dy_vbf_wgt[~is_VBFfilter_mc]
    dy_vbf_val_fromDy100To200 = array('d', dy_vbf_val_fromDy100To200) # make the array double
    weights = array('d', dy_vbf_val_fromDy100To200_wgt) # make the array double
    h100.FillN(len(dy_vbf_val_fromDy100To200), dy_vbf_val_fromDy100To200, weights)

    
    # hVBF
    is_VBFfilter_mc = dy_vbf_isVBF_sample
    dy_vbf_val_fromVBFFiltered = dy_vbf_val[is_VBFfilter_mc]
    dy_vbf_val_fromVBFFiltered_wgt = dy_vbf_wgt[is_VBFfilter_mc]
    dy_vbf_val_fromVBFFiltered = array('d', dy_vbf_val_fromVBFFiltered) # make the array double
    weights = array('d', dy_vbf_val_fromVBFFiltered_wgt) # make the array double
    hVBF.FillN(len(dy_vbf_val_fromVBFFiltered), dy_vbf_val_fromVBFFiltered, weights)
    

    

    # ── STACK HISTOGRAMS ────────────────────────────────────────────────
    # h100_nc.SetFillColor(ROOT.kBlue-7)
    h100.SetFillColor(ROOT.kGreen-7)
    hVBF.SetFillColor(ROOT.kRed-7)
    
    stack = ROOT.THStack("stack",f"2018UL Drell-Yan stitching;{xlabel};Events")
    
    stack.Add(h100)
    stack.Add(hVBF)
    
    hSum = h100_nc.Clone("hSum")
    # hSum.Add(h100_nc)
    hSum.SetLineColor(ROOT.kBlack)
    hSum.SetLineWidth(2)
    
    # ── DRAW ────────────────────────────────────────────────────────────────
    c = ROOT.TCanvas("c","Stitching Validation",800,600)
    c.SetLogy()
    ROOT.gStyle.SetOptStat(0)
    # stack.Draw("HIST")
    # hSum.Draw("HIST SAME")

    # stack.Draw("HIST")
    # hSum.Draw("E SAME")
    rp = ROOT.TRatioPlot(stack, hSum)
    rp.Draw()
    if "m_{jj}" in xlabel:
        rp.GetLowerRefYaxis().SetRangeUser(0.5, 1.5)
    
    # leg = ROOT.TLegend(0.65,0.65,0.9,0.9)
    leg = ROOT.TLegend(0.75,0.75,1.0,1.0)
    # leg.AddEntry(h50,  "DY-M50, gjj_mass ≤ 350","f")
    leg.AddEntry(h100, "DY-M100-200, gjj_mass < 350","f")
    leg.AddEntry(hVBF,  "DY_VBF_Filter, gjj_mass > 350","f")
    leg.AddEntry(hSum,  "DY-M100-200 (no cut)","l")
    leg.Draw()
    c.SetGrid()
    c.Update()
    c.Draw()
    c.SaveAs(f"plots/{save_fname}.pdf")
    c.SaveAs(f"plots/{save_fname}.root")


def plotVarsTHStack(events_100To200, events_vbf, variables2plot, plot_bins, save_fname="DY2018UL_THStack"):
    for var in variables2plot:
        if ("_range2" in var):
            var_reduced = var.replace("_range2","")
        else:
            var_reduced = var
        plot_var = getPlotVar(var)
            
        print(f"plot_var: {plot_var}, var: {var}")
        if plot_var not in plot_bins.keys():
            print(f"{plot_var} not available in plot_bins. skipping!")
            continue
        xmin, xmax, _ = plot_bins[plot_var]["binning_linspace"]
        xlabel = plot_bins[plot_var].get("xlabel").replace("$","")
        dy100To200 = ak.to_numpy(events_100To200[var_reduced])
        dy100To200_wgt = ak.to_numpy(events_100To200.wgt_nominal)
        dy_vbf = ak.to_numpy(events_vbf[var_reduced])
        dy_vbf_wgt = ak.to_numpy(events_vbf.wgt_nominal)
        dy_vbf_isVBF_sample = ak.to_numpy(events_vbf.is_VBFfilter_mc)
        # save_fname_var = f"DY2018UL_{var}_Ram"
        save_fname_var = f"{save_fname}_{var}"
        plotStitchedHStack(dy100To200, dy_vbf, dy100To200_wgt, dy_vbf_wgt, dy_vbf_isVBF_sample, nbins=64, xmin=xmin, xmax=xmax, xlabel=xlabel, save_fname=save_fname_var)

if __name__ == "__main__":
    client =  Client(n_workers=63,  threads_per_worker=1, processes=True, memory_limit='8 GiB') 

    with open("plot_settings.json", "r") as file:
        plot_bins = json.load(file)
    # V1_fields_2compute = [
    #     "wgt_nominal",
    #     "nBtagLoose_nominal",
    #     "nBtagMedium_nominal",
    #     "mu1_pt",
    #     "mu2_pt",
    #     "mu1_eta",
    #     "mu2_eta",
    #     "mu1_phi",
    #     "mu2_phi",
    #     "dimuon_pt",
    #     "dimuon_eta",
    #     "dimuon_phi",
    #     "dimuon_mass",
    #     "jet1_phi_nominal",
    #     "jet1_pt_nominal",
    #     "jet2_pt_nominal",
    #     "jet2_phi_nominal",
    #     "jet1_eta_nominal",
    #     "jet2_eta_nominal",
    #     "jj_mass_nominal",
    #     "jj_dEta_nominal",
    #     "event",
    #     "njets_nominal",
    #     "gjj_mass",
    #     # "run",
    #     # "event",
    #     # "luminosityBlock",
    # ]
    # fields2plot = list(plot_bins.keys())
    # fields2plot.remove
    fields2compute = [
        "nBtagLoose_nominal",
        "nBtagMedium_nominal",
        "jj_mass_nominal",
        "jj_dEta_nominal",
        "jet1_pt_nominal",
        "wgt_nominal",
    ]
    variables2plot = [
         'gjj_mass',
         'njets_nominal',
         'jet1_pt_nominal',
         'jet2_pt_nominal',
         'jet1_eta_nominal',
         'jet2_eta_nominal',
         'jet1_phi_nominal',
         'jet2_phi_nominal',
         'jet1_qgl_nominal',
         'jet2_qgl_nominal',
         'jj_dEta_nominal',
         'jj_mass_nominal',
         'jj_pt_nominal',
         'jj_dPhi_nominal',
         'zeppenfeld_nominal',
         'rpt_nominal',
         'pt_centrality_nominal',
         'nsoftjets2_nominal',
         'htsoft2_nominal',
         'nsoftjets5_nominal',
         'htsoft5_nominal',
         'dimuon_mass',
         'dimuon_pt',
         'dimuon_eta',
         'dimuon_phi',
         'dimuon_cos_theta_cs',
         'dimuon_phi_cs',
         'dimuon_cos_theta_eta',
         'dimuon_phi_eta',
         'mmj_min_dPhi_nominal',
         'mmj_min_dEta_nominal',
         'll_zstar_log_nominal',
         'dimuon_ebe_mass_res',
         'dimuon_ebe_mass_res_rel',
         'dimuon_rapidity',
         'mu1_pt',
         'mu2_pt',
         'mu1_eta',
         'mu2_eta',
         'mu1_phi',
         'mu2_phi',
         'mu1_pt_over_mass',
         'mu2_pt_over_mass',
    ]
    fields2compute = variables2plot + fields2compute
    fields2compute = list(set(fields2compute)) # remove redundant fields

    
    print(f"variables2plot: {variables2plot}")
    print(f"fields2compute: {fields2compute}")

    label="vbf_dy_validationMay30_2025"

    year = "2018"
    load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{label}/stage1_output/{year}/f1_0"
    # load_path = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/2018/f1_0/"
    region="signal"
    # region="h-sidebands"
    # target_chunksize = 150_000
    target_chunksize = 300_000
    # target_chunksize = 500_000
    # target_len = 400_000
    target_len = 200_000
    # target_len = 4_000_000
    
    # dy 100To200
    load_path_100To200 = f"{load_path}/dy_M-100To200"
    # load_path_100To200 = f"{load_path}/dy_M-100To200_aMCatNLO"
    
    # events_100To200 = dak.from_parquet(f"{load_path_100To200}/*/*.parquet")[:target_len]
    events_100To200 = dak.from_parquet(f"{load_path_100To200}/*/*.parquet")
    events_100To200 = events_100To200.repartition(rows_per_partition=target_chunksize)
    events_100To200 = filterRegion(events_100To200, region=region)
    events_100To200["wgt_nominal"] = events_100To200.wgt_nominal * 247.9 / 233.8592627 # change the cross section
    # events_100To200 = applyVBF_phaseCut(events_100To200) # gjj mass cut
    # events_100To200 = applyVBF_invPhaseCut(events_100To200) # gjj mass cut
    events_100To200_orig = events_100To200
    # events_100To200 = applyVBF_cutV1(events_100To200)
    events_100To200 = ak.zip({var: events_100To200[var] for var in fields2compute}) # add only variables to plot
    # vbf-filter
    load_path_vbf = f"{load_path}/dy_m105_160_vbf_amc"
    
    # load_path_vbf = f"{load_path}/dy_VBF_filter_NewZWgt"
    # events_vbf = dak.from_parquet(f"{load_path_vbf}/*/*.parquet")[:target_len]
    events_vbf = dak.from_parquet(f"{load_path_vbf}/*/*.parquet")
    events_vbf = events_vbf.repartition(rows_per_partition=target_chunksize)
    events_vbf = filterRegion(events_vbf, region=region)
    # events_vbf = stitch_twoSamples(events_vbf, events_100To200_orig
    keepTwo = True
    events_vbf = stitch_twoSamples(events_vbf, events_100To200_orig, keepTwo=keepTwo)
    # events_vbf = applyVBF_phaseCut(events_vbf) # gjj mass cut
    # events_vbf = applyVBF_invPhaseCut(events_vbf) # gjj mass cut
    # events_vbf = applyVBF_cutV1(events_vbf)
    if keepTwo:
        fields2compute = fields2compute + ["is_VBFfilter_mc"]
    events_vbf = ak.zip({var: events_vbf[var] for var in fields2compute}) # add only variables to plot

    # now compute
    events_100To200, events_vbf = dask.compute((events_100To200, events_vbf))[0]

    if "jj_mass_nominal" in variables2plot:
        variables2plot += ["jj_mass_nominal_range2"]

    # my old stitching code start ------------------------------------------
    # for var in variables2plot:
    #     if ("_range2" in var):
    #         var_reduced = var.replace("_range2","")
    #     else:
    #         var_reduced = var
    #     plot_var = getPlotVar(var)
            
    #     print(f"plot_var: {plot_var}, var: {var}")
    #     if plot_var not in plot_bins.keys():
    #         print(f"{plot_var} not available in plot_bins. skipping!")
    #         continue
    #     xmin, xmax, _ = plot_bins[plot_var]["binning_linspace"]
    #     xlabel = plot_bins[plot_var].get("xlabel").replace("$","")
    #     dy100To200 = ak.to_numpy(events_100To200[var_reduced])
    #     dy100To200_wgt = ak.to_numpy(events_100To200.wgt_nominal)
    #     # dy100To200_wgt = np.sign(dy100To200_wgt)
    #     dy_vbf = ak.to_numpy(events_vbf[var_reduced])
    #     dy_vbf_wgt = ak.to_numpy(events_vbf.wgt_nominal)
    #     # dy_vbf_wgt = np.sign(dy_vbf_wgt)
    #     save_fname = f"DY2018UL_{var}_"
    #     plot_histograms_pyroot(dy100To200, dy_vbf, dy100To200_wgt, dy_vbf_wgt,nbins=64, xmin=xmin, xmax=xmax, xlabel=xlabel, save_fname=save_fname, normalize=False)
    # my old stitching code end ------------------------------------------

    # stage1 plot
    save_fname = "DY2018UL_THStackSignalFitReg_stage1"
    plotVarsTHStack(events_100To200, events_vbf, variables2plot, plot_bins, save_fname=save_fname)


    # VBF cat plot
    events_100To200 = applyVBF_cutV1(events_100To200)
    events_vbf = applyVBF_cutV1(events_vbf)
    save_fname = "DY2018UL_THStackSignalFitReg_VbfCat"
    plotVarsTHStack(events_100To200, events_vbf, variables2plot, plot_bins, save_fname=save_fname)
    
    

        
    
