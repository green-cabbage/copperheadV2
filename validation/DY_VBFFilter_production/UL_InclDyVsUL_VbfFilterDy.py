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

plt.style.use(hep.style.CMS)

"""
This code prints ggH/VBF channel yields after applying category cuts
"""

def applyVBF_cutV1(events):
    btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
    vbf_cut = (events.jj_mass_nominal > 700) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
    # vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
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

def applyGGH_cutV1(events):
    btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
    vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
    vbf_cut = ak.fill_none(vbf_cut, value=False)
    dimuon_mass = events.dimuon_mass
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
    dimuon_mass = events.dimuon_mass
    ggH_filter = (
        ~vbf_cut & 
        ~btag_cut # btag cut is for VH and ttH categories
    )
    return events[ggH_filter]

def applyGGH_noJetPt(events):
    btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
    vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5)
    vbf_cut = ak.fill_none(vbf_cut, value=False)
    dimuon_mass = events.dimuon_mass
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


def plot_normalized_histograms_pyroot(dy100To200, dy_vbf, dy100To200_wgt, dy_vbf_wgt, nbins=50, xmin=None, xmax=None, title="2018 UL", xlabel="Observable", save_fname = "normalized_hist_signWgt"):

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
    h1.Scale(1/h1.Integral())
    h2.Scale(1/h2.Integral())

    # Set styles
    h1.SetLineColor(ROOT.kBlue)
    h2.SetLineColor(ROOT.kRed)
    h1.SetLineWidth(2)
    h2.SetLineWidth(2)

    h1.GetXaxis().SetTitle(xlabel)
    h1.GetYaxis().SetTitle("A.U.")
    h1.SetTitle(title)

    # Draw
    c = ROOT.TCanvas("c", "c", 800, 600)
    h1.Draw("E")
    h2.Draw("E SAME")

    # Legend
    legend = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    legend.AddEntry(h1, "DY 100-200", "l")
    legend.AddEntry(h2, "DY VBF", "l")
    legend.Draw()

    c.SetGrid()
    c.Update()
    c.Draw()
    c.SaveAs(f"plots/{save_fname}.pdf")

    return c, h1, h2  # Useful if you want to save or manipulate further
 

if __name__ == "__main__":
    client =  Client(n_workers=40,  threads_per_worker=1, processes=True, memory_limit='8 GiB') 

    V1_fields_2compute = [
        "wgt_nominal",
        "nBtagLoose_nominal",
        "nBtagMedium_nominal",
        "mu1_pt",
        "mu2_pt",
        "mu1_eta",
        "mu2_eta",
        "mu1_phi",
        "mu2_phi",
        "dimuon_pt",
        "dimuon_eta",
        "dimuon_phi",
        "dimuon_mass",
        "jet1_phi_nominal",
        "jet1_pt_nominal",
        "jet2_pt_nominal",
        "jet2_phi_nominal",
        "jet1_eta_nominal",
        "jet2_eta_nominal",
        "jj_mass_nominal",
        "jj_dEta_nominal",
        "event",
        "njets_nominal",
        # "run",
        # "event",
        # "luminosityBlock",
    ]

    label="vbf_dy_validationMay30_2025"

    year = "2018"
    load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{label}/stage1_output/{year}/f1_0"
    
    target_chunksize = 150_000
    
    
    # dy 100To200
    load_path_100To200 = f"{load_path}/dy_M-100To200"
    events_100To200 = dak.from_parquet(f"{load_path_100To200}/*/*.parquet")
    events_100To200 = events_100To200.repartition(rows_per_partition=target_chunksize)
    events_100To200 = ak.zip({field: events_100To200[field] for field in V1_fields_2compute})
    events_100To200 = filterRegion(events_100To200, region="signal")
    events_100To200 = applyVBF_cutV1(events_100To200)
    events_100To200 = events_100To200.compute()
    
    # vbf-filter
    load_path_vbf = f"{load_path}/dy_m105_160_vbf_amc"
    events_vbf = dak.from_parquet(f"{load_path_vbf}/*/*.parquet")
    events_vbf = events_vbf.repartition(rows_per_partition=target_chunksize)
    events_vbf = ak.zip({field: events_vbf[field] for field in V1_fields_2compute})
    events_vbf = filterRegion(events_vbf, region="signal")
    events_vbf = applyVBF_cutV1(events_vbf)
    events_vbf = events_vbf.compute()

    fields2plot = ["mu1_eta", "mu2_eta","mu1_pt", "mu2_pt","mu1_phi", "mu2_phi", "dimuon_mass","dimuon_pt","dimuon_eta","dimuon_phi"]
    with open("plot_settings.json", "r") as file:
        plot_bins = json.load(file)
    for field in fields2plot:
        xmin, xmax, _ = plot_bins[field]["binning_linspace"]
        xlabel = plot_bins[field].get("xlabel").replace("$","")
        dy100To200 = ak.to_numpy(events_100To200[field])
        dy100To200_wgt = ak.to_numpy(events_100To200.wgt_nominal)
        dy100To200_wgt = np.sign(dy100To200_wgt)
        dy_vbf = ak.to_numpy(events_vbf[field])
        dy_vbf_wgt = ak.to_numpy(events_vbf.wgt_nominal)
        dy_vbf_wgt = np.sign(dy_vbf_wgt)
        save_fname = f"DY2018UL_{field}"
        plot_normalized_histograms_pyroot(dy100To200, dy_vbf, dy100To200_wgt, dy_vbf_wgt,nbins=64, xmin=xmin, xmax=xmax, xlabel=xlabel, save_fname=save_fname)
    
