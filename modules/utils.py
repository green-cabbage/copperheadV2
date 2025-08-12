import logging
from rich.logging import RichHandler
from rich.console import Console
from typing import Optional
import os
import sys
import awkward as ak
import ROOT as rt
import ROOT
import numpy as np
import matplotlib.pyplot as plt

LOGGER_NAME = "CopperHead"
NO_GIT_INFO_AVAILABLE = "No git info available"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ColorLogFormatter(logging.Formatter):
     """A class for formatting colored logs.
     Reference: https://stackoverflow.com/a/70796089/2302094
     """

     # FORMAT = "%(prefix)s%(msg)s"
    #  FORMAT = "\n[%(levelname)s] - [%(filename)s:#%(lineno)d] - %(prefix)s%(levelname)s - %(message)s %(suffix)s\n"
     FORMAT = "\n{}[%(levelname)5s] - [%(filename)s:#%(lineno)d] - [%(funcName)s; %(module)s]{} - %(prefix)s%(message)s %(suffix)s\n".format(
         bcolors.HEADER, bcolors.ENDC
     )
    #  FORMAT = "\n%(asctime)s - [%(filename)s:#%(lineno)d] - %(prefix)s%(levelname)s - %(message)s %(suffix)s\n"

     LOG_LEVEL_COLOR = {
         "DEBUG": {'prefix': bcolors.OKBLUE, 'suffix': bcolors.ENDC},
         "INFO": {'prefix': bcolors.OKGREEN, 'suffix': bcolors.ENDC},
         "WARNING": {'prefix': bcolors.WARNING, 'suffix': bcolors.ENDC},
         "CRITICAL": {'prefix': bcolors.FAIL, 'suffix': bcolors.ENDC},
         "ERROR": {'prefix': bcolors.FAIL+bcolors.BOLD, 'suffix': bcolors.ENDC+bcolors.ENDC},
     }

     def format(self, record):
         """Format log records with a default prefix and suffix to terminal color codes that corresponds to the log level name."""
         if not hasattr(record, 'prefix'):
             record.prefix = self.LOG_LEVEL_COLOR.get(record.levelname.upper()).get('prefix')

         if not hasattr(record, 'suffix'):
             record.suffix = self.LOG_LEVEL_COLOR.get(record.levelname.upper()).get('suffix')

         formatter = logging.Formatter(self.FORMAT, datefmt='%m/%d/%Y %I:%M:%S %p' )
         return formatter.format(record)

logger = logging.getLogger(LOGGER_NAME) # need to give it a name, otherwise *way* too much info gets printed out from e.g. numba
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(ColorLogFormatter())

# Set up stream handler (for stdout)
formatter = logging.Formatter("%(message)s")
stream_handler = RichHandler(show_time=False, rich_tracebacks=True,tracebacks_word_wrap=False)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

def ifPathExists(load_path):
    if not os.path.exists(load_path):
        logger.error(f"Path: {load_path} does not exists")
        sys.exit()
    else:
        logger.info(f"Path exists: {load_path}")

def get_git_info():
    """Get the current git commit hash, branch name, and the difference between the current version of the code and the last commit.
    Returns:
        tuple: A tuple containing the commit hash, branch name, and the difference.
    """
    try:
        import subprocess
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "diff"], text=True).strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Subprocess error while getting git info: {e}")
        commit_hash, branch_name, diff = None, None, None
    except Exception as e:
        logger.error(f"Unexpected error while getting git info: {e}")
        commit_hash, branch_name, diff = None, None, None

    return commit_hash, branch_name, diff

def get_git_info_str():
    """Get the current git commit hash, branch name, and the difference as a string.
    Returns:
        str: A string containing the commit hash, branch name, and the difference.
    """
    commit_hash, branch_name, diff = get_git_info()
    if commit_hash is None or branch_name is None:
        return NO_GIT_INFO_AVAILABLE
    else:
        return f"Commit: {commit_hash}, Branch: {branch_name}, Diff: {diff}"

def fillSampleValues(events, sample_dict, sample: str, fields2load=None):
    """
    inputs:
    sample_dict: dictionary with sample name as keys, lazy dask awkward zip as values

    return
    computed sample_dict: dictionary with sample name as keys, eager dask awkward zip as values

    Description:
    takes lazy dask awkward zips from sample_dict and computes only fields specified in fields2load. This is then returned
    """
    # find which sample group sample_name belongs to
    if fields2load is None:
        fields2load = ["wgt_nominal", "BDT_score", "dimuon_mass", "subCategory_idx"]
    if sample in sample_dict.keys():
        # compute in parallel fields to load
        computed_zip = ak.zip({
            field : events[field] for field in fields2load
        }).compute()
        for field in fields2load:
            sample_dict[sample][field] = ak.to_numpy(computed_zip[field])
            
    else:
        print(f"sample {sample} not present in sample_dict!")

    return sample_dict



def getDimuMassBySubCat(sample_dict, sample="", nSubCats=5):
    dimuon_mass = sample_dict[sample]["dimuon_mass"]
    wgt_nominal = sample_dict[sample]["wgt_nominal"]
    subCat_ixs = sample_dict[sample]["subCategory_idx"]
    dict_by_subCat = {}
    for target_subCat in range(nSubCats):
        subCat_filter = target_subCat == subCat_ixs
        dimuon_mass_subCat = dimuon_mass[subCat_filter]
        wgt_nominal_subCat = wgt_nominal[subCat_filter]
        subCat_dict = {
            "dimuon_mass" : dimuon_mass_subCat,
            "wgt_nominal" : wgt_nominal_subCat,
        }
        dict_by_subCat[target_subCat] = subCat_dict
    return dict_by_subCat



def ensure_compacted(df, compacted_dir_path, exception_samples=[]):
    compacted_dir = compacted_dir_path
    if not os.path.exists(compacted_dir):
        print(f"Compacted path for {compacted_dir_path} not found. Creating compacted dataset...")
        # Read original data
        # orig_path = orig_dir_path
        # df = dd.read_parquet(orig_path)
        # print(f"Original rows = {len(df)}")
        # if len(df) == 0:
            # print(f"WARNING: {orig_path} is empty!")
        # Repartition and write to compacted path if not in exception samples
        keep_npartitions = False
        for sample_name in exception_samples:
            if sample_name in compacted_dir_path:
                keep_npartitions = True # if sample name is mentioned in the new directory path, then skip reparitioning. Typically this is for signal MC samples that have high efficiency in our selection in signal region
        if not keep_npartitions:
            target_chunksize = 250_000
            df_repart = df.repartition(rows_per_partition=target_chunksize)
        else:
            df_repart = df
            
        df_repart.to_parquet(compacted_dir)
        print(f"Compacted dataset created at {compacted_dir}")
    else:
        # Optionally, could check if compacted_dir is empty or incomplete, but skipping for now
        pass




def filterRegion(events, region="h-peak"):
    dimuon_mass = events.dimuon_mass
    if region =="h-peak":
        region = (dimuon_mass > 115) & (dimuon_mass < 135)
    elif region =="h-sidebands":
        region = ((dimuon_mass > 110) & (dimuon_mass < 115)) | ((dimuon_mass > 135) & (dimuon_mass < 150))
    elif region =="signal":
        region = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
    elif region =="z-peak":
        region = (dimuon_mass >= 70) & (dimuon_mass <= 110.0)

    events = events[region]
    return events


def fillEventNans(events, category="vbf"):
    """
    """
    if category == "vbf":
        for field in events.fields:
            if "phi" in field:
                events[field] = ak.fill_none(events[field], value=-10) # we're working on a DNN, so significant deviation may be warranted
            else: # for all other fields (this may need to be changed)
                events[field] = ak.fill_none(events[field], value=0)
    else:
        print("ERROR: unsupported category!")
        raise ValueError
    return events


def convertVectorType4D(vector, vector_name):
    new_vector = ak.zip(
        {
            "pt": vector.pt,
            "eta": vector.eta,
            "phi": vector.phi,
            "mass": vector.mass,
            "charge": vector.charge,
        },
        with_name=vector_name,
        behavior=vector.behavior,
    )


def getNewRangeHist(hist2copy: rt.TH1D, new_hist_name: str, xlow_new: float, xhigh_new: float):
    """
    """
    # get binwise content
    # Initialize result dictionary
    bin_dict = {"xlow": [], "xhigh": [], "content": []}
    
    # Fill dictionary with bin info
    hist = hist2copy
    for i in range(1, hist.GetNbinsX() + 1):
        xlow = hist.GetBinLowEdge(i)
        xhigh = xlow + hist.GetBinWidth(i)
        content = hist.GetBinContent(i)
        if (xlow >= xlow_new) and (xhigh <= xhigh_new):
            bin_dict["xlow"].append(xlow)
            bin_dict["xhigh"].append(xhigh)
            bin_dict["content"].append(content)
        
    # print(bin_dict)
    new_nbins = len(bin_dict["content"])

    new_hist = ROOT.TH1D(new_hist_name, new_hist_name, new_nbins, xlow_new, xhigh_new)
    for i in range(1, new_hist.GetNbinsX() + 1):
        content = bin_dict["content"][i-1]
        new_hist.SetBinContent(i, content)

    # # sanity check
    # for i in range(1, new_hist.GetNbinsX() + 1):
    #     xlow = new_hist.GetBinLowEdge(i)
    #     xhigh = xlow + new_hist.GetBinWidth(i)
    #     content = new_hist.GetBinContent(i)
    #     print(f"{new_hist_name} bin{i} {(xlow, xhigh, content)}")
    # print(f"new_nbins: {new_nbins}")
    # raise ValueError
    return new_hist

def getGOF_KS(x: rt.RooRealVar, data: rt.RooDataHist, pdf: rt.RooAbsPdf, cat_name:str, save_path:str):
    """
    Get KS value for specific value
    """
    # print(save_path)
    # raise ValueError
    nbins = x.getBins()
    var_name = x.GetName()
    # # Generate toy dataset
    # data = pdf.generate(ROOT.RooArgSet(x), 1000)
    hist_data_orig = data.createHistogram(var_name).Clone("clone")# clone it just in case
    # Create a histogram of the PDF
    hist_pdf_orig = pdf.createHistogram(var_name, nbins)


    plot_line_width = 0.5
    # -------------------------------------------
    # Sanity check: plot the pdf and cdf of signal region PDF
    # -------------------------------------------
    # data_counts = np.array([hist_data_orig.GetBinContent(i) for i in range(1, hist_data_orig.GetNbinsX()+1)])
    pdf_counts = np.array([hist_pdf_orig.GetBinContent(i) for i in range(1, hist_pdf_orig.GetNbinsX()+1)])
    # print(f"data_counts: {data_counts}")
    # print(f"pdf_counts: {pdf_counts}")
    pdf_cdf = np.cumsum(pdf_counts) / np.sum(pdf_counts)
    
    bin_centers = []
    hist = hist_pdf_orig
    for i in range(1, hist.GetNbinsX() + 1):  # 1-based bin index
        center = hist.GetBinCenter(i)
        bin_centers.append(center)
    bin_centers = np.array(bin_centers)
    plt.plot(bin_centers, pdf_cdf, label='PDF CDF')
    plt.xlabel('mass')
    plt.ylabel('')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/GoF_cdfs_SignalFitRange_{cat_name}.pdf")
    plt.clf()
    
    # Draw the normalized pdf histogram:
    plt.plot(bin_centers, pdf_counts/np.sum(pdf_counts), label='PDF PDF')
    plt.xlabel('mass')
    plt.ylabel('')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/GoF_pdfs_SignalFitRange_{cat_name}.pdf")
    plt.clf()


    # -------------------------------------------
    # Do KS test
    # -------------------------------------------
    return_dict = {}
    for test_range_name in ["loSB", "hiSB"]:
        xlow, xhigh = list(x.getRange(test_range_name))
        data_hist_new = getNewRangeHist(hist_data_orig, f"data_hist_{test_range_name}", xlow, xhigh)
        pdf_hist_new = getNewRangeHist(hist_pdf_orig, f"pdf_hist_{test_range_name}", xlow, xhigh)
        data_counts = np.array([data_hist_new.GetBinContent(i) for i in range(1, data_hist_new.GetNbinsX()+1)])
        pdf_counts = np.array([pdf_hist_new.GetBinContent(i) for i in range(1, pdf_hist_new.GetNbinsX()+1)])
        # print(f"data_counts: {data_counts}")
        # print(f"pdf_counts: {pdf_counts}")
        # nevents=np.sum(data_counts)
        data_cdf = np.cumsum(data_counts) / np.sum(data_counts)
        pdf_cdf = np.cumsum(pdf_counts) / np.sum(pdf_counts)
        ks_statistic = np.max(np.abs(data_cdf - pdf_cdf))
        print(f"ks_statistic {cat_name} {test_range_name}: {ks_statistic}")
        nevents = data_hist_new.Integral()
        
        return_dict[test_range_name] = {
            "ks_statistic": ks_statistic,
            "nevents" : nevents,
                                       }
        
    
        # Draw the cdf histogram
        # Extract bin centers (excluding underflow/overflow)
        bin_centers = []
        hist = data_hist_new
        for i in range(1, hist.GetNbinsX() + 1):  # 1-based bin index
            center = hist.GetBinCenter(i)
            bin_centers.append(center)
        bin_centers = np.array(bin_centers)
        plt.plot(bin_centers, data_cdf, label='Data CDF', linewidth=plot_line_width)
        plt.plot(bin_centers, pdf_cdf, label='PDF CDF', linewidth=plot_line_width)
        plt.xlabel('mass')
        plt.ylabel('')
        # plt.title('Simple NumPy Plot')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/GoF_cdfs_{test_range_name}_{cat_name}.pdf")
        plt.clf()

        # Draw the normalized pdf histogram:
        plt.plot(bin_centers, data_counts/np.sum(data_counts), label='Data PDF', linewidth=plot_line_width)
        plt.plot(bin_centers, pdf_counts/np.sum(pdf_counts), label='PDF PDF', linewidth=plot_line_width)
        plt.xlabel('mass')
        plt.ylabel('')
        # plt.title('Simple NumPy Plot')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/GoF_pdfs_{test_range_name}_{cat_name}.pdf")
        plt.clf()
        
    return return_dict

def applyRegionCatCuts(events, category: str, region_name: str, process: str, variation:str, do_vbf_filter_study: bool):
    # do mass region cut
    mass = events.dimuon_mass
    z_peak = ((mass > 70) & (mass < 110))
    h_sidebands =  ((mass > 110) & (mass < 115)) | ((mass > 135) & (mass < 150))
    h_peak = ((mass > 115) & (mass < 135))
    if region_name == "signal":
        region = h_sidebands | h_peak
    elif region_name == "h-peak":
        region = h_peak 
    elif region_name == "h-sidebands":
        region = h_sidebands 
    elif region_name == "z-peak":
        region = z_peak 
    else: 
        print("ERROR: acceptable region!")
        raise ValueError
    
    # do category cut
    if category == "nocat": 
        # print("nocat mode!")
        prod_cat_cut =  ak.ones_like(region, dtype="bool")
        # prod_cat_cut = ak.fill_none(events[f"jj_mass_{variation}"] > 400, value=False)
        # prod_cat_cut = prod_cat_cut & ak.fill_none(events[f"jet1_pt_{variation}"] > 35, value=False)
        
    else: # VBF or ggH
        btagLoose_filter = ak.fill_none((events[f"nBtagLoose_{variation}"] >= 2), value=False)
        btagMedium_filter = ak.fill_none((events[f"nBtagMedium_{variation}"] >= 1), value=False) & ak.fill_none((events[f"njets_{variation}"] >= 2), value=False)
        btag_cut = btagLoose_filter | btagMedium_filter
        # vbf_cut = ak.fill_none(events.vbf_cut, value=False) # in the future none values will be replaced with False
        vbf_cut = (events[f"jj_mass_{variation}"] > 400) & (events[f"jj_dEta_{variation}"] > 2.5) & (events[f"jet1_pt_{variation}"] > 35) 
        vbf_cut = ak.fill_none(vbf_cut, value=False)
        if category == "vbf":
            # print("vbf mode!")
            prod_cat_cut =  vbf_cut
            prod_cat_cut = prod_cat_cut & ~btag_cut # btag cut is for VH and ttH categories
        elif category == "ggh":
            # print("ggH mode!")
            prod_cat_cut =  ~vbf_cut 
            prod_cat_cut = prod_cat_cut & ~btag_cut # btag cut is for VH and ttH categories
        else:
            print("Error: invalid category option!")
            raise ValueError

    if do_vbf_filter_study:
        if "dy_" in process:
            is_vbf_filter = ("dy_VBF_filter" in process) or (process =="dy_m105_160_vbf_amc")
            if is_vbf_filter:
                # print(f"applying VBF filter cut on: {process}")
                
                vbf_filter = ak.fill_none((events.gjj_mass > 350), value=False)
                prod_cat_cut =  (prod_cat_cut  
                            & vbf_filter
                )
            else:
                # print(f"cutting off inclusive dy: {process}")
                vbf_filter = ak.fill_none((events.gjj_mass > 350), value=False) 
                prod_cat_cut =  (
                    prod_cat_cut  
                    & ~vbf_filter 
                )
        else:
            # print(f"no extra processing for {process}")
            pass
    
    category_selection = (
        prod_cat_cut & 
        region 
    )
    # filter events fro selected category
    
    # print(f"len(events) {process} b4 selection: {len(events)}")
    events = events[category_selection]
    return events