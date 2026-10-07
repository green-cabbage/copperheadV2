import awkward as ak
import dask_awkward as dak
import dask
import argparse
import sys
import os
import numpy as np
import json
import yaml
from collections import OrderedDict
from modules.selection import filterRegion
import glob
import pickle
from pathlib import Path

import logging
from modules.utils import logger
from modules import selection

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Add it to sys.path
sys.path.insert(0, parent_dir)
# Now you can import your module
from src.lib.histogram.plotting import plotDataMC_compare

# PDF and alpha_s member weights written by stage1 (one histogram per member in
# stage2). Kept in step with stage3/make_templates.py, which defines the same
# constants but cannot be imported here because it pulls in ROOT.
PDF_MEMBER_PREFIX = "wgt_pdfMemberHessEig"
PDF_N_EIGENVECTOR_MEMBERS = 100
PDF_UNC_COMBINATION = "hessian"  # "hessian" (divisor 1) or "rms" (divisor N-1)
# LHEPdfWeight[101] (alpha_s = 0.116) and [102] (alpha_s = 0.120), in that order.
PDF_ALPHA_S_MEMBERS = ("wgt_pdfAlphaS101_up", "wgt_pdfAlphaS102_up")
ALPHA_S_UNC_SCALE = 1.0
# Estimators built from those members instead of an explicit up/down label pair.
MEMBER_ESTIMATORS = ("pdf_hessian", "alpha_s", "pdf_alpha_s")
# Reserved scalar keys in the systematics YAML: settings, not nuisance groups.
RESERVED_CONFIG_KEYS = ("variable",)
# Plotted observable when the YAML names none. The stage-2 score histograms are
# the only ones whose edges come from the DNN binning config rather than the
# vbf plot settings, so this name also selects that path.
DEFAULT_PLOT_VARIABLE = "DNN_score"
VBF_PLOT_SETTINGS = "src/lib/histogram/plot_settings_vbfCat_MVA_input.json"


def load_systematics_config(path):
    """Read explicit nuisance labels; no inference or silent missing variations."""
    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def unique_mapping(loader, node):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str) or key in mapping:
                raise ValueError(f"Non-string or duplicate YAML key: {key!r}")
            mapping[key] = loader.construct_object(value_node)
        return mapping

    UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
    with open(path) as stream:
        config = yaml.load(stream, Loader=UniqueKeyLoader)
    return split_config_options(config)


def split_config_options(config):
    """Separate the reserved settings from the nuisance groups: (groups, options)."""
    if not isinstance(config, dict) or not config:
        raise ValueError("Systematics YAML must be a nonempty mapping")
    options = {key: config[key] for key in RESERVED_CONFIG_KEYS if key in config}
    groups = {key: value for key, value in config.items()
              if key not in RESERVED_CONFIG_KEYS}
    variable = options.get("variable", DEFAULT_PLOT_VARIABLE)
    if not isinstance(variable, str) or not variable.strip():
        raise ValueError(f"'variable' must be a nonempty string, not {variable!r}")
    options["variable"] = variable
    return validate_systematics_config(groups), options


def validate_systematics_config(config):
    """Return the legacy list or an ordered mapping of named uncertainty groups."""
    if not isinstance(config, dict) or not config:
        raise ValueError("Systematics YAML must be a nonempty mapping of group names to lists")
    if "systematics" in config and set(config) != {"systematics"}:
        raise ValueError("Do not mix the legacy 'systematics' list with named groups")
    for name, entries in config.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(entries, list):
            raise ValueError("Each systematic group needs a nonempty name and a list")
    entries = [entry for group in config.values() for entry in group]
    names, labels, estimators = set(), set(), set()
    for entry in entries:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError("Each systematic needs a name")
        if set(entry) - {"name", "up", "down", "groups", "type"}:
            raise ValueError(f"Unknown systematic configuration keys: {entry}")
        kind = entry.get("type", "updown")
        if kind not in ("updown",) + MEMBER_ESTIMATORS:
            raise ValueError(f"{entry['name']}: unknown systematic type {kind!r}")
        # A member estimator reads fixed weight columns, so an up/down pair here
        # would be silently ignored rather than applied.
        if kind == "updown":
            if not {"up", "down"} <= entry.keys():
                raise ValueError("Each systematic needs name, up, and down")
        elif entry.keys() & {"up", "down"}:
            raise ValueError(f"{entry['name']}: '{kind}' reads member weights; drop up and down")
        string_keys = ("name", "up", "down") if kind == "updown" else ("name",)
        if any(not isinstance(entry[key], str) or not entry[key].strip()
               for key in string_keys):
            raise ValueError("Systematic names and variation labels must be nonempty strings")
        if entry["name"] in names:
            raise ValueError(f"Duplicate systematic: {entry['name']}")
        names.add(entry["name"])
        if kind == "updown":
            pair = {entry["up"], entry["down"]}
            if len(pair) != 2 or "nominal" in pair or pair & labels:
                raise ValueError(f"Duplicate or nominal variation labels: {entry['name']}")
            labels.update(pair)
        else:
            # Repeating one estimator over the same MC groups would count it twice.
            estimator = (kind, tuple(entry.get("groups", ())))
            if estimator in estimators:
                raise ValueError(f"Duplicate {kind} over the same MC groups: {entry['name']}")
            estimators.add(estimator)
        if "groups" in entry:
            groups = entry["groups"]
            if (not isinstance(groups, list) or not groups
                    or any(not isinstance(group, str) or not group for group in groups)
                    or len(set(groups)) != len(groups)):
                raise ValueError(f"groups must be a nonempty list of unique group names: {entry['name']}")
    return config["systematics"] if "systematics" in config else config


def build_prediction_uncertainty(hist_groups, systematics, region, category,
                                 background_groups, scope="background", bands="stat+syst"):
    """Sum each nuisance coherently across samples, then combine envelopes."""
    if scope not in {"background", "background+signal"}:
        raise ValueError(f"Unknown uncertainty scope: {scope}")
    if bands not in {"stat+syst", "syst-only", "both"}:
        raise ValueError(f"Unknown uncertainty bands: {bands}")
    named_groups = systematics if isinstance(systematics, dict) else None
    # Validate programmatic calls as well as YAML input, including cross-group reuse.
    validate_systematics_config(named_groups if named_groups is not None else {"systematics": systematics})
    if named_groups is not None:
        systematics = [entry for entries in named_groups.values() for entry in entries]
    group_for_nuisance = {entry["name"]: name for name, entries in (named_groups or {}).items()
                          for entry in entries}
    known_groups = set(hist_groups) - {"data"}
    for nuisance in systematics:
        unknown = set(nuisance.get("groups", [])) - known_groups
        if unknown:
            raise ValueError(f"{nuisance['name']}: unknown MC groups {sorted(unknown)}")
    # A combined pdf_alpha_s beside its own components is a legitimate comparison
    # plot, but its contribution to the grey total is then counted twice.
    combined = {tuple(n.get("groups", ())) for n in systematics
                if n.get("type") == "pdf_alpha_s"}
    split = {tuple(n.get("groups", ())) for n in systematics
             if n.get("type") in ("pdf_hessian", "alpha_s")}
    if combined & split:
        logger.warning(
            "pdf_alpha_s overlaps pdf_hessian/alpha_s over the same MC groups: the grey "
            "stat+syst total double counts them. Read the dashed boundaries, not the total."
        )
    selected = list(background_groups)
    if scope == "background+signal":
        selected += [group for group in ("ggH", "VBF") if group in hist_groups]
    edges = None

    def project(histogram, variation, value, context):
        nonlocal edges
        if variation not in histogram.axes["variation"]:
            raise ValueError(f"{context}: missing variation {variation!r}")
        projected = histogram[{"region": region, "channel": category,
                               "variation": variation, "val_sumw2": value}]
        if projected.ndim != 1:
            raise ValueError(f"{context}: expected one score axis")
        sample_edges = np.asarray(projected.axes[0].edges)
        if edges is None:
            edges = sample_edges.copy()
        elif not np.array_equal(edges, sample_edges):
            raise ValueError(f"{context}: incompatible score binning")
        values = np.asarray(projected.values(), dtype=float)
        if not np.all(np.isfinite(values)) or (value == "sumw2" and np.any(values < 0)):
            raise ValueError(f"{context}: invalid {value} values")
        return values

    samples = []
    for group in selected:
        for index, histogram in enumerate(hist_groups[group]):
            context = f"{group} sample {index}"
            nominal = project(histogram, "nominal", "value", context)
            variance = project(histogram, "nominal", "sumw2", context)
            samples.append((group, histogram, nominal, variance, context))
    if not samples:
        raise ValueError("No MC histograms for the selected uncertainty scope")
    prediction = np.sum([sample[2] for sample in samples], axis=0)
    stat_variance = np.sum([sample[3] for sample in samples], axis=0)
    up_squared = np.zeros_like(prediction)
    down_squared = np.zeros_like(prediction)
    group_squared = {name: {"up": np.zeros_like(prediction), "down": np.zeros_like(prediction)}
                     for name in (named_groups or {})}

    def coherent_shift(nuisance, variation):
        """One variation's deviation from nominal, summed over the affected samples."""
        shift = np.zeros_like(prediction)
        for group, histogram, nominal, _, context in samples:
            if "groups" not in nuisance or group in nuisance["groups"]:
                shifted = project(histogram, variation, "value",
                                  f"{nuisance['name']}, {context}")
                shift += shifted - nominal
        return shift

    def pdf_member_squared(nuisance):
        """arXiv:2203.05506 Eq. (6.5): quadrature over the eigenvector members."""
        if PDF_UNC_COMBINATION not in ("hessian", "rms"):
            raise ValueError(f"PDF_UNC_COMBINATION is {PDF_UNC_COMBINATION!r}; "
                             f"expected 'hessian' or 'rms'")
        members = [f"{PDF_MEMBER_PREFIX}{member:03d}_up"
                   for member in range(PDF_N_EIGENVECTOR_MEMBERS)]
        divisor = 1.0 if PDF_UNC_COMBINATION == "hessian" else float(len(members) - 1)
        return np.sum([coherent_shift(nuisance, member) ** 2 for member in members],
                      axis=0) / divisor

    def alpha_s_squared(nuisance):
        """PDF4LHC15 Eqs. (27)-(28): half-difference of the 0.120 and 0.116 members."""
        low, high = (coherent_shift(nuisance, member) for member in PDF_ALPHA_S_MEMBERS)
        return (ALPHA_S_UNC_SCALE * (high - low) / 2.0) ** 2

    for nuisance in systematics:
        kind = nuisance.get("type", "updown")
        if kind == "updown":
            shifts = [coherent_shift(nuisance, nuisance[direction])
                      for direction in ("up", "down")]
            up = np.maximum.reduce([shifts[0], shifts[1], np.zeros_like(prediction)]) ** 2
            down = np.maximum.reduce([-shifts[0], -shifts[1], np.zeros_like(prediction)]) ** 2
        else:
            # Member estimators are symmetric by construction, so up equals down.
            squared = np.zeros_like(prediction)
            if kind in ("pdf_hessian", "pdf_alpha_s"):
                squared = squared + pdf_member_squared(nuisance)
            if kind in ("alpha_s", "pdf_alpha_s"):
                squared = squared + alpha_s_squared(nuisance)
            up = down = squared
        up_squared += up
        down_squared += down
        if named_groups is not None:
            group_squared[group_for_nuisance[nuisance["name"]]]["up"] += up
            group_squared[group_for_nuisance[nuisance["name"]]]["down"] += down
    return {"nominal": prediction, "sumw2": stat_variance,
            "syst_up": np.sqrt(up_squared), "syst_down": np.sqrt(down_squared),
            "binning": edges, "scope": scope, "bands": bands,
            "systematic_groups": {name: {"syst_up": np.sqrt(values["up"]),
                                         "syst_down": np.sqrt(values["down"])}
                                  for name, values in group_squared.items()}}


def plotStage2DNN_score(hist_dict_bySampleGroup, var, plot_settings, full_save_path, region_name, category, do_logscale=True, binning=None, lumi="", status="Private", systematics=None, uncertainty_scope="background", uncertainty_bands="stat+syst", systematics_config=None):
    """
    hist_dict_bySampleGroup : dictionary with sample group (data, DY, VV) as keys and list of relecant hep histograms as values
    """
    # logger.info(f"hist_dict_bySampleGroup: {hist_dict_bySampleGroup}")

    data_dict = {}
    bkg_MC_dict = {}
    sig_MC_dict = {}
    plot_var = getPlotVar(var)
    if plot_var not in plot_settings.keys():
        logger.info(f"variable {var} not configured in plot settings!")
        return
    for group_name, sample_hist_l  in hist_dict_bySampleGroup.items():
        logger.info(f"{group_name} hist_list types: {[type(h) for h in sample_hist_l]}")
        logger.info(f"{group_name} hist_list len: {len(sample_hist_l)}")
        if len(sample_hist_l) == 0:
            logger.info(f"No histograms found for {group_name}, skipping!")
            continue

        logger.debug(f"Combining histograms for {group_name}...")
        logger.debug(f"Sample histograms keys: {[h.axes.name for h in sample_hist_l]}")

        for i, h in enumerate(sample_hist_l):
            logger.debug(f"Histogram {i} axes: {h.axes.name}")
            for axis in h.axes:
                logger.debug(f"  Axis: {axis.name}, type: {type(axis)}, labels: {getattr(axis, 'categories', 'None')}, edges: {getattr(axis, 'edges', 'None')}")

        # logger.info("sample_hist_l compute:")
        # logger.info(dask.compute(sample_hist_l))
        # logger.info("=" * 50 )

        sample_hist = sum(sample_hist_l)
        to_project_setting = {
            "region" : region_name,
            "channel" : category,
            "variation" : "nominal",
            # "sample_group": group_name,
        }
        logger.debug(f"to_project_setting: {to_project_setting}")
        logger.debug(f"sample_hist: {sample_hist}")

        #  Print/check the type of sample_hist and its keys
        logger.info(f"Type of sample_hist: {type(sample_hist)}")
        logger.info(f"Keys in sample_hist: {sample_hist.axes.name}")

        to_project_setting_val = to_project_setting.copy()
        logger.debug(f"to_project_setting_val: {to_project_setting_val}")
        to_project_setting_val["val_sumw2"] = "value"
        logger.debug(f"to_project_setting_val: {to_project_setting_val}")
        hist_val = sample_hist[to_project_setting_val].view()
        # ------------------------------------------------------
        to_project_setting_w2 = to_project_setting.copy()
        to_project_setting_w2["val_sumw2"] = "sumw2"
        hist_w2 = sample_hist[to_project_setting_w2].view()
        logger.info(f"to_project_setting: {to_project_setting}")
        logger.info(f"hist_val {group_name}: {hist_val}")
        logger.info(f"hist_w2 {group_name}: {hist_w2}")
        if np.sum(hist_val)==0 and systematics is None:
            logger.info(f"Empty hist from {group_name}. Skipping!")
            continue
        hist_dict = {
            "hist_arr" : hist_val,
            "hist_w2_arr": hist_w2
        }

        if "data" in group_name:
            if region_name != "h-peak":
                data_dict = hist_dict
            else: # keep data blinded
                data_dict = {key: np.zeros_like(value) for key, value in hist_dict.items()}
        elif group_name in {"ggH", "VBF"}: # signal
            sig_MC_dict[group_name] = hist_dict
        else: # bkg MC
            bkg_MC_dict[group_name] = hist_dict
    # order bkg_MC_dict in a specific way for plotting, smallest yielding process first:
    bkg_MC_order = ["VVV", "VV", "Ewk", "Top", "DYVBF", "DY","DYJ01", "DYJ2"]
    bkg_MC_dict = {process: bkg_MC_dict[process] for process in bkg_MC_order if process in bkg_MC_dict}
    logger.info(f"data_dict : {data_dict}")
    logger.info(f"bkg_MC_dict : {bkg_MC_dict}")
    logger.info(f"sig_MC_dict : {sig_MC_dict}")

    # -------------------------------------------------------
    # All data are prepped, now plot Data/MC histogram
    # -------------------------------------------------------
    # full_save_path = args.save_path+f"/{args.year}/mplhep/Reg_{region_name}/Cat_{args.category}/{args.label}"
    # logger.info(f"full_save_path: {full_save_path}")

    if not os.path.exists(full_save_path):
        os.makedirs(full_save_path)
    # tag = "Run2_nanoAODv12_AK8jets"
    dnn_tag = plot_var
    full_save_fname = f"{full_save_path}/{var}_{region_name}_{dnn_tag}.pdf"
    logger.info(f"full_save_fname: {full_save_fname}")
    # raise ValueError

    if binning is None:
        binning = np.linspace(*plot_settings[plot_var]["binning_linspace"])

    uncertainty_kwargs = {}
    if systematics is not None:
        # Snapshot the configuration before rendering; preserve source comments too.
        config_document = systematics if isinstance(systematics, dict) else {"systematics": systematics}
        config_bytes = (Path(systematics_config).read_bytes() if systematics_config
                        else yaml.safe_dump(config_document, sort_keys=False).encode())
        on_disk = yaml.safe_load(config_bytes)
        if isinstance(on_disk, dict):
            on_disk = {key: value for key, value in on_disk.items()
                       if key not in RESERVED_CONFIG_KEYS}
        if on_disk != config_document:
            raise ValueError("Systematics YAML changed since loading; refusing to save a mismatched copy")
        uncertainty = build_prediction_uncertainty(
            hist_dict_bySampleGroup, systematics, region_name, category,
            list(bkg_MC_dict), uncertainty_scope, uncertainty_bands,
        )
        if not np.array_equal(np.asarray(binning), uncertainty["binning"]):
            raise ValueError("Plot binning does not match stage2 histogram binning")
        suffix = f"_unc_{uncertainty_scope}_{uncertainty_bands}"
        full_save_fname = full_save_fname.replace(".pdf", f"{suffix}.pdf")
        uncertainty_kwargs = {
            "prediction_uncertainty": uncertainty,
            "extra_header_lines": [
                f"Uncertainty scope: {uncertainty_scope}; bands: {uncertainty_bands}",
                "Systematics: " + json.dumps(systematics),
            ],
        }

    plotDataMC_compare(
        binning,
        data_dict,
        bkg_MC_dict,
        full_save_fname.replace(".pdf", "_log.pdf"),
        sig_MC_dict=sig_MC_dict,
        title = "",
        x_title = plot_settings[plot_var].get("xlabel"),
        y_title = plot_settings[plot_var].get("ylabel"),
        lumi = lumi,
        status = status,
        log_scale = do_logscale,
        plot_ratio_range = "default", # options: "default" or "auto" or list with format [0.8, 1.2]
        **uncertainty_kwargs,
    )
    if systematics is not None:
        Path(full_save_fname.replace(".pdf", "_log.yaml")).write_bytes(config_bytes)
    plotDataMC_compare(
        binning,
        data_dict,
        bkg_MC_dict,
        full_save_fname,
        sig_MC_dict=sig_MC_dict,
        title = "",
        x_title = plot_settings[plot_var].get("xlabel"),
        y_title = plot_settings[plot_var].get("ylabel"),
        lumi = lumi,
        status = status,
        log_scale = False,
        plot_ratio_range = "default", # options: "default" or "auto" or list with format [0.8, 1.2]
        **uncertainty_kwargs,
    )
    if systematics is not None:
        Path(full_save_fname).with_suffix(".yaml").write_bytes(config_bytes)


def getPickledHist_byFname(pickled_filelist, load_path):
    return_dict = {}
    for fname in pickled_filelist:
        with open(fname, "rb") as f:
            hist = pickle.load(f)
        key_name = fname.replace(f"{load_path}", "").replace("_hist.pkl", "")
        # logger.info(f"key_name: {key_name}")
        return_dict[key_name] = hist

    return return_dict

def load_group_process_indicators(sample_config_path):
    """
    Build {plot_group_name: [process_name, ...]} indicators from a
    samples.yaml-style config (see configs/samples/samples.yaml), instead of
    a hardcoded list that silently drops any process added/renamed later
    (e.g. MiNNLO DY: dyTo2Mu_M-50_MiNNLO, dyTo2Mu_M-100to200_MiNNLO).

    Indicators are the union of a group's default `processes` and every
    year's `processes_per_year` override, across ALL years in the file --
    not just one requested year -- since a stage2 histogram directory can
    mix multiple years (e.g. --year run2 globs 2016preVFP/2016postVFP/2017/
    2018 together) and a process name means the same sample in any year.
    """
    with open(sample_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    def _union_group_processes(section):
        groups = (cfg.get(section) or {}).get("groups", {}) or {}
        out = {}
        for group_name, gcfg in groups.items():
            procs = set(gcfg.get("processes") or [])
            for year_procs in (gcfg.get("processes_per_year") or {}).values():
                procs.update(year_procs or [])
            out[group_name] = sorted(procs)
        return out

    bkg = _union_group_processes("background")
    sig = _union_group_processes("signal")

    return {
        # "data" isn't an MC sample in samples.yaml -- stage2 combines all
        # data-era files into one "data" histogram, so this stays literal.
        "data": ["data"],
        "ggH": sig.get("GGH", []),
        "VBF": sig.get("VBF", []),
        "DYVBF": bkg.get("DYVBF", []),
        "DY": bkg.get("DY", []),
        "Top": bkg.get("TT", []) + bkg.get("ST", []),
        "Ewk": bkg.get("EWK", []),
        "VV": bkg.get("VV", []),
        "VVV": bkg.get("VVV", []),
    }


def arrangeHist_bySampleGroup(pickled_hist_dict, sample_group_dict):
    """
    sample_group_dict: {plot_group_name: [process_name, ...]}, as built by
    load_group_process_indicators().
    """
    hist_bySampleGroup = {sample_group: [] for sample_group in sample_group_dict.keys()}
    for hist_name, hist_instance in pickled_hist_dict.items():
        # loop over hist_name and add them to the appropriate sample group
        for sample_group, name_indicators in sample_group_dict.items():
            for name_indicator in name_indicators:
                if name_indicator in hist_name:
                    hist_bySampleGroup[sample_group].append(hist_instance)
                    continue

    for sample_group, hist_l in hist_bySampleGroup.items():
        logger.info(f"{sample_group}, len:{len(hist_l)}")
        # check hist_l number of bins
        for i, h in enumerate(hist_l):
            logger.warning(f"  {i} : {h.axes.name}, bins: {[getattr(axis, 'edges', 'None') for axis in h.axes]}")
    return hist_bySampleGroup

def getPlotVar(var: str):
    """
    Helper function that removes the variations in variable name if they exist
    """
    if "_nominal" in var:
        plot_var = var.replace("_nominal", "")
    else:
        plot_var = var
    return plot_var

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-label",
        "--label",
        dest="label",
        default="Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar",
        action="store",
        help="label",
    )
    parser.add_argument(
    "-cat",
    "--category",
    dest="category",
    default="vbf",
    action="store",
    help="string value production category we're working on",
    )
    parser.add_argument(
        "-save",
        "--save_path",
        dest="save_path",
        default="validation/from_stage2/",
        action="store",
        help="string value production category we're working on",
    )
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="label",
    )
    parser.add_argument(
    "--load",
    dest="load_path",
    default="Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar",
    action="store",
    help="label",
    )
    parser.add_argument(
    "--mva_name",
    dest="mva_name",
    default="Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar",
    action="store",
    help="label",
    )
    parser.add_argument(
    "--vbf_filter_study",
    dest="do_vbf_filter_study",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Enable DY vs DY-VBF-filter study mode for grouping and output naming.",
    )
    parser.add_argument(
        "--sample-config",
        dest="sample_config",
        default="configs/samples/samples.yaml",
        help="Path to the sample configuration YAML file (same one run_stage2_vbf.py uses to resolve MC process groups).",
    )
    parser.add_argument(
    "-reg",
    "--region",
    dest="region",
    default="signal",
    action="store",
    help="region value to plot, available regions are: h_peak, h_sidebands, z_peak and signal (h_peak OR h_sidebands)",
    )
    parser.add_argument(
        "--log-level",
        default=logging.INFO,
        type=lambda x: getattr(logging, x),
        help="Configure the logging level.",
    )    
    parser.add_argument("--systematics-config", help="YAML list of nuisance up/down labels")
    parser.add_argument("--uncertainty-scope", choices=["background", "background+signal"], default="background")
    parser.add_argument("--uncertainty-bands", choices=["stat+syst", "syst-only", "both"], default="stat+syst")
    args = parser.parse_args()
    if not args.systematics_config and (args.uncertainty_scope != "background" or args.uncertainty_bands != "stat+syst"):
        parser.error("Nondefault uncertainty options require --systematics-config")
    if args.systematics_config:
        systematics, plot_options = load_systematics_config(args.systematics_config)
    else:
        systematics, plot_options = None, {"variable": DEFAULT_PLOT_VARIABLE}
    plot_variable = plot_options["variable"]

    logger.setLevel(args.log_level)
    
    year = args.year
    if year == "run2":
        year_param = "*"
    elif year == "2016":
        year_param = "2016*"
    else:
        year_param = year

    load_root = Path(args.load_path)
    if plot_variable != DEFAULT_PLOT_VARIABLE:
        # run_stage2_vbf.py names the directory after the variable it filled, so
        # `score_<label>_<postfix>` becomes `<variable>_<label>_<postfix>`.
        if not load_root.name.startswith("score_"):
            raise ValueError(f"Cannot derive the {plot_variable} histogram directory "
                             f"from {load_root.name!r}; expected a 'score_' prefix")
        load_root = load_root.with_name(plot_variable + load_root.name[len("score"):])
    load_path = f"{load_root}/{year_param}"

    logger.info(f"Looking for pickled histograms in: {load_path}")

    pickled_filelist = glob.glob(f"{load_path}/*.pkl")
    logger.info(f"load_path : {load_path}")
    # logger.info(f"pickled_hists : {pickled_filelist}")

    pickled_hist_dict = getPickledHist_byFname(pickled_filelist, load_path)
    logger.info(f"pickled_hist_dict.keys() : {pickled_hist_dict.keys()}")
    sample_group_dict = load_group_process_indicators(args.sample_config)
    logger.info(f"sample_group_dict (from {args.sample_config}) : {sample_group_dict}")
    hist_dict_bySampleGroup = arrangeHist_bySampleGroup(pickled_hist_dict, sample_group_dict)
    logger.info(f"hist_dict_bySampleGroup.keys() : {hist_dict_bySampleGroup.keys()}")

    # read lumi value from configs/parameters/lumi.yaml
    infile_lumi = os.path.join("configs", "parameters", "lumi.yaml")
    with open(infile_lumi, "r") as f:
        lumi_config = yaml.safe_load(f)
    lumi_dict = lumi_config.get("integrated_lumis", {})
    lumi = lumi_dict.get(year, 0.0)
    # convert from pb to fb
    lumi = round(lumi / 1000.0, 1)
    if lumi == 0.0:
        logger.error(f"lumi for year {year} is not defined!")
        raise ValueError(f"lumi for year {year} is not defined!")

    lumi_val = lumi

    with open(VBF_PLOT_SETTINGS, "r") as file:
        plot_settings = json.load(file)
    # logger.info(f"plot_settings: {plot_settings}")
    var = plot_variable
    if var == DEFAULT_PLOT_VARIABLE:
        binning = selection.binning
    else:
        # Kinematic variables take their edges from the same vbf plot settings that
        # stage2 filled them with, so `binning is None` lets plotStage2DNN_score
        # build the linspace from that one source.
        if var not in plot_settings:
            raise ValueError(f"{var} is not configured in {VBF_PLOT_SETTINGS}")
        binning = None
    region_name = args.region
    category = args.category
    output_tag = args.mva_name
    if args.do_vbf_filter_study and "_vbf_filter_study" not in output_tag:
        output_tag = f"{output_tag}_vbf_filter_study"
    full_save_path = f"{args.save_path}/{args.year}/Reg_{region_name}/Cat_{category}/{output_tag}_NoVHveto/"
    plotStage2DNN_score(
        hist_dict_bySampleGroup,
        var,
        plot_settings,
        full_save_path,
        region_name,
        category,
        do_logscale=True,
        binning=binning,
        lumi=lumi_val,
        status="Private",
        systematics=systematics,
        uncertainty_scope=args.uncertainty_scope,
        uncertainty_bands=args.uncertainty_bands,
        systematics_config=args.systematics_config,
    )
