import itertools
import logging
import os
import subprocess
import sys
from pathlib import Path
from shlex import join as shjoin

from modules.utils import logger

logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# CLI flags (no argparse)
# -----------------------------------------------------------------------------
ARGS = set(sys.argv[1:])
DRY_RUN = "--dry-run" in ARGS
FORCE = "--force" in ARGS
DEBUG = "--debug" in ARGS
MINIMUM_SET = "--minimum-set" in ARGS

if DEBUG:
    logger.setLevel(logging.DEBUG)

# -----------------------------------------------------------------------------
# User config
# -----------------------------------------------------------------------------
BASE_SCRIPT = [
    "python",
    "plotter/validation_plotter_unified.py",
    "--analysis",
    "XZZ2l2nu",
    "--NanoAODv",
    "15",
    # XZZ2l2nu-specific sample grouping: 2017 DY is dyTo2L_M-50_aMCatNLO alone.
    # HMuMu keeps configs/samples/samples.yaml untouched.
    "--sample-config",
    os.environ.get("XZZ_SAMPLE_CONFIG", "configs/samples/samples_xzz2l2nu.yaml"),
]

stage1_dir = (
    Path(os.environ.get("XZZ_STAGE1_ROOT", "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean"))
    / os.environ.get(
        "XZZ_LABEL",
        "Run2_nanoAODv15_XZZ2l2nu_Full2017_Sep01",
    )
    / "stage1_output"
)
LOAD_PATH = stage1_dir / "{year}" / "f1_0"
logger.info(f"Using LOAD_PATH: {LOAD_PATH}")

# Get the main directory based on run2 and nanoAOD version from the load path, e.g. Run3_nanoAODv12
# Example: Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV -> Run3_nanoAODv12
stage1_name = stage1_dir.parent.name
parts = stage1_name.split("_")
logger.info(f"Parsed stage1_name: {stage1_name}, parts: {parts}")
outputDir = "_".join(parts[:2]) if len(parts) >= 2 else stage1_name
logger.info(f"outputDir: {outputDir}")

# SAVE_TAG = "XZZ2l2nu_mumu"
# SAVE_TAG = "incDY_DNNWgts"
# Override with XZZ_SAVE_TAG to send a run to a fresh figs/ directory instead of
# overwriting an existing one (a new tag makes --force unnecessary).
SAVE_TAG = os.environ.get("XZZ_SAVE_TAG", "XZZ2l2nu_Zpt55Cut")
SAVE_ROOT = Path("./validation/figs") / outputDir / f"{stage1_name}_{SAVE_TAG}"
logger.info(f"Using SAVE_ROOT: {SAVE_ROOT}")

# Prevent overwriting
if SAVE_ROOT.exists() and not FORCE:
    raise RuntimeError(f"SAVE_ROOT exists: {SAVE_ROOT} (use --force to proceed)")

# years = ["2022preEE", "2022postEE", "2023", "2023BPix", "2024"]
# years = ["2022preEE"]
years = [os.environ.get("XZZ_YEAR", "2017")]
# years = ["2023"]
# years = ["2023BPix"]
# years = ["2024"]

# categories = ["nocat", "vbf", "ggh"]
# XZZ2l2nu dilepton channels are prefixed with "2l2nu_"
# ==========================================================================
# XZZ2l2nu IS BLINDED. The default below is the three control regions ONLY.
#
# Plotting any other category -- 2l2nu_vbf_*, 2l2nu_njet0_*, 2l2nu_njet1_*, or
# the inclusive 2l2nu_mumu / 2l2nu_ee in the Z-peak region -- is UNBLINDING the
# analysis. Do not change this default. Override per-run with XZZ_CATEGORIES
# only when the user has explicitly asked for those categories.
#
# The njet0 / njet1 / VBF subdivisions OF a control region (2l2nu_mllsb_njet0_mumu,
# 2l2nu_emu_cr_vbf, ...) are still control regions and ARE PART OF THE DEFAULT:
# every CR is plotted jet-inclusive plus vbf / njet1 / njet0.
#
# Control regions of the reference analysis (slide 7 of
# ref_materials/HZZ_2l2nu_stage1/HZZ_220526_X_zz_2l2v_update.pdf). Each pins its
# own m(ll) window, so region_options below is ignored for these categories.
# The e-mu CR has no mumu/ee split: it is the opposite-flavour channel itself.
# ==========================================================================
# Each control region is plotted ONLY in its vbf / njet1 / njet0 jet categories,
# on top of the existing mumu/ee split. 5 CRs x 3 = 15. The jet-inclusive parents
# are deliberately not plotted; the three splits already partition each CR.
# The names are the ones modules.selection generates; the splits are the parent
# CR plus one jet_category, so no cut can drift between them.
categories = [
    # m(ll) sideband first, as requested
    "2l2nu_mllsb_vbf_mumu",
    "2l2nu_mllsb_njet1_mumu",
    "2l2nu_mllsb_njet0_mumu",
    "2l2nu_mllsb_vbf_ee",
    "2l2nu_mllsb_njet1_ee",
    "2l2nu_mllsb_njet0_ee",
    # low MET
    "2l2nu_lowmet_vbf_mumu",
    "2l2nu_lowmet_njet1_mumu",
    "2l2nu_lowmet_njet0_mumu",
    "2l2nu_lowmet_vbf_ee",
    "2l2nu_lowmet_njet1_ee",
    "2l2nu_lowmet_njet0_ee",
    # e-mu CR -- opposite-flavour channel, no mumu/ee split
    "2l2nu_emu_cr_vbf",
    "2l2nu_emu_cr_njet1",
    "2l2nu_emu_cr_njet0",
]
# categories = ["ggh"]
# categories = ["vbf"]
# categories = ["nocat"]

# Override with XZZ_CATEGORIES as a comma-separated list, e.g.
#   XZZ_CATEGORIES=2l2nu_lowmet_mumu,2l2nu_emu_cr
# The special value "all" expands to every XZZ2l2nu category known to
# modules.selection, control regions included.
_categories_env = os.environ.get("XZZ_CATEGORIES", "").strip()
if _categories_env:
    if _categories_env == "all":
        from modules.selection import XZZ2L2NU_CATEGORIES

        categories = list(XZZ2L2NU_CATEGORIES)
    else:
        categories = [c.strip() for c in _categories_env.split(",") if c.strip()]
    logger.info(f"XZZ_CATEGORIES override -> {categories}")


JJ_ETA_REGIONS = ["all"]

# Boolean flags
vbf_filter_study_options = [False]  # True/False list
# XZZ2l2nu defaults to no Z-pT weights (standing user preference).
# True divides separate_wgt_zpt out of wgt_nominal; other weights stay in place.
# Explicit comparisons remain available through XZZ_ZPT_OPTIONS=on|off|both.
remove_zpt_weights_options = [True]
add_dnn_zpt_weights_options = [False]  # True/False list
# min_set_of_vars = MINIMUM_SET  # minimal set of vars
min_set_of_vars = False  # False -> plot every available variable

# Override with XZZ_MIN_VARS=1 (or 0) to pick the minimal variable set for a
# single run without editing the default above.
_min_vars_env = os.environ.get("XZZ_MIN_VARS", "").strip()
if _min_vars_env:
    min_set_of_vars = _min_vars_env not in ("0", "false", "False", "no")
    logger.info(f"XZZ_MIN_VARS override -> min_set_of_vars={min_set_of_vars}")

_zpt_env = os.environ.get("XZZ_ZPT_OPTIONS", "").strip().lower()
if _zpt_env:
    _zpt_map = {"on": [False], "off": [True], "both": [True, False]}
    if _zpt_env not in _zpt_map:
        raise ValueError(
            f"XZZ_ZPT_OPTIONS={_zpt_env!r} not understood; use on, off or both"
        )
    remove_zpt_weights_options = _zpt_map[_zpt_env]
    logger.info(
        f"XZZ_ZPT_OPTIONS override -> remove_zpt_weights_options={remove_zpt_weights_options}"
    )

region_options = [["z-peak"]]

njets_options = ["inclusive"]
# njets_options = ["inclusive", "0", "1", "2"]
# njets_options = ["0", "1", "2"]
# njets_options = ["inclusive", "0", "1", "2"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def build_command(
    year: str,
    save_path: Path,
    load_path: Path,
    cat: str,
    vbf_filter_study: bool,
    remove_zpt_weights: bool,
    add_dnn_zpt_weights: bool,
    region: list[str],
    njets: str,
    jj_eta_regions: str,
) -> list[str]:
    cmd = (
        BASE_SCRIPT
        + ["-y", year]
        + ["--save_path", str(save_path)]
        + ["--load", str(load_path)]
        + ["-cat", cat]
        # HMuMu MC scheme: full background stack (EWK, VV, TOP=TT+ST, DY) plus the
        # ggH/VBF signal lines, exactly as the H->mumu control plots are built.
        # AN sample scheme adds W+jets (HT-binned) and keeps the triboson group.
        # EWK stays listed for year-generality; it resolves empty for 2017.
        + ["-bkgorder", "EWK", "VV", "VVV", "WJETS", "TOP", "DY"]
        # Inclusive stream list. Streams a year does not have are simply absent from the
        # sample config's DATA group and are ignored, so one list serves every Run 2 year.
        # Was hard-coded to 2017's B-F, which silently dropped data_A from the 2018 plots.
        + ["-data", "A", "B", "C", "D", "E", "F", "G", "H"]
        + ["--use-compacted", "compacted"]  # "", "compacted", "compacted_WithDNNScore"
        # Gateway is opt-in so the same driver can run against a local cluster.
        + (["--use_gateway", "--cluster_index", "0"]
           if os.environ.get("XZZ_USE_GATEWAY", "1") == "1" else [])
        + ["--njets", str(njets)]
        + ["--jj-eta-region", jj_eta_regions]
    )

    if DEBUG:
        cmd += ["--log-level", "DEBUG"]

    if min_set_of_vars:
        cmd += ["--minimum_set"]

    if vbf_filter_study:
        cmd += ["--vbf_filter_study"]

    if region:
        cmd += ["--region", *region]

    if remove_zpt_weights:
        cmd += ["--remove_zpt_weights"]

    if add_dnn_zpt_weights:
        cmd += ["--use_dnn_zpt_weights"]

    return cmd


def run_all_combos():
    job_idx = 0

    for year in years:
        load_path = Path(str(LOAD_PATH).format(year=year))

        combo_iter = itertools.product(
            JJ_ETA_REGIONS,
            categories,
            vbf_filter_study_options,
            remove_zpt_weights_options,
            add_dnn_zpt_weights_options,
            region_options,
            njets_options,
        )

        for jj_eta_regions, cat, vbf_flag, zpt_flag, dnn_flag, region_list, njets in combo_iter:
            # skip meaningless combos
            if cat == "vbf" and njets != "inclusive":
                logger.debug(f"Skipping vbf with njets={njets} (not meaningful)")
                continue

            # if vbf_filter_study: remove z-peak from region
            region = [r for r in region_list if not (vbf_flag and r == "z-peak")]

            # structured save dirs (much easier to browse)
            save_path = (
                SAVE_ROOT
                / f"VBFfilter_{vbf_flag}"
            )

            save_path.mkdir(parents=True, exist_ok=True)

            cmd = build_command(
                year=year,
                save_path=save_path,
                load_path=load_path,
                cat=cat,
                vbf_filter_study=vbf_flag,
                remove_zpt_weights=zpt_flag,
                add_dnn_zpt_weights=dnn_flag,
                region=region,
                njets=njets,
                jj_eta_regions=jj_eta_regions,
            )

            job_idx += 1
            logger.info("\n" + "=" * 80)
            logger.info(
                f"[{job_idx:04d}] {year} {cat} njets={njets} vbf={vbf_flag} zptRm={zpt_flag} dnnZpt={dnn_flag}"
            )
            logger.info(shjoin(cmd))
            # if job_idx < 3:  # only print the first few commands for sanity check
            #     continue;

            if not DRY_RUN:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_all_combos()
