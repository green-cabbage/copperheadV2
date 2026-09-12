import numpy as np
import awkward as ak
import pandas as pd
from modules.classify_year import is_run3


def filterRegion(events, region="h-peak", mass_field="dimuon_mass"):
    if isinstance(events, pd.DataFrame):
        fields = events.columns
    else: # awkward zip
        fields = events.fields  
    if mass_field not in fields:
        raise ValueError(f"{mass_field} not found in events fields for region selection.")
    mass = events[mass_field]
    z_peak = (mass >= 70.0) & (mass < 110.0)
    h_peak = (mass >= 115.0) & (mass < 135.0)
    h_sidebands = ((mass >= 110.0) & (mass < 115.0)) | (
        (mass >= 135.0) & (mass < 150.0)
    )
    if region == "z-peak":
        mask = z_peak
    elif region == "h-peak":
        mask = h_peak
    elif region == "h-sidebands":
        mask = h_sidebands
    elif region == "signal":
        mask = h_sidebands | h_peak
    elif region == "full":
        mask = z_peak | h_sidebands | h_peak
    else:
        raise ValueError(
            f"Invalid region selection: {region}. Valid options are: 'z-peak', 'h-peak', 'h-sidebands', 'signal', 'full'."
        )

    return mask, events[mask]


def applyRegionCatCuts_XZZ2l2nu(
    events,
    category: str,
    region_name: str,
    variation: str = "nominal",
    njets_selection: str = "inclusive",  # available options ["inclusive", "0", "1", "2"]
):
    """Select an XZZ -> 2l2nu dilepton channel and mass region."""
    channel_fields = {
        "2l2nu_mumu": "is_mm",
        "2l2nu_ee": "is_ee",
        "2l2nu_emu": "is_em",
        "2l2nu_vbf_mumu": "is_mm",
        "2l2nu_njet1_mumu": "is_mm",
        "2l2nu_njet0_mumu": "is_mm",
        "2l2nu_vbf_ee": "is_ee",
        "2l2nu_njet1_ee": "is_ee",
        "2l2nu_njet0_ee": "is_ee",
    }
    jet_categories = {
        "2l2nu_vbf_mumu": "vbf",
        "2l2nu_njet1_mumu": "njet1",
        "2l2nu_njet0_mumu": "njet0",
        "2l2nu_vbf_ee": "vbf",
        "2l2nu_njet1_ee": "njet1",
        "2l2nu_njet0_ee": "njet0",
    }
    # Leading/subleading lepton eta fields of each channel. The muon columns are
    # pad_none'd in stage1, so ee events have null mu*_eta and vice versa.
    lepton_eta_fields = {
        "is_mm": ("mu1_eta", "mu2_eta"),
        "is_ee": ("el1_eta", "el2_eta"),
        "is_em": ("mu1_eta", "el1_eta"),
    }

    if category not in channel_fields:
        raise ValueError(
            f"Invalid XZZ -> 2l2nu category: {category}. "
            f"Valid options are: {', '.join(repr(name) for name in channel_fields)}."
        )

    channel_field = channel_fields[category]
    fields = events.columns if isinstance(events, pd.DataFrame) else events.fields

    def fill_false(mask):
        if isinstance(events, pd.DataFrame):
            return mask.fillna(False)
        return ak.fill_none(mask, value=False)

    if channel_field not in fields:
        raise KeyError(
            f"[selection] Missing required field for {category} selection: "
            f"{channel_field}"
        )

    channel_cut = fill_false(events[channel_field])


    # apply Z pt cut. dilepton_pt is the channel-aware Z candidate (mm -> dimuon,
    # ee -> diele, em -> e+mu); dimuon_pt is the fallback for older stage1 output
    # and is null outside the mm channel.
    for zpt_field in ("dilepton_pt", "dimuon_pt"): # FIXME: we should probably just use dilepton_pt.
        if zpt_field in fields:
            break
    else:
        raise KeyError(
            "[selection] Missing required field for XZZ -> 2l2nu Z pt "
            "selection: tried dilepton_pt, dimuon_pt"
        )
    channel_cut = channel_cut & fill_false(events[zpt_field] > 55.0)

    if channel_field in ("is_mm", "is_ee"):
        if "PuppiMET_pt" not in fields:
            raise KeyError(
                "[selection] Missing required field for XZZ -> 2l2nu "
                "MET selection: PuppiMET_pt"
            )
        channel_cut = channel_cut & fill_false(events["PuppiMET_pt"] < 100.0)

    jet_category = jet_categories.get(category)
    if jet_category is not None:
        use_var = (
            "nominal"
            if (isinstance(variation, str) and variation.startswith("wgt"))
            else variation
        )

        def varcol(base):
            for candidate in (f"{base}_{use_var}", f"{base}_nominal", base):
                if candidate in fields:
                    return events[candidate]
            raise KeyError(
                f"[selection] Missing required field for {category}: tried "
                f"{base}_{use_var}, {base}_nominal, {base}"
            )

        has_jet30 = fill_false(varcol("jet1_pt") > 30.0)

        if jet_category == "njet0":
            category_cut = ~has_jet30
        else:
            jet1_eta = varcol("jet1_eta")
            jet2_eta = varcol("jet2_eta")
            eta_min = np.minimum(jet1_eta, jet2_eta)
            eta_max = np.maximum(jet1_eta, jet2_eta)
            lep1_field, lep2_field = lepton_eta_fields[channel_field]
            for lep_field in (lep1_field, lep2_field):
                if lep_field not in fields:
                    raise KeyError(
                        f"[selection] Missing required field for {category}: "
                        f"{lep_field}"
                    )
            lep1_eta = events[lep1_field]
            lep2_eta = events[lep2_field]
            leptons_between_jets = (
                (lep1_eta > eta_min)
                & (lep1_eta < eta_max)
                & (lep2_eta > eta_min)
                & (lep2_eta < eta_max)
            )

            # Stage-1 stores up to four jets; use jets 3 and 4 for the
            # paper's veto on additional pT > 30 GeV jets inside the gap.
            central_extra_jet = has_jet30 & False # initialize False array
            for jet_index in (3, 4):
                jet_pt = varcol(f"jet{jet_index}_pt")
                jet_eta = varcol(f"jet{jet_index}_eta")
                central_extra_jet = central_extra_jet | fill_false(
                    (jet_pt > 30.0) & (jet_eta > eta_min) & (jet_eta < eta_max)
                )

            vbf_cut = fill_false(
                (varcol("jet2_pt") > 30.0)
                & (varcol("jj_dEta") > 4.0)
                & (varcol("jj_mass") > 500.0)
                & leptons_between_jets
                & (~central_extra_jet)
            )
            if jet_category == "vbf":
                category_cut = vbf_cut
            else: # 2l2nu njet 1 category
                category_cut = has_jet30 & (~vbf_cut)

        channel_cut = channel_cut & category_cut

    # ---------------------------------------------------------
    #  Select events based on number of jets
    # ---------------------------------------------------------
    if njets_selection != "inclusive":
        use_var = (
            "nominal"
            if (isinstance(variation, str) and variation.startswith("wgt"))
            else variation
        )
        for cand in (f"njets_{use_var}", "njets_nominal", "njets"):
            if cand in fields:
                njets = events[cand]
                break
        else:
            raise KeyError(
                f"[selection] Missing required field for njets selection: tried "
                f"njets_{use_var}, njets_nominal, njets"
            )

        if njets_selection == "0":
            njets_mask = njets == 0
        elif njets_selection == "1":
            njets_mask = njets == 1
        elif njets_selection == "2":
            njets_mask = njets >= 2
        else:
            raise ValueError(
                f"Invalid njets_selection='{njets_selection}'. Valid options: 'inclusive', '0', '1', '2'."
            )

        channel_cut = channel_cut & fill_false(njets_mask)

    region_cut, _ = filterRegion(
        events,
        region=region_name,
        mass_field="dilepton_mass",
    )
    return events[channel_cut & region_cut]


def applyRegionCatCuts_HMuMu(
    events,
    category: str,
    region_name: str,
    process: str,
    variation: str,
    do_vbf_filter_study: bool = False,
    do_VH_veto: bool = False,
    jj_eta_region: str = "all",
    njets_selection: str = "inclusive",  # available options ["inclusive", "0", "1", "2"],
    year: str | None = None,
):
    use_var = (
        "nominal"
        if (isinstance(variation, str) and variation.startswith("wgt"))
        else variation
    )

    # Helper to fetch the right column, falling back to _nominal or base if needed
    def varcol(base):
        """
        Fetch the appropriate column from the events object, handling variations.

        Attempts to retrieve the column named '{base}_{use_var}', falling back to '{base}_nominal' and then '{base}'.
        Raises a KeyError if none of these columns are present in events.fields.

        Parameters
        ----------
        base : str
            The base name of the column to retrieve.

        Returns
        -------
        awkward.Array
            The selected column from the events object.

        Raises
        ------
        KeyError
            If none of the candidate columns are found in events.fields.
        """
        # print(f"Fetching variable column for: {base}")
        # print(f"Using variation: {use_var}")
        for cand in (f"{base}_{use_var}", f"{base}_nominal", base):
            if cand in events.fields:
                return events[cand]
        raise KeyError(
            f"[selection] Missing required field for selection: tried {base}_{use_var}, {base}_nominal, {base}"
        )

    # do mass region cut
    region, _ = filterRegion(events, region=region_name)

    # --- category cuts: USE varcol(...) for JES/JER-affected columns ---
    nbt_loose = varcol("nBtagLoose")
    nbt_medium = varcol("nBtagMedium")
    jj_mass = varcol("jj_mass")
    jj_dEta = varcol("jj_dEta")
    jet1_pt = varcol("jet1_pt")
    njets = varcol("njets")

    prod_cat_cut = ak.ones_like(region, dtype="bool")

    # do category cut
    if category == "nocat":
        prod_cat_cut = prod_cat_cut  # no additional cut
    else:  # VBF or ggH
        if do_VH_veto:
            print("Applying VH veto!")
            # NOTE: fatjet and MET veto for VH: nfatJets_drmuon == 0 and MET_pt < 150 GeV
            fatjet_veto = ak.fill_none((events.nfatJets_drmuon == 0), value=False)
            met_veto = ak.fill_none((events.MET_pt < 150), value=False)

            # INFO: Apply both fatjet and MET vetoes together
            prod_cat_cut = prod_cat_cut & fatjet_veto & met_veto

        # NOTE: btag cut for VH and ttH categories
        btagLoose_filter = ak.fill_none((nbt_loose >= 2), value=False)
        btagMedium_filter = ak.fill_none((nbt_medium >= 1), value=False) & ak.fill_none(
            (njets >= 2), value=False
        )
        btag_cut = btagLoose_filter | btagMedium_filter

        vbf_cut = (jj_mass > 400) & (jj_dEta > 2.5) & (jet1_pt > 35)
        vbf_cut = ak.fill_none(vbf_cut, value=False)

        if category == "vbf":
            # print("vbf mode!")
            prod_cat_cut = prod_cat_cut & vbf_cut
            prod_cat_cut = prod_cat_cut & (
                ~btag_cut
            )  # btag cut is for VH and ttH categories
        elif category == "ggh":
            # print("ggH mode!")
            prod_cat_cut = prod_cat_cut & (~vbf_cut)
            prod_cat_cut = prod_cat_cut & (
                ~btag_cut
            )  # btag cut is for VH and ttH categories
        elif category == "bJetVeto":
            # print("ggH mode!")
            prod_cat_cut = prod_cat_cut & (
                ~btag_cut
            )  # btag cut is for VH and ttH categories
        else:
            raise ValueError(
                "Invalid category option! Valid options are: 'vbf', 'ggh', 'nocat'."
            )

    if do_vbf_filter_study:
        process_lower = process.lower()
        if process_lower.startswith("dy"):
            gjj_threshold = 300 if (year is not None and is_run3(year)) else 350
            vbf_filter = ak.fill_none((events.gjj_mass > gjj_threshold), value=False)
            is_vbf_filter = "dy_vbf_filter" in process_lower
            if is_vbf_filter:
                # print(f"applying VBF filter cut on: {process}")

                prod_cat_cut = prod_cat_cut & vbf_filter
            else:
                prod_cat_cut = prod_cat_cut & (~vbf_filter)

    # ---------------------------------------------------------
    #  Select events based on number of jets
    # ---------------------------------------------------------
    if njets_selection != "inclusive":
        if njets_selection == "0":
            njets_mask = (njets == 0)
        elif njets_selection == "1":
            njets_mask = (njets == 1)
        elif njets_selection == "2":
            njets_mask = (njets >= 2)
        else:
            raise ValueError(
                f"Invalid njets_selection='{njets_selection}'. Valid options: 'inclusive', '0', '1', '2'."
            )
        prod_cat_cut = prod_cat_cut & ak.fill_none(njets_mask, value=False)

    # ---------------------------------------------------------
    #  jet-eta region selection (pair topology)
    # ---------------------------------------------------------
    if jj_eta_region and jj_eta_region != "all":

        # 1) prefer precomputed mask if present
        if jj_eta_region in events.fields:
            jj_eta_mask = ak.fill_none(events[jj_eta_region], value=False)

        else:
            # 2) compute from jet1_eta/jet2_eta (variation-safe)
            jet1_eta = varcol("jet1_eta")
            jet2_eta = varcol("jet2_eta")

            a1 = abs(jet1_eta)
            a2 = abs(jet2_eta)

            # basic regions
            j1_c = a1 < 2.5
            j2_c = a2 < 2.5

            j1_f25 = a1 > 2.5
            j2_f25 = a2 > 2.5

            j1_he = (a1 > 2.5) & (a1 < 3.0)
            j2_he = (a2 > 2.5) & (a2 < 3.0)

            j1_f30 = a1 > 3.0
            j2_f30 = a2 > 3.0

            masks = {
                "jj_both_central": j1_c & j2_c,
                "jj_non_central": ~ (j1_c & j2_c),
                "jj_one_fwd25_one_central": (j1_f25 & j2_c) | (j2_f25 & j1_c),
                "jj_one_he_one_central": (j1_he & j2_c) | (j2_he & j1_c),
                "jj_one_fwd30_one_central": (j1_f30 & j2_c) | (j2_f30 & j1_c),
                "jj_both_fwd25": j1_f25 & j2_f25,
                "jj_both_he": j1_he & j2_he,
                "jj_both_fwd30": j1_f30 & j2_f30,
                "jj_one_he_one_fwd30": (j1_he & j2_f30) | (j2_he & j1_f30),
            }

            if jj_eta_region not in masks:
                raise ValueError(
                    f"Invalid jj_eta_region='{jj_eta_region}'. "
                    f"Valid: all, {', '.join(masks.keys())}"
                )

            jj_eta_mask = ak.fill_none(masks[jj_eta_region], value=False)

        prod_cat_cut = prod_cat_cut & jj_eta_mask

    category_selection = prod_cat_cut & region
    events = events[category_selection]
    return events


def applyRegionCatCuts(
    events,
    category: str,
    region_name: str,
    process: str,
    variation: str,
    do_vbf_filter_study: bool = False,
    do_VH_veto: bool = False,
    jj_eta_region: str = "all",
    njets_selection: str = "inclusive",
    year: str | None = None,
    analysis: str = "HMuMu",
):
    """Dispatch region/category selection for the requested analysis."""
    if analysis == "XZZ2l2nu":
        return applyRegionCatCuts_XZZ2l2nu(
            events,
            category,
            region_name,
            variation=variation,
            njets_selection=njets_selection,
        )
    if analysis == "HMuMu":
        return applyRegionCatCuts_HMuMu(
            events,
            category,
            region_name,
            process,
            variation,
            do_vbf_filter_study=do_vbf_filter_study,
            do_VH_veto=do_VH_veto,
            jj_eta_region=jj_eta_region,
            njets_selection=njets_selection,
            year=year,
        )
    raise ValueError(
        f"Invalid analysis: {analysis}. Valid options are: 'HMuMu', 'XZZ2l2nu'."
    )


def applyRegionCatCutsByScore(
    events,
    category: str,
    region_name: str,
    process: str,
    variation: str,
    do_vbf_filter_study: bool = False,
    year: str | None = None,
    do_VH_veto: bool = False,
    jj_eta_region: str = "all",
    njets_selection: str = "inclusive",
):
    """
    Apply the same region-level selection as `applyRegionCatCuts`, but assign
    ggH/VBF categories using transformer scores instead of the cut-based VBF
    definition.

    Strategy:
    - if transf_vbf_score > transf_ggh_score, tag as VBF
    - otherwise tag as ggH
    """
    use_var = (
        "nominal"
        if (isinstance(variation, str) and variation.startswith("wgt"))
        else variation
    )

    # Helper to fetch the right column, falling back to _nominal or base if needed
    def varcol(base):
        """
        Fetch the appropriate column from the events object, handling variations.

        Attempts to retrieve the column named '{base}_{use_var}', falling back to '{base}_nominal' and then '{base}'.
        Raises a KeyError if none of these columns are present in events.fields.

        Parameters
        ----------
        base : str
            The base name of the column to retrieve.

        Returns
        -------
        awkward.Array
            The selected column from the events object.

        Raises
        ------
        KeyError
            If none of the candidate columns are found in events.fields.
        """
        # print(f"Fetching variable column for: {base}")
        # print(f"Using variation: {use_var}")
        for cand in (f"{base}_{use_var}", f"{base}_nominal", base):
            if cand in events.fields:
                return events[cand]
        raise KeyError(
            f"[selection] Missing required field for selection: tried {base}_{use_var}, {base}_nominal, {base}"
        )

    # do mass region cut
    region, _ = filterRegion(events, region=region_name)

    # --- category cuts: USE varcol(...) for JES/JER-affected columns ---
    nbt_loose = varcol("nBtagLoose")
    nbt_medium = varcol("nBtagMedium")
    jj_mass = varcol("jj_mass")
    jj_dEta = varcol("jj_dEta")
    jet1_pt = varcol("jet1_pt")
    njets = varcol("njets")

    prod_cat_cut = ak.ones_like(region, dtype="bool")

    required_fields = {"transf_vbf_score", "transf_ggh_score"}
    missing_fields = sorted(required_fields - set(events.fields))
    if missing_fields:
        raise KeyError(
            "Missing transformer score field(s) required for score-based "
            f"categorization: {missing_fields}"
        )

    vbf_score = ak.fill_none(events["transf_vbf_score"], float("-inf"))
    ggh_score = ak.fill_none(events["transf_ggh_score"], float("-inf"))
    # is_vbf = ak.fill_none(vbf_score > ggh_score, value=False)
    is_vbf = ak.fill_none((vbf_score/(vbf_score + ggh_score)) > 0.92522, value=False)
    # is_vbf = ak.fill_none(vbf_score > 0.925, value=False)

    if category == "nocat":
        prod_cat_cut = prod_cat_cut  # no additional cut
    else:
        # NOTE: btag cut for VH and ttH categories
        btagLoose_filter = ak.fill_none((nbt_loose >= 2), value=False)
        btagMedium_filter = ak.fill_none((nbt_medium >= 1), value=False) & ak.fill_none(
            (njets >= 2), value=False
        )
        btag_cut = btagLoose_filter | btagMedium_filter

        if category == "vbf":
            prod_cat_cut = prod_cat_cut & is_vbf
            prod_cat_cut = prod_cat_cut & (~btag_cut)         
        elif category == "ggh":
            prod_cat_cut = prod_cat_cut & (~is_vbf)
            prod_cat_cut = prod_cat_cut & (~btag_cut)        
        else:
            raise ValueError(
                "Invalid category option! Valid options are: 'vbf', 'ggh', 'nocat'."
            )

    if do_vbf_filter_study:
        process_lower = process.lower()
        if process_lower.startswith("dy"):
            gjj_threshold = 300 if (year is not None and is_run3(year)) else 350
            vbf_filter = ak.fill_none((events.gjj_mass > gjj_threshold), value=False)
            is_vbf_filter = "dy_vbf_filter" in process_lower
            if is_vbf_filter:
                prod_cat_cut = prod_cat_cut & vbf_filter
            else:
                prod_cat_cut = prod_cat_cut & (~vbf_filter)

    category_selection = prod_cat_cut & region
    events = events[category_selection]
    return events


binning_based_on_significanceScan = np.array([
  0.000000,
  0.349433,
  0.662083,
  0.882777,
  1.066689,
  1.250601,
  1.388535,
  1.590838,
  1.793141,
  1.958661,
  2.069008,
  2.262116,
  2.482810,
  3.678237,
])

binning_based_on_significanceScanV2 = np.array(  # 17 bins /depot/cms/users/shar1172/HHWWyy_DNN_For_HMuMu/best_binning_25bins_0p01.txt
    [  # one used for September 25, 2025 HiggsMuMu working group meeting.
        0.000000,
        0.179242,
        0.358485,
        0.537727,
        0.716970,
        0.896212,
        1.075455,
        1.254697,
        1.433940,
        1.613182,
        1.792425,
        1.971667,
        2.150910,
        2.330152,
        2.509395,
        2.688637,
        3.047122,
        4.301819,
    ]
)


# Binning for DNN scores
binning_HPScan_21bins = np.array([  #Latest training; 03 Sep 2025 (21 bins)
    0.0,
    0.382,
    0.579,
    0.733,
    0.863,
    0.979,
    1.087,
    1.191,
    1.291,
    1.389,
    1.487,
    1.584,
    1.683,
    1.783,
    1.884,
    1.989,
    2.098,
    2.214,
    2.338,
    2.478,
    2.65,
    3.188,
])

binning_HPScan_17bins = np.array(  # Latest training; 03 Sep 2025 (17 bins) having yields ~0.6 in each bin
    [
        0.0,
        0.435,
        0.655,
        0.826,
        0.972,
        1.105,
        1.233,
        1.355,
        1.476,
        1.596,
        1.719,
        1.842,
        1.97,
        2.104,
        2.249,
        2.409,
        2.606,
        3.188,
    ]
)

binning_HPScan_13bins = np.array([  #Latest training; 03 Sep 2025 (13 bins)
        0.0,
        0.511,
        0.765,
        0.962,
        1.136,
        1.298,
        1.457,
        1.614,
        1.775,
        1.94,
        2.115,
        2.309,
        2.539,
        3.188,
    ]
)

binning_August = np.array(  # _August DNN training
    [
        0.0,
        0.564,
        0.84,
        1.059,
        1.255,
        1.442,
        1.629,
        1.819,
        2.018,
        2.236,
        2.492,
        3.188,
    ]
)

binning_DNN_HIG19006 = np.array([
    0,
    0.07,
    0.432,
    0.71,
    0.926,
    1.114,
    1.28,
    1.428,
    1.564,
    1.686,
    1.798,
    1.9,
    2.0,
    2.8,
])

# binning = binning_HPScan_21bins
# binning = binning_HPScan_13bins
# binning = binning_HPScan_17bins
# binning = binning_based_on_significanceScan
binning = binning_based_on_significanceScanV2  # 17 bins; one used for September 25, 2025 HiggsMuMu working group meeting.
