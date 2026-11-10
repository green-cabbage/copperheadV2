"""Official UL electron reconstruction and ID factors."""
import awkward as ak

from modules.correctionlib_file_cache import get_corrset, get_corr_input_names
from modules.utils import logger


def add_electron_sfs_correctionlib(electrons, config, electron_id):
    """
    Add electron reco and ID SFs using correctionlib.

    Event SF = product over the selected electrons; zero-electron events get unity.
    As for muons, evaluated before energy corrections and FSR (pt_uncorr), at the supercluster eta.

    Parameters
    ----------
    electrons : awkward records
        Selected electrons with fields `.pt_uncorr` and `.superclusterEta`.
    config : dict
        config["xzz_electron_sf"] must contain:
          - "json_file": path to electron.json(.gz)
          - "correction": correction name
          - "reco_below20", "reco_above20": reco working points
          - "id": map of selection ID (electron_id_v9/v12 value) -> SF working point (null = no ID SF)
    electron_id : str or None
        Selection ID applied to the electrons; None if no ID is applied (no ID SF).
    """
    ele_sf_lookup_info = config["xzz_electron_sf"]
    logger.debug("Electron SF lookup info from config: %s", ele_sf_lookup_info)

    ele_corr = get_corrset(ele_sf_lookup_info["json_file"])[ele_sf_lookup_info["correction"]]
    logger.debug(f"get_corr_input_names ele_corr: {get_corr_input_names(ele_corr)}")

    working_points = {"electronReco": (ele_sf_lookup_info["reco_below20"], ele_sf_lookup_info["reco_above20"])}
    if electron_id is not None:
        if electron_id not in ele_sf_lookup_info["id"]:
            raise ValueError(
                f"No electron ID SF working point for selection ID '{electron_id}': "
                f"add it to xzz_electron_sf.<year>.id in electron.yaml (null = no ID SF)"
            )
        id_wp = ele_sf_lookup_info["id"][electron_id]
        if id_wp is not None:
            working_points["electronID"] = (id_wp, id_wp)
    logger.debug(f"Electron SF working points: {working_points}")

    pt = ak.where(electrons.pt_uncorr < 10.0, 10.0, electrons.pt_uncorr)  # SF table floor
    eta_sc = electrons.superclusterEta
    below20 = pt < 20.0
    result = {}
    for name, (wp_below20, wp_above20) in working_points.items():
        result[name] = {}
        for variation, value_type in [("nom", "sf"), ("up", "sfup"), ("down", "sfdown")]:
            values = ak.where(
                below20,
                ele_corr.evaluate(config["year"], value_type, wp_below20, eta_sc, ak.where(below20, pt, 19.99)),
                ele_corr.evaluate(config["year"], value_type, wp_above20, eta_sc, ak.where(below20, 20.0, pt)),
            )
            result[name][variation] = ak.prod(values, axis=1)
    return result
