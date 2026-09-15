"""Fixed-WP UParT event weights for XZZ UL2018 NanoAODv15.

Efficiencies must be derived from analysis MC before any b-tag selection,
separately by process group and jet hadron flavor. The strict systematic
evaluator rejects uncovered jets; the central production evaluator exposes
the explicitly selected supported-domain and nearest-boundary conventions.
"""
from functools import lru_cache
import gzip
import json
from pathlib import Path

import awkward as ak
import correctionlib
import numpy as np


PAYLOAD = "/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run2-2018-UL-NanoAODv15/2026-06-18/btagging.json.gz"


@lru_cache(maxsize=1)
def _payload():
    with gzip.open(PAYLOAD, "rt") as handle:
        raw = json.load(handle)
    return correctionlib.CorrectionSet.from_file(PAYLOAD), {
        item["name"]: item for item in raw["corrections"]
    }


def _category(node, key):
    return next(item["value"] for item in node["content"] if item["key"] == key)


def _domain(raw, name, flavor):
    node = _category(_category(_category(raw[name]["data"], "central"), "M"), flavor)
    eta_low, eta_high = node["edges"]
    pt_edges = node["content"][0]["edges"]
    return eta_low, eta_high, pt_edges[0], pt_edges[-1]


def fixedwp_product(tagged, efficiency, scale_factor):
    """BTV method 1a: SF for tagged jets, (1-SF*eff)/(1-eff) otherwise."""
    if not (ak.to_list(ak.num(tagged)) == ak.to_list(ak.num(efficiency))
            == ak.to_list(ak.num(scale_factor))):
        raise ValueError("Tag decisions, efficiencies and SFs must have identical jet layout")
    eff = ak.to_numpy(ak.flatten(efficiency))
    sf = ak.to_numpy(ak.flatten(scale_factor))
    if not (np.isfinite(eff).all() and np.isfinite(sf).all()):
        raise ValueError("Nonfinite tagging efficiency or scale factor")
    if np.any((eff < 0) | (eff >= 1) | (sf < 0) | (sf * eff > 1)):
        raise ValueError("Invalid tagging probabilities; inspect MC efficiency statistics and SFs")
    factors = ak.where(tagged, scale_factor,
                       (1 - scale_factor * efficiency) / (1 - efficiency))
    return ak.prod(factors, axis=1)


def u_part_event_weights(jets, efficiencies, *, analysis, year, nano_version):
    """Return nominal and independent bc/light correlated/uncorrelated weights.

    ``jets`` are exactly the candidates queried by the medium b-tag veto,
    including untagged jets. ``efficiencies`` has the same jagged layout.
    Out-of-domain jets raise before evaluation; JSON flow values are not a
    validated extrapolation prescription. Production central conventions use
    ``central_event_weights`` below and remain explicitly distinguished.
    """
    if (analysis, str(year), nano_version) != ("XZZ2l2nu", "2018", 15):
        raise ValueError("UParT fixed-WP implementation is restricted to XZZ2l2nu 2018 v15")
    counts = ak.to_numpy(ak.num(jets))
    if not np.array_equal(counts, ak.to_numpy(ak.num(efficiencies))):
        raise ValueError("Efficiency map lookup does not match candidate jet layout")
    pt = ak.to_numpy(ak.flatten(jets.pt))
    eta = np.abs(ak.to_numpy(ak.flatten(jets.eta)))
    flavor = ak.to_numpy(ak.flatten(jets.hadronFlavour))
    score = ak.to_numpy(ak.flatten(jets.btagUParTAK4B))
    if not (np.isfinite(pt).all() and np.isfinite(eta).all() and np.isfinite(score).all()):
        raise ValueError("Nonfinite b-tag candidate coordinates or discriminator")
    if not np.isin(flavor, [0, 4, 5]).all():
        raise ValueError("Expected NanoAOD hadron flavors 0, 4 or 5")
    corrections, raw = _payload()
    for flav in [0, 4, 5]:
        name = "UParTAK4_light" if flav == 0 else "UParTAK4_comb"
        elo, ehi, plo, phi = _domain(raw, name, flav)
        outside = (flavor == flav) & ((eta < elo) | (eta >= ehi) | (pt < plo) | (pt >= phi))
        if outside.any():
            raise ValueError(f"{outside.sum()} flavor-{flav} jets outside measured {name} "
                             f"domain: {plo} <= pt < {phi}, {elo} <= abs(eta) < {ehi}; "
                             "an explicit extrapolation prescription is required")
    tagged = jets.btagUParTAK4B > corrections["UParTAK4_wp_values"].evaluate("M")
    specs = {"nominal": (None, "central")}
    for group in ["bc", "light"]:
        for source in ["correlated", "uncorrelated"]:
            for direction in ["up", "down"]:
                specs[f"{group}_{source}_{direction}"] = (group, f"{direction}_{source}")
    result = {}
    for label, (varied_group, variation) in specs.items():
        sf = np.ones(len(pt))
        for group, mask, name in [("bc", flavor != 0, "UParTAK4_comb"),
                                  ("light", flavor == 0, "UParTAK4_light")]:
            if mask.any():
                systematic = variation if group == varied_group else "central"
                sf[mask] = corrections[name].evaluate(systematic, "M", flavor[mask], eta[mask], pt[mask])
        result[label] = fixedwp_product(tagged, efficiencies, ak.unflatten(sf, counts))
    return result


@lru_cache(maxsize=8)
def _efficiency_map(path):
    with open(path) as handle:
        result = json.load(handle)
    if (result['analysis'], str(result['year']), result['nano_version']) != ('XZZ2l2nu', '2018', 15):
        raise ValueError('Efficiency map must describe XZZ2l2nu 2018 NanoAODv15')
    return result


def central_event_weights(jets, dataset, efficiency_file):
    """Central supported/nearest conventions used by the full 2018 validation.

    Uses every veto candidate, including untagged jets. The original jet pT
    selects efficiency bins; only the SF coordinates are clipped for nearest.
    Unknown samples fail rather than silently borrowing another process map.
    """
    path = Path(efficiency_file)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    maps = _efficiency_map(str(path))
    if dataset not in maps['datasets']:
        raise ValueError(f'No XZZ b-tag efficiency group configured for dataset {dataset!r}')
    group = maps['datasets'][dataset]
    counts = ak.to_numpy(ak.num(jets))
    pt = ak.to_numpy(ak.flatten(jets.pt))
    eta = np.abs(ak.to_numpy(ak.flatten(jets.eta)))
    flavor = ak.to_numpy(ak.flatten(jets.hadronFlavour))
    score = ak.to_numpy(ak.flatten(jets.btagUParTAK4B))
    if not (np.isfinite(pt).all() and np.isfinite(eta).all()
            and np.isfinite(score).all() and np.isin(flavor, [0, 4, 5]).all()):
        raise ValueError('Invalid b-tag candidate coordinates, score or flavor')
    efficiency = np.full(len(pt), np.nan)
    sf = {name: np.ones(len(pt)) for name in ['supported', 'nearest']}
    outside = np.zeros(len(pt), dtype=bool)
    corrections, raw = _payload()
    if maps['threshold'] != corrections['UParTAK4_wp_values'].evaluate('M'):
        raise ValueError('Efficiency map and calibration working points differ')
    for flav in [0, 4, 5]:
        mask = flavor == flav
        for bin_ in maps['groups'][group][str(flav)]:
            efficiency[mask & (pt >= bin_['pt_low']) & (pt < bin_['pt_high'])] = bin_['efficiency']
        name = 'UParTAK4_light' if flav == 0 else 'UParTAK4_comb'
        elo, ehi, plo, phi = _domain(raw, name, flav)
        covered = mask & (eta >= elo) & (eta < ehi) & (pt >= plo) & (pt < phi)
        outside |= mask & ~covered
        if covered.any():
            sf['supported'][covered] = corrections[name].evaluate('central', 'M', flavor[covered], eta[covered], pt[covered])
        if mask.any():
            sf['nearest'][mask] = corrections[name].evaluate(
                'central', 'M', flavor[mask],
                np.clip(eta[mask], elo, np.nextafter(ehi, elo)),
                np.clip(pt[mask], plo, np.nextafter(phi, plo)))
    if not np.isfinite(efficiency).all():
        raise ValueError('B-tag candidate is outside the efficiency map')
    tagged = jets.btagUParTAK4B > maps['threshold']
    result = {name: fixedwp_product(tagged, ak.unflatten(efficiency, counts), ak.unflatten(values, counts))
              for name, values in sf.items()}
    result['outside_coverage'] = ak.any(ak.unflatten(outside, counts), axis=1)
    return result
