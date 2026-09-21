"""Run-2 PuppiMET XY correction with a pinned, explicitly configured payload."""
from functools import lru_cache
import hashlib
import gzip
import json
from pathlib import Path

import awkward as ak
import correctionlib
import numpy as np


@lru_cache(maxsize=4)
def _payload(filename, sha256):
    path = Path(filename)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError("XZZ MET XY payload checksum does not match configuration")
    # The published XY routine translates MET x/y by an npv-dependent offset;
    # its formula has no upper MET-pT cut. The JSON wraps that same formula in
    # a [0,6500) bin. Extend only this geometric input guard, keeping the
    # original formula, coefficients, phi bounds and run selection intact.
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        payload = json.load(stream)
    for item in payload["corrections"]:
        if item["name"] not in {
            f"{quantity}_metphicorr_puppimet_{kind}"
            for quantity in ("pt", "phi") for kind in ("mc", "data")
        }:
            continue
        node = item["data"]
        if (node.get("nodetype") != "binning"
                or node.get("input") != "met_pt"
                or node.get("edges") != [0.0, 6500.0]
                or len(node.get("content", [])) != 1
                or node.get("flow") != "error"):
            raise ValueError("Unexpected PuppiMET XY pT-domain schema")
        node["edges"] = [0.0, "inf"]
    return correctionlib.CorrectionSet.from_string(json.dumps(payload))


def puppimet_xy(pt, phi, npvs, run, *, is_mc, config):
    """Return corrected pT/phi; do not modify the original MET arrays."""
    values = [np.asarray(ak.to_numpy(v), dtype=np.float64) for v in (pt, phi, npvs, run)]
    if any(not np.isfinite(v).all() for v in values):
        raise ValueError("Nonfinite input to XZZ PuppiMET XY correction")
    # The published UL XY recipe caps the vertex count before PF/Puppi lookup:
    # https://lathomas.web.cern.ch/METStuff/XYCorrections/XYMETCorrection_withUL17andUL18andUL16.h
    values[2] = np.minimum(values[2], 100.0)
    corr = _payload(config["file"], config["sha256"])
    suffix = "mc" if is_mc else "data"
    corrected = tuple(
        corr[f"{quantity}_metphicorr_puppimet_{suffix}"].evaluate(*values)
        for quantity in ("pt", "phi")
    )
    if any(not np.isfinite(v).all() for v in corrected):
        raise ValueError("Nonfinite output from XZZ PuppiMET XY correction")
    return tuple(ak.Array(v) for v in corrected)
