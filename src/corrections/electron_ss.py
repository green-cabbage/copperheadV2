"""
EGM electron energy scale (data) and smearing (MC) for Run 2 UL NanoAODv15.

NanoAODv9 electrons already carry scale and smearing; NanoAODv15 ones do not, so the EGM
payloads (same keys for every UL year) are applied here.
Template: jsonpog-integration-master-examples/examples/egmScaleAndSmearingExample.py
"""
import awkward as ak
import correctionlib.schemav2 as cs
from modules.correctionlib_file_cache import get_corrset

# Standard normal per electron, deterministic across reruns and Dask partitions
_gauss = cs.Correction(
    name="ele_ss_rng",
    description="Deterministic N(0,1) for electron smearing",
    version=1,
    inputs=[cs.Variable(name=n, type="real") for n in ("event", "lumi", "pt", "eta", "phi")],
    output=cs.Variable(name="rng", type="real"),
    data=cs.HashPRNG(
        nodetype="hashprng",
        inputs=["event", "lumi", "pt", "eta", "phi"],
        distribution="normal",
    ),
).to_evaluator()


def apply_electron_ss(events, ss_file: str, is_mc: bool):
    """
    Overwrite Electron pt (and mass) with the scale-corrected (data) or smeared (MC) value.
    The original pt is kept as Electron.pt_uncorr. MC also gets pt_ss_{smear,scale}_{up,down}.
    """
    cset = get_corrset(ss_file)
    ele = events.Electron
    pt, r9, sc_eta = ele.pt, ele.r9, ele.superclusterEta
    events["Electron", "pt_uncorr"] = pt

    if is_mc:
        smear_syst = cset["SmearAndSyst"]
        smear = smear_syst.evaluate("smear", pt, r9, sc_eta)
        esmear = smear_syst.evaluate("esmear", pt, r9, sc_eta)
        escale = smear_syst.evaluate("escale", pt, r9, sc_eta)
        event, lumi = ak.broadcast_arrays(events.event, events.luminosityBlock, pt)[:2]
        n = _gauss.evaluate(event * 1.0, lumi * 1.0, pt, ele.eta, ele.phi)
        factor = 1 + smear * n
        events["Electron", "pt_ss_smear_up"] = pt * (1 + (smear + esmear) * n)
        events["Electron", "pt_ss_smear_down"] = pt * (1 + (smear - esmear) * n)
        events["Electron", "pt_ss_scale_up"] = pt * factor * (1 + escale)
        events["Electron", "pt_ss_scale_down"] = pt * factor * (1 - escale)
    else:
        run = ak.broadcast_arrays(events.run, pt)[0] * 1.0  # payload expects real
        factor = cset.compound["Scale"].evaluate("scale", run, sc_eta, r9, pt, ele.seedGain * 1.0)

    events["Electron", "pt"] = pt * factor
    events["Electron", "mass"] = ele.mass * factor
