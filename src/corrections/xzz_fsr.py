"""Recover an electron's linked NanoAOD FSR photon with explicit flavor cuts."""
import awkward as ak
import numpy as np


def photon_quality_mask(leptons, photon, cuts):
    """Apply optional reference FSR quality cuts after photon association."""
    mask = ak.ones_like(photon.pt, dtype=bool)
    if cuts.get("photon_droveret2_max") is not None:
        mask = mask & (photon.dROverEt2 < cuts["photon_droveret2_max"])
    if cuts.get("photon_dr_max") is not None:
        dphi = np.arctan2(np.sin(leptons.phi - photon.phi),
                         np.cos(leptons.phi - photon.phi))
        dr = np.hypot(leptons.eta - photon.eta, dphi)
        mask = mask & (dr < cuts["photon_dr_max"])
    return ak.fill_none(mask, False)


def associated_photon(leptons, photons, cuts):
    """Resolve NanoAOD association or the explicit reference-scan diagnostic."""
    mode = cuts.get("association", "nanoAOD")
    if mode == "reference_scan":
        pairs = ak.cartesian({"lepton": leptons, "photon": photons}, axis=1, nested=True)
        photon = pairs.photon
        eligible = (
            (photon.pt > cuts["photon_pt_min"])
            & (abs(photon.eta) < cuts["photon_abs_eta_max"])
            & (photon.relIso03 < cuts["photon_rel_iso_max"])
            & photon_quality_mask(pairs.lepton, photon, cuts)
        )
        ranked = ak.argsort(ak.where(eligible, photon.dROverEt2, np.inf), axis=2, stable=True)
        return ak.firsts(ak.mask(photon, eligible)[ranked], axis=2)
    if mode != "nanoAOD":
        raise ValueError(f"Unknown XZZ FSR association: {mode}")
    index = leptons.fsrPhotonIdx
    valid = (index >= 0) & (index < ak.num(photons, axis=1))
    return ak.pad_none(photons, 1)[ak.mask(index, valid)]


def recover_electron_fsr(electrons, photons, cuts):
    """Return dressed p4 coordinates and a mask, retaining unmatched electrons."""
    photon = associated_photon(electrons, photons, cuts)
    recover = ak.fill_none(
        (photon.pt > cuts["photon_pt_min"])
        & (abs(photon.eta) < cuts["photon_abs_eta_max"])
        & (photon.relIso03 < cuts["photon_rel_iso_max"])
        & photon_quality_mask(electrons, photon, cuts),
        False,
    )

    def momentum(obj):
        pt, eta, phi = (ak.values_astype(obj[k], np.float64) for k in ("pt", "eta", "phi"))
        return pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)

    lx, ly, lz = momentum(electrons)
    gx, gy, gz = momentum(photon)
    mass = ak.values_astype(electrons.mass, np.float64)
    # Preserve the signed mass-squared encoded by NanoAOD/ROOT, including tiny
    # negative electron masses arising from the serialized input four-vector.
    energy = np.sqrt(lx*lx + ly*ly + lz*lz + mass*abs(mass)) + np.sqrt(gx*gx + gy*gy + gz*gz)
    px, py, pz = lx + gx, ly + gy, lz + gz
    pt = np.hypot(px, py)
    mass2 = energy*energy - px*px - py*py - pz*pz
    dressed = {
        "pt": pt,
        "eta": np.arcsinh(pz / pt),
        "phi": np.arctan2(py, px),
        "mass": np.sign(mass2) * np.sqrt(abs(mass2)),
    }
    return {k: ak.where(recover, v, electrons[k]) for k, v in dressed.items()}, recover
