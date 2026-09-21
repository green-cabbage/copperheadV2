"""Recover an electron's linked NanoAOD FSR photon with explicit flavor cuts."""
import awkward as ak
import numpy as np


def recover_electron_fsr(electrons, photons, cuts):
    """Return dressed p4 coordinates and a mask, retaining unmatched electrons."""
    index = electrons.fsrPhotonIdx
    valid = (index >= 0) & (index < ak.num(photons, axis=1))
    photon = ak.pad_none(photons, 1)[ak.mask(index, valid)]
    recover = ak.fill_none(
        valid
        & (photon.pt > cuts["photon_pt_min"])
        & (abs(photon.eta) < cuts["photon_abs_eta_max"])
        & (photon.relIso03 < cuts["photon_rel_iso_max"]),
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
