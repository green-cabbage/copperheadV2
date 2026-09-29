"""UParT fixed-WP Loose/Medium event weights with chunk-local MC efficiencies."""

import awkward as ak
import numpy as np


FLAVOURS = np.array([0, 4, 5])
PT_EDGES = np.array([20., 30., 50., 70., 100., 140., 200., 300., 600., 1000., np.inf])
ETA_EDGES = np.array([0., 1.2, 2.4, 2.5])


def efficiency_bin_indices(jets):
    """Index finite selected jets with flavour 0/4/5, pT >= 20 and |eta| < 2.5.
    Each tuple element follows the same flattened event/jet order."""
    flat = ak.flatten(jets, axis=1)
    return (
        np.searchsorted(FLAVOURS, ak.to_numpy(flat.hadronFlavour)),
        np.searchsorted(PT_EDGES, ak.to_numpy(flat.pt), side="right") - 1,
        np.searchsorted(ETA_EDGES, np.abs(ak.to_numpy(flat.eta)), side="right") - 1,
    )


def derive_mc_efficiencies(jets, WP_L, WP_M):
    """Derive unweighted flavour/pT/|eta| efficiencies from this pre-veto chunk.
    Requires WP_L < WP_M; score -1 fails both WPs and remains in the denominator."""
    indices = efficiency_bin_indices(jets)
    score = ak.to_numpy(ak.flatten(jets.btagUParTAK4B), allow_missing=False)
    shape = (len(FLAVOURS), len(PT_EDGES) - 1, len(ETA_EDGES) - 1)
    N_total = np.zeros(shape, dtype=np.int64)
    N_L = np.zeros(shape, dtype=np.int64)
    N_M = np.zeros(shape, dtype=np.int64)

    # Count every repeated bin index; Medium tags also contribute to Loose.
    np.add.at(N_total, indices, 1)
    np.add.at(N_L, indices, score > WP_L)
    np.add.at(N_M, indices, score > WP_M)
    epsilon_L = np.divide(N_L, N_total, out=np.full(shape, np.nan), where=N_total > 0)
    epsilon_M = np.divide(N_M, N_total, out=np.full(shape, np.nan), where=N_total > 0)
    return dict(N_total=N_total, N_L=N_L, N_M=N_M, epsilon_L=epsilon_L, epsilon_M=epsilon_M)


def lookup_mc_efficiencies(jets, efficiency_maps):
    """Restore per-jet efficiencies in the original jagged event layout.
    Requires populated map cells for every supplied jet."""
    indices = efficiency_bin_indices(jets)
    counts = ak.to_numpy(ak.num(jets, axis=1))
    epsilon_L = ak.unflatten(efficiency_maps["epsilon_L"][indices], counts)
    epsilon_M = ak.unflatten(efficiency_maps["epsilon_M"][indices], counts)
    return epsilon_L, epsilon_M


def jet_sf(btvjson, jets, wp, bc_syst="central", light_syst="central"):
    """Evaluate UParT SFs with independent b/c and light systematic labels.
    Original jet coordinates remain unchanged for selection and MC efficiencies."""
    counts = ak.to_numpy(ak.num(jets, axis=1))
    flat = ak.flatten(jets, axis=1)
    jet_pt = ak.to_numpy(flat.pt)
    jet_eta = np.abs(ak.to_numpy(flat.eta)).astype(np.float64)
    jet_flav = ak.to_numpy(flat.hadronFlavour)
    light_jets = np.where(jet_flav == 0)
    bc_jets = np.where(jet_flav != 0)
    sf = np.ones(len(flat), dtype=float)
    # NOTE: analysis choice: clip lookup |eta| to the b/c (2.4) or light (2.5) limit.
    # Use just below the exclusive edge; preserve original jets and efficiency bins.
    if bc_jets[0].size:
        bc_eta = np.minimum(jet_eta[bc_jets], np.nextafter(2.4, 0.0))
        sf[bc_jets] = btvjson["UParTAK4_comb"].evaluate(
            bc_syst, wp, jet_flav[bc_jets], bc_eta, jet_pt[bc_jets]
        )
    if light_jets[0].size:
        light_eta = np.minimum(jet_eta[light_jets], np.nextafter(2.5, 0.0))
        sf[light_jets] = btvjson["UParTAK4_light"].evaluate(
            light_syst, wp, jet_flav[light_jets], light_eta, jet_pt[light_jets]
        )
    return ak.unflatten(sf, counts)


def event_weight_method1a(tagged_at_L, tagged_at_M, SF_L, SF_M, epsilon_L, epsilon_M):
    """Multiply the Medium, Loose-only and untagged probability ratios.
    Requires matching jagged arrays and nonzero observed-category denominators."""
    # https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/#scale-factor-recommendations-for-event-reweighting
    i = tagged_at_M
    j = tagged_at_L & ~tagged_at_M
    k = ~tagged_at_L
    factor_i = (SF_M[i] * epsilon_M[i]) / epsilon_M[i]
    numerator_j = (SF_L[j] * epsilon_L[j]) - (SF_M[j] * epsilon_M[j])
    denominator_j = epsilon_L[j] - epsilon_M[j]
    factor_j = ak.where(numerator_j < 0, 1.0, numerator_j / denominator_j)
    factor_k = (1 - SF_L[k] * epsilon_L[k]) / (1 - epsilon_L[k])
    return ak.prod(factor_i, axis=1) * ak.prod(factor_j, axis=1) * ak.prod(factor_k, axis=1)


def btag_weights_LM(btvjson, jets, WP_L, WP_M, era):
    """Return nominal L+M weights and fixed-WP uncertainty multipliers, before veto.
    Requires matching WPs and nonzero nominal weights; invalid ratios raise."""
    efficiency_maps = derive_mc_efficiencies(jets, WP_L, WP_M)
    epsilon_L, epsilon_M = lookup_mc_efficiencies(jets, efficiency_maps)
    tagged_at_L = jets.btagUParTAK4B > WP_L
    tagged_at_M = jets.btagUParTAK4B > WP_M

    def event_weight(bc_syst="central", light_syst="central"):
        SF_L = jet_sf(btvjson, jets, "L", bc_syst, light_syst)
        SF_M = jet_sf(btvjson, jets, "M", bc_syst, light_syst)
        return event_weight_method1a(
            tagged_at_L, tagged_at_M, SF_L, SF_M, epsilon_L, epsilon_M,
        )

    btag_wgt = event_weight()
    btag_syst = {}
    # Share correlated sources across eras; keep pre/postVFP statistical sources separate.
    for flavour in ("bc", "light"):
        for correlation in ("correlated", "uncorrelated"):
            name = f"btagSF{flavour}_{'correlated' if correlation == 'correlated' else era}"
            btag_syst[name] = {}
            for direction in ("up", "down"):
                label = f"{direction}_{correlation}"
                varied = event_weight(**{f"{flavour}_syst": label})
                # NOTE: the processor already multiplies by nominal btag_wgt;
                # supply varied/nominal so each modifier replaces that factor once.
                with np.errstate(divide="raise", invalid="raise"):
                    btag_syst[name][direction] = varied / btag_wgt
    return btag_wgt, btag_syst
