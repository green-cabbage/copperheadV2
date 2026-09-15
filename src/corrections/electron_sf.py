"""Official UL2018 electron reconstruction and Fall17V2 WP90 ID factors."""
from functools import lru_cache

import awkward as ak
import correctionlib
import numpy as np


ELECTRON_SF_2018 = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2018_UL/electron.json.gz"


@lru_cache(maxsize=1)
def _electron_correction():
    return correctionlib.CorrectionSet.from_file(ELECTRON_SF_2018)["UL-Electron-ID-SF"]


def electron_sfs_2018(electrons):
    """Multiply selected-electron factors; zero-electron events get unity."""
    counts = ak.to_numpy(ak.num(electrons))
    pt = ak.to_numpy(ak.flatten(electrons.pt))
    eta_sc = ak.to_numpy(ak.flatten(electrons.eta + electrons.deltaEtaSC))
    result = {}
    for name, working_point in [("electronReco", "RecoAbove20"), ("electronID", "wp90iso")]:
        result[name] = {}
        for variation, value_type in [("nom", "sf"), ("up", "sfup"), ("down", "sfdown")]:
            values = (_electron_correction().evaluate("2018", value_type, working_point, eta_sc, pt)
                      if len(pt) else np.empty(0))
            result[name][variation] = ak.prod(ak.unflatten(values, counts), axis=1)
    return result
