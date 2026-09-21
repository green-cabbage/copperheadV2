"""Official UL electron reconstruction and Fall17V2 WP90 ID factors."""
from functools import lru_cache

import awkward as ak
import correctionlib
import numpy as np


@lru_cache(maxsize=4)
def _electron_correction(filename):
    return correctionlib.CorrectionSet.from_file(filename)["UL-Electron-ID-SF"]


def electron_sfs(electrons, filename, year):
    """Multiply selected-electron factors; zero-electron events get unity."""
    counts = ak.to_numpy(ak.num(electrons))
    pt = ak.to_numpy(ak.flatten(electrons.pt))
    eta_sc = ak.to_numpy(ak.flatten(electrons.eta + electrons.deltaEtaSC))
    result = {}
    for name, working_point in [("electronReco", "RecoAbove20"), ("electronID", "wp90iso")]:
        result[name] = {}
        for variation, value_type in [("nom", "sf"), ("up", "sfup"), ("down", "sfdown")]:
            values = (_electron_correction(filename).evaluate(str(year), value_type, working_point, eta_sc, pt)
                      if len(pt) else np.empty(0))
            result[name][variation] = ak.prod(ak.unflatten(values, counts), axis=1)
    return result
