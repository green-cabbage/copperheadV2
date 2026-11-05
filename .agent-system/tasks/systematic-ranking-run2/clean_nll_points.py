#!/usr/bin/env python3
"""Drop badly-converged grid points from a combine --algo grid scan.

TSpline3 passes through every point, so a non-converged point is a visible
kink. Three traps this handles:

  * hadd of a coarse scan and a dense refill leaves DUPLICATE r values with
    different deltaNLL (seen at r=1.695). TSpline3 cannot represent that.
  * measuring deviation against a straight line through the neighbours counts
    the curve's real curvature as deviation. It cascades and eats good points.
    A local quadratic fit to the 6 nearest neighbours does not.
  * ONE threshold cannot serve both halves. The upper branch scatters an order
    of magnitude more than the minimum region, so a cut tight enough to catch a
    zigzag at mu~0.65 (|res| 0.025 where neighbours sit at 0.0007) deletes most
    of the upper branch and truncates the curve. Hence two criteria, OR'd:
      --thresh  absolute |res|, for gross outliers in the noisy upper branch
      --ratio   |res| / local MAD, for small zigzags in very smooth regions

Points within --protect of a 68% crossing are never dropped, so the quoted
uncertainty cannot move.

  python3 clean_nll_points.py IN.root OUT.root --crossings 0.537 1.520
"""
import argparse, array, numpy as np, ROOT

p = argparse.ArgumentParser()
p.add_argument("src"); p.add_argument("dst")
p.add_argument("--thresh", type=float, default=0.03)
p.add_argument("--ratio", type=float, default=4.0)
p.add_argument("--protect", type=float, default=0.08)
p.add_argument("--crossings", type=float, nargs="*", default=[])
p.add_argument("--rmin", type=float, default=0.0)
a = p.parse_args()

f = ROOT.TFile.Open(a.src); t = f.Get("limit")
raw = []
for i in range(t.GetEntries()):
    t.GetEntry(i)
    d = 2 * t.deltaNLL
    if d == d and t.r >= a.rmin:
        raw.append((t.r, d, t.deltaNLL))
f.Close()

seen = {}
for r, d, dn in sorted(raw):
    seen.setdefault(round(r, 4), (r, d, dn))
pts = [seen[k] for k in sorted(seen)]
print(f"{len(raw)} entries -> {len(pts)} after de-duplication on r")

X = np.array([q[0] for q in pts]); Y = np.array([q[1] for q in pts])
res = np.zeros(len(pts))
for i in range(len(pts)):
    idx = [j for j in range(max(0, i - 3), min(len(pts), i + 4)) if j != i]
    res[i] = Y[i] - np.polyval(np.polyfit(X[idx], Y[idx], 2), X[i])

scale = np.zeros(len(pts))
for i in range(len(pts)):
    w = res[max(0, i - 8):min(len(pts), i + 9)]
    scale[i] = max(np.median(np.abs(w - np.median(w))), 1e-4)
ratio = np.abs(res) / scale

def guarded(x):
    return any(abs(x - c) <= a.protect for c in a.crossings)

drop = [i for i in range(len(pts))
        if (abs(res[i]) > a.thresh or ratio[i] > a.ratio)
        and i > 1 and not guarded(X[i])]
print(f"dropping {len(drop)} of {len(pts)}  (|res|>{a.thresh} OR ratio>{a.ratio}; "
      f"protecting +-{a.protect} around {a.crossings})")
for i in drop:
    why = "abs" if abs(res[i]) > a.thresh else "ratio"
    print(f"   r={X[i]:7.3f}  y={Y[i]:9.4f}  res={res[i]:+8.4f}  ratio={ratio[i]:6.1f}  [{why}]")

keep = [pts[i] for i in range(len(pts)) if i not in drop]
fo = ROOT.TFile(a.dst, "RECREATE")
nt = ROOT.TTree("limit", "limit")
r_ = array.array('f', [0]); dn_ = array.array('f', [0]); q_ = array.array('f', [0])
nt.Branch("r", r_, "r/F"); nt.Branch("deltaNLL", dn_, "deltaNLL/F")
nt.Branch("quantileExpected", q_, "quantileExpected/F")
for rv, dv, dnv in keep:
    r_[0] = rv; dn_[0] = dnv; q_[0] = -1.0; nt.Fill()
nt.Write(); fo.Close()
print(f"kept {len(keep)} -> {a.dst}")
