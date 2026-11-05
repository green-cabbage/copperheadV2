#!/usr/bin/env python3
"""Table-9-style post-fit breakdown of the combined VBF+ggH fit.

Reads the four --algo singles rungs and forms the nested (cumulative) quadrature
differences. The four components close to the systematic row by telescoping:
  Experimental = sqrt(s0^2 - s1^2),  Theoretical = sqrt(s1^2 - s2^2),
  SimSize      = sqrt(s2^2 - s3^2),  Statistical = s3,
  Systematic   = sqrt(s0^2 - s3^2),  Total = s0.
"""
import json, math, sys, ROOT

# Revision 6: optional argv so the re-centred chain can be summarised without
# touching the published numbers.  argv[1] = results dir, argv[2] = nominal-rung
# file, argv[3] = output json name.  Defaults are revision 4's, unchanged.
W = sys.argv[1] if len(sys.argv) > 1 else "results/comb_vbf_ggh"
NOM = sys.argv[2] if len(sys.argv) > 2 else "higgsCombine.f2_tol.MultiDimFit.mH125.root"
OUTJ = sys.argv[3] if len(sys.argv) > 3 else "postfit_breakdown.json"
RUNGS = [("s0_nominal",   NOM),
         ("s1_exp",       "higgsCombine.q_p1_exp.MultiDimFit.mH125.root"),
         ("s2_exp_theo",  "higgsCombine.q_p2_exp_theo.MultiDimFit.mH125.root"),
         ("s3_all",       "higgsCombine.q_p3_all.MultiDimFit.mH125.root")]

def sym(path):
    f = ROOT.TFile(f"{W}/{path}"); t = f.Get("limit")
    best, vals = None, []
    for e in t:
        if e.quantileExpected == -1: best = e.r
        else: vals.append(e.r)
    if best is None or len(vals) != 2:
        sys.exit(f"FATAL {path}: best={best} vals={vals}")
    lo, hi = min(vals), max(vals)
    slo, shi = best - lo, hi - best
    if min(slo, shi) <= 1e-4 or max(slo, shi) > 1.5:
        sys.exit(f"FATAL {path}: implausible interval -{slo} +{shi} (range edge?)")
    return best, slo, shi, 0.5 * (slo + shi)

s, out = {}, {}
for name, path in RUNGS:
    best, slo, shi, ssym = sym(path)
    s[name] = ssym
    out[name] = {"r": best, "sigma_lo": slo, "sigma_hi": shi, "sigma_sym": ssym}
    print(f"{name:14s} r={best:.4f}  -{slo:.5f}/+{shi:.5f}  sym={ssym:.5f}")

def q(a, b):
    d = a * a - b * b
    if d < 0: sys.exit(f"FATAL negative quadrature difference {a} {b}")
    return math.sqrt(d)

rows = {
    "Total uncertainty":        s["s0_nominal"],
    "Statistical uncertainty":  s["s3_all"],
    "Systematic uncertainty":   q(s["s0_nominal"], s["s3_all"]),
    "Experimental":             q(s["s0_nominal"], s["s1_exp"]),
    "Theoretical":              q(s["s1_exp"], s["s2_exp_theo"]),
    "Size of simulated samples": q(s["s2_exp_theo"], s["s3_all"]),
}
clos = math.sqrt(rows["Experimental"]**2 + rows["Theoretical"]**2
                 + rows["Size of simulated samples"]**2) / rows["Systematic uncertainty"]
tot_clos = math.sqrt(rows["Systematic uncertainty"]**2 + rows["Statistical uncertainty"]**2) / rows["Total uncertainty"]
print()
for k, v in rows.items(): print(f"{k:28s} {v:.5f}")
print(f"\nsub-rows / systematic = {clos:.4f}   (syst+stat)/total = {tot_clos:.4f}")
json.dump({"channel": "combined VBF+ggH, post-fit", "rungs": out, "rows": rows,
           "closure_sub_over_syst": clos, "closure_total": tot_clos},
          open(f"{W}/{OUTJ}", "w"), indent=2)
print(f"\nwrote {W}/{OUTJ}")
