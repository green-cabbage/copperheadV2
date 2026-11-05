#!/bin/bash
# Re-centred post-fit chain: same as run_combined_postfit.sh + run_combined_nll.sh,
# but the 95 global observables <name>_In are set to the same H-sideband values
# that assignValueOnly gave the nuisances, so the post-fit Asimov is internally
# consistent and r_hat should land on 1.000 instead of 1.01727.
# Plan + null controls: recentre_globals_plan.md
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${here}/results/comb_vbf_ggh"
OUT="${here}/results/comb_vbf_ggh_recentred"
E="/work/users/yun79/sideHustle2/copperheadV2/.pixi/envs/combine"
export PATH="$E/bin:$PATH" LD_LIBRARY_PATH="$E/lib:${LD_LIBRARY_PATH:-}" ROOTSYS="$E"
export ROOT_INCLUDE_PATH="$E/include"
export CPLUS_INCLUDE_PATH="$E/x86_64-conda-linux-gnu/sysroot/usr/include:$E/include"
mkdir -p "$OUT" && cd "$OUT"
S="${OUT}/STATE"; echo "start $(date -u)" > "$S"
printf '{"r": "#mu"}\n' > poi_translate.json
cp "${SRC}/postfit_rungs.tsv" .

IN=$(awk '{printf "%s_In=%s,", $1, $2}' "${SRC}/sb_postfit_values.txt" | sed 's/,$//')
NIN=$(awk 'END{print NR}' "${SRC}/sb_postfit_values.txt")
echo "globals to re-centre: $NIN" >> "$S"
[ "$NIN" -eq 95 ] || { echo "FATAL expected 95 globals, got $NIN" >> "$S"; exit 1; }

WS="${SRC}/comb_vbf_ggh_postfit.root"
IDLE="BWZ_Redux_a_coeff,BWZ_Redux_b_coeff,BWZ_Redux_c_coeff,FEWZxBern_c2,FEWZxBern_c3,FEWZxBern_c4"
BASE="MH,pdf_index_ggh,${IDLE}"
SET="pdf_index_ggh=0,${IN}"

# ---- A. NLL curve, 100-point grid (the published post-fit settings) ----------
echo "A: nll grid" >> "$S"
combine -M MultiDimFit "$WS" -m 125 --algo grid --points 100 -t -1 --expectSignal 1 \
  --snapshotName CombVBFSBPostfit --setParameterRanges r=0,3 --setParameters "$SET" \
  --robustFit 1 --setRobustFitTolerance 0.2 --cminDefaultMinimizerStrategy 1 \
  --freezeParameters "$BASE" --saveNLL -n .cnll_postfit_rc > cnll_postfit_rc.log 2>&1 &

# ---- B. breakdown rungs, published $C verbatim + $IN -------------------------
C="-M MultiDimFit $WS -m 125 --algo singles -t -1 --expectSignal 1 \
   --snapshotName CombVBFSBPostfit --setParameterRanges r=0,3 --setParameters $SET \
   --robustFit 1 --setRobustFitTolerance 0.2 --cminDefaultMinimizerStrategy 1"
echo "B: rungs" >> "$S"
( combine $C --freezeParameters "$BASE" -n .f2_tol > f2_tol.log 2>&1 ) &
while IFS=$'\t' read -r name frz vals; do
  [ "$name" = "p0_nominal" ] && continue
  ( combine $C --freezeParameters "${BASE},${frz}" -n ".q_${name}" > "q_${name}.log" 2>&1 ) &
done < postfit_rungs.tsv

# ---- C. nominal rung with plain strategy 0, no tolerance override ------------
# the published chain needed tol 0.2 + strategy 1 because the upward robustFit
# scan died on the inconsistent Asimov; a self-consistent one may not need it.
echo "C: nominal strategy 0" >> "$S"
( combine -M MultiDimFit "$WS" -m 125 --algo singles -t -1 --expectSignal 1 \
    --snapshotName CombVBFSBPostfit --setParameterRanges r=0,3 --setParameters "$SET" \
    --robustFit 1 --cminDefaultMinimizerStrategy 0 --freezeParameters "$BASE" \
    -n .f0_s0 > f0_s0.log 2>&1 ) &
wait
echo "combine done $(date -u)" >> "$S"

# ---- D. guard: the flag must have been applied, 95 times, in every job ------
for f in cnll_postfit_rc f2_tol q_p1_exp q_p2_exp_theo q_p3_all f0_s0; do
  n=$(grep -c "Set Default Value of Parameter .*_In To" "${f}.log" || true)
  echo "guard ${f}: ${n} _In set" >> "$S"
done
echo "all done $(date -u)" >> "$S"
