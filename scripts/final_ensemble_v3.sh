#!/usr/bin/env bash
# v3 final stage: final_ensemble_v2.sh (extent selector, exact OOF score, rubric, test CSV) with the seed threshold
# and the extent levels as parameters. v3-loss members are calibrated lower than v1/v2 members, so blends that
# contain them need T_SEED 0.4-0.5 (see src/blend_search_tuned_v3.py); final_ensemble_v2.sh always uses 0.6.
#   NAMES="..." WEIGHTS="..." T_SEED=0.4 LEVELS="0.3 0.4 0.5 0.6" scripts/final_ensemble_v3.sh
# Outputs carry a _t<T_SEED> tag, so they never overwrite the v2 files. OUT=<csv> for the submission
# (default submissions/submission_v3_tuned_t<T_SEED>.csv); WORKERS=<n> CPU processes.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
WORK=${FILSEG_WORK:-$ROOT/work}
read -r -a NAMES <<< "${NAMES:?set NAMES}"
read -r -a W <<< "${WEIGHTS:?set WEIGHTS}"
T_SEED=${T_SEED:-0.4}
read -r -a LV <<< "${LEVELS:-0.3 0.4 0.5 0.6}"
WORKERS=${WORKERS:-$(( $(nproc) - 1 ))}
[ ${#W[@]} -eq ${#NAMES[@]} ] || { echo "WEIGHTS needs ${#NAMES[@]} values"; exit 1; }
printf '%s\n' "${LV[@]}" | grep -qx "$T_SEED" || { echo "T_SEED ($T_SEED) must be one of LEVELS (${LV[*]})"; exit 1; }
TAG="$(IFS=+; echo "${NAMES[*]}")_t${T_SEED}"
LOG=$ROOT/logs/final_v3_tuned_${TAG}.txt
SEL=$WORK/selectors_v2/$TAG
OOF_CSV=$WORK/oof_sub_v3_tuned_${TAG}.csv
OUT=${OUT:-$ROOT/submissions/submission_v3_tuned_t${T_SEED}.csv}
mkdir -p "$ROOT/logs"

echo "== v3 tuned final: members ${NAMES[*]}  weights ${W[*]}  t_seed ${T_SEED}  levels ${LV[*]}" | tee -a "$LOG"
"$PY" "$ROOT/src/train_selector_v2.py" --names "${NAMES[@]}" --weights "${W[@]}" --t-seed "$T_SEED" \
  --levels "${LV[@]}" --workers "$WORKERS" --out "$SEL.pkl" 2>&1 | grep -v -i warn | tee -a "$LOG"
"$PY" "$ROOT/src/make_submission_v2.py" --config "${SEL}_config.json" --oof --nested-scores "${SEL}_nested.parquet" \
  --workers "$WORKERS" --out "$OOF_CSV" 2>&1 | grep -v -i warn | sed 's/^/[OOF] /' | tee -a "$LOG"
"$PY" "$ROOT/src/evaluate_v2.py" "$OOF_CSV" --labels "v3_tuned_${TAG}" --out "$WORK/eval_v3" 2>&1 \
  | grep -v -i warn | sed 's/^/[rubric] /' | tee -a "$LOG"
"$PY" "$ROOT/src/make_submission_v2.py" --config "${SEL}_config.json" --workers "$WORKERS" \
  --out "$OUT" 2>&1 | grep -v -i warn | tee -a "$LOG"
echo "submission: $OUT (nested OOF PQ in $LOG; compare with: $PY src/compare_v3.py <baseline_oof.csv> $OOF_CSV)" \
  | tee -a "$LOG"
