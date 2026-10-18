#!/usr/bin/env bash
# v2 final stage on the out-of-fold maps of the ensemble members.
#   NAMES="..." scripts/final_ensemble_v2.sh                   -> blend-weight search (logs/final_v2_<members>.txt)
#   NAMES="..." WEIGHTS="..." scripts/final_ensemble_v2.sh     -> extent selector, exact OOF score, test CSV
# Optional: WEIGHT_GRID="1 1 2 0 0,1 1 2 2 2" (comma-separated) for the search, OUT=<csv> for the submission.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
WORK=${FILSEG_WORK:-$ROOT/work}
read -r -a NAMES <<< "${NAMES:-r34_1024 rgy040_1024 r34_1536 r34_2048 effv2s_1536}"
TAG=$(IFS=+; echo "${NAMES[*]}")
LOG=$ROOT/logs/final_v2_${TAG}.txt
OUT=${OUT:-$ROOT/submissions/submission_v2.csv}
mkdir -p "$ROOT/logs"

if [ -z "${WEIGHTS:-}" ]; then
  if [ ${#NAMES[@]} -eq 5 ]; then
    DEFAULT_GRID="1 1 2 0 0,1 1 2 2 0,1 1 2 0 2,1 1 2 2 2,0 0 1 1 1,0 0 1 2 1,1 1 1 1 1,0 0 0 1 1"
  else
    DEFAULT_GRID=$(printf '1 %.0s' "${NAMES[@]}")
  fi
  IFS=',' read -r -a GRID <<< "${WEIGHT_GRID:-$DEFAULT_GRID}"
  echo "== blend weights for ${NAMES[*]} (t_seed 0.6, t_mask 0.6, link 8, fill holes; best area/score filter, no selector)" | tee "$LOG"
  for W in "${GRID[@]}"; do
    # shellcheck disable=SC2086
    r=$("$PY" "$ROOT/src/tune_postprocess.py" --names "${NAMES[@]}" --weights $W --seeds 0.6 --masks 0.6 \
        --links 8 --closes -1 --out "$WORK/preds/tmp_final_v2.parquet" 2>&1 | grep -v -i warn | sed -n 4p)
    echo "weights $W: $r" | tee -a "$LOG"
  done
  echo "pick the best weights (last-but-one column is PQ) and rerun with WEIGHTS=\"...\"" | tee -a "$LOG"
  exit 0
fi

read -r -a W <<< "$WEIGHTS"
[ ${#W[@]} -eq ${#NAMES[@]} ] || { echo "WEIGHTS needs ${#NAMES[@]} values"; exit 1; }
SEL=$WORK/selectors_v2/$TAG
echo "== v2 final: members ${NAMES[*]}  weights ${W[*]}" | tee -a "$LOG"
"$PY" "$ROOT/src/train_selector_v2.py" --names "${NAMES[@]}" --weights "${W[@]}" --out "$SEL.pkl" 2>&1 \
  | grep -v -i warn | tee -a "$LOG"
"$PY" "$ROOT/src/make_submission_v2.py" --config "${SEL}_config.json" --oof --nested-scores "${SEL}_nested.parquet" \
  --out "$WORK/oof_sub_v2_${TAG}.csv" 2>&1 | grep -v -i warn | sed 's/^/[OOF] /' | tee -a "$LOG"
"$PY" "$ROOT/src/make_submission_v2.py" --config "${SEL}_config.json" --out "$OUT" 2>&1 \
  | grep -v -i warn | tee -a "$LOG"
