#!/usr/bin/env bash
# Final stage on the out-of-fold maps of the ensemble members.
#   scripts/final_ensemble.sh                 -> search blend weights (writes logs/final_<members>.txt)
#   WEIGHTS="1 1 2" scripts/final_ensemble.sh -> rescorer on that blend, OOF CSV score, test submission
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
WORK=${FILSEG_WORK:-$ROOT/work}
read -r -a NAMES <<< "${NAMES:-r34_1024 rgy040_1024 r34_1536}"
TAG=$(IFS=+; echo "${NAMES[*]}")
LOG=$ROOT/logs/final_${TAG}.txt
STRUCT=(--seeds 0.6 --masks 0.6 --links 8 --closes -1)

if [ -z "${WEIGHTS:-}" ]; then
  echo "== blend weights (t_seed 0.6, t_mask 0.6, link 8, fill holes; best area/score filter)" | tee "$LOG"
  for W in "1 1 1" "1 1 2" "0 1 2" "1 0 2" "0 0 1"; do
    r=$("$PY" "$ROOT/src/tune_postprocess.py" --names "${NAMES[@]}" --weights $W "${STRUCT[@]}" \
        --out "$WORK/preds/tmp_final.parquet" 2>&1 | grep -v -i warn | sed -n 4p)
    echo "weights $W: $r" | tee -a "$LOG"
  done
  exit 0
fi

read -r -a W <<< "$WEIGHTS"
mkdir -p "$WORK/rescorers" "$ROOT/submissions"
echo "== final: members ${NAMES[*]}  weights ${W[*]}" | tee -a "$LOG"
"$PY" "$ROOT/src/train_rescorer.py" --names "${NAMES[@]}" --weights "${W[@]}" --t-seed 0.6 --t-mask 0.6 --link 8 \
  --min-area 100 --out "$WORK/rescorers/$TAG.pkl" 2>&1 | grep -v -i warn | tee -a "$LOG"
"$PY" - "$WORK/rescorers/${TAG}_config.json" "${W[*]}" <<'EOF'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["weights"] = [float(x) for x in sys.argv[2].split()]
json.dump(c, open(p, "w"), indent=1)
EOF
"$PY" "$ROOT/src/make_submission.py" --names "${NAMES[@]}" --weights "${W[@]}" --config "$WORK/rescorers/${TAG}_config.json" \
  --oof --out "$WORK/oof_sub_${TAG}.csv" 2>&1 | grep -v -i warn | sed 's/^/[OOF, rescorer in-sample] /' | tee -a "$LOG"
"$PY" "$ROOT/src/make_submission.py" --names "${NAMES[@]}" --weights "${W[@]}" --config "$WORK/rescorers/${TAG}_config.json" \
  --out "$ROOT/submissions/submission.csv" 2>&1 | grep -v -i warn | tee -a "$LOG"
