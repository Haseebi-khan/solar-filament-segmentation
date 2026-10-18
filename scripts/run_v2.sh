#!/usr/bin/env bash
# v2 pipeline, run stage by stage (each stage skips work that is already on disk):
#
#   scripts/run_v2.sh selector-v1   # extent selector on the 3 trained v1 members -> submissions/submission_v2_v1members.csv
#   scripts/run_v2.sh probes        # leaderboard probes: v1 submission with masks shrunk / grown by 1 px
#   scripts/run_v2.sh targets       # 2048 px soft targets for the new member
#   scripts/run_v2.sh train         # trains r34_2048 and effv2s_1536, 5 folds each (~9-10 h on one MI300X)
#   scripts/run_v2.sh predict       # OOF + test probability maps of the new members (8x TTA)
#   scripts/run_v2.sh search        # blend-weight search over all 5 members (log only)
#   WEIGHTS="1 1 2 2 2" scripts/run_v2.sh final   # selector on all 5 members -> submissions/submission_v2.csv
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
export FILSEG_WORK=${FILSEG_WORK:-$ROOT/work}
V1_MEMBERS="r34_1024 rgy040_1024 r34_1536"
V2_MEMBERS=${MEMBERS_V2:-"r34_2048 effv2s_1536"}
STAGE=${1:?usage: scripts/run_v2.sh selector-v1|probes|targets|train|predict|search|final}

case "$STAGE" in
  selector-v1)
    NAMES="$V1_MEMBERS" WEIGHTS="1 1 2" OUT="$ROOT/submissions/submission_v2_v1members.csv" \
      "$ROOT/scripts/final_ensemble_v2.sh" ;;
  probes)
    for r in -1 1; do
      n=$([ "$r" -lt 0 ] && echo erode1 || echo dilate1)
      "$PY" "$ROOT/tools/morph_submission_v2.py" "$ROOT/submissions/submission.csv" --radius "$r" \
        --out "$ROOT/submissions/submission_v2_probe_$n.csv"
    done ;;
  targets)
    [ -f "$FILSEG_WORK/cache/train_soft_2048.npy" ] && echo "train_soft_2048.npy exists" \
      || "$PY" "$ROOT/src/prepare_targets_v2.py" --sizes 2048 ;;
  train)
    MEMBERS_V2="$V2_MEMBERS" "$ROOT/scripts/train_queue_v2.sh" ;;
  predict)
    for m in $V2_MEMBERS; do
      if [ -f "$FILSEG_WORK/preds/$m/test_folds.npy" ]; then echo "skip $m (maps exist)"; continue; fi
      "$PY" "$ROOT/src/predict.py" --name "$m" --folds 0 1 2 3 4 --tta 8
    done ;;
  search)
    NAMES="$V1_MEMBERS $V2_MEMBERS" "$ROOT/scripts/final_ensemble_v2.sh" ;;
  final)
    : "${WEIGHTS:?set WEIGHTS to one value per member, e.g. WEIGHTS=\"1 1 2 2 2\" (pick from the search log)}"
    NAMES="$V1_MEMBERS $V2_MEMBERS" WEIGHTS="$WEIGHTS" "$ROOT/scripts/final_ensemble_v2.sh" ;;
  *)
    echo "unknown stage $STAGE"; exit 1 ;;
esac
