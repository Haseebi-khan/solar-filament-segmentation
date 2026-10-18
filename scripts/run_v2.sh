#!/usr/bin/env bash
# v2 pipeline, run stage by stage (each stage skips work that is already on disk):
#
#   scripts/run_v2.sh selector-v1   # extent selector on the 3 trained v1 members -> submissions/submission_v2_v1members.csv
#   scripts/run_v2.sh probes        # leaderboard probes: v1 submission with masks shrunk / grown by 1 px
#   scripts/run_v2.sh targets       # 2048 px soft targets for the new member
#   scripts/run_v2.sh train         # trains r34_2048 and effv2s_1536, 5 folds each, 2 folds at a time
#   scripts/run_v2.sh predict       # OOF + test probability maps of the new members, both members at once
#   scripts/run_v2.sh search        # blend-weight search over all 5 members (log only)
#   WEIGHTS="1 1 2 2 2" scripts/run_v2.sh final   # selector on all 5 members -> submissions/submission_v2.csv
#   scripts/run_v2.sh evaluate      # rubric metrics + figures for every OOF CSV in work/ -> work/eval_v2/
#
# Resources: PARALLEL (default 2) GPU jobs at once, WORKERS (default: all cores but one) CPU processes,
# TTA (default 8) flip/rotation views at prediction time. Every stage's wall time is appended to
# logs/timing_v2.txt (the organisers also judge end-to-end computational efficiency).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
export FILSEG_WORK=${FILSEG_WORK:-$ROOT/work}
export PARALLEL=${PARALLEL:-2}
export WORKERS=${WORKERS:-$(( $(nproc) - 1 ))}
TTA=${TTA:-8}
V1_MEMBERS="r34_1024 rgy040_1024 r34_1536"
V2_MEMBERS=${MEMBERS_V2:-"r34_2048 effv2s_1536"}
STAGE=${1:?usage: scripts/run_v2.sh selector-v1|probes|targets|train|predict|search|final|evaluate}
mkdir -p "$ROOT/logs"
START=$SECONDS

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
    if [ -f "$FILSEG_WORK/cache/train_soft_2048.npy" ]; then echo "train_soft_2048.npy exists"
    else "$PY" "$ROOT/src/prepare_targets_v2.py" --sizes 2048; fi ;;
  train)
    MEMBERS_V2="$V2_MEMBERS" "$ROOT/scripts/train_queue_v2.sh" ;;
  predict)
    pids=()
    for m in $V2_MEMBERS; do
      if [ -f "$FILSEG_WORK/preds/$m/test_folds.npy" ]; then echo "skip $m (maps exist)"; continue; fi
      echo "predicting $m (TTA $TTA) -> logs/predict_v2_$m.out"
      "$PY" "$ROOT/src/predict.py" --name "$m" --folds 0 1 2 3 4 --tta "$TTA" > "$ROOT/logs/predict_v2_$m.out" 2>&1 &
      pids+=("$!")
      while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 10; done
    done
    for p in "${pids[@]}"; do wait "$p" || echo "a prediction job failed: see logs/predict_v2_*.out"; done ;;
  search)
    NAMES="$V1_MEMBERS $V2_MEMBERS" "$ROOT/scripts/final_ensemble_v2.sh" ;;
  final)
    : "${WEIGHTS:?set WEIGHTS to one value per member, e.g. WEIGHTS=\"1 1 2 2 2\" (pick from the search log)}"
    NAMES="$V1_MEMBERS $V2_MEMBERS" WEIGHTS="$WEIGHTS" "$ROOT/scripts/final_ensemble_v2.sh" ;;
  evaluate)
    csvs=(); labels=()
    v1=$FILSEG_WORK/oof_sub_r34_1024+rgy040_1024+r34_1536.csv
    if [ -f "$v1" ]; then csvs+=("$v1"); labels+=("v1_rescorer_in_sample"); fi
    for f in "$FILSEG_WORK"/oof_sub_v2_*.csv; do
      [ -f "$f" ] || continue
      csvs+=("$f"); labels+=("$(basename "$f" .csv | sed 's/^oof_sub_//')_nested")
    done
    [ ${#csvs[@]} -gt 0 ] || { echo "no OOF CSVs in $FILSEG_WORK yet"; exit 1; }
    "$PY" "$ROOT/src/evaluate_v2.py" "${csvs[@]}" --labels "${labels[@]}" --out "$FILSEG_WORK/eval_v2" ;;
  *)
    echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "$(date '+%F %T') stage $STAGE${WEIGHTS:+ (weights $WEIGHTS)}: $(( SECONDS - START )) s" | tee -a "$ROOT/logs/timing_v2.txt"
