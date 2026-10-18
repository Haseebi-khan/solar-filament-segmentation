#!/usr/bin/env bash
# v2: the whole pipeline in one command. Resumable: after an interruption, run it again and finished
# work (trained folds, probability maps, the selector on the v1 members) is skipped.
#
#   nohup scripts/run_all_v2.sh > logs/run_all_v2.out 2>&1 &
#   tail -f logs/run_all_v2.out          # stop everything: kill <pid printed on the first line>
#
# About 9-10 h on one AMD MI300X:
#   1. checks: GPU visible to PyTorch, v1 models / image cache / maps present; PARALLEL and WORKERS
#      are chosen from the hardware (GPU memory, CPU cores)
#   2. leaderboard probe files from the v1 submission (CPU, ~2 min)
#   3. GPU: trains r34_2048 and effv2s_1536, 5 folds each, PARALLEL folds at a time (~8 h)
#      CPU meanwhile: extent selector on the 3 v1 members -> submissions/submission_v2_v1members.csv
#      (this is also an early test of the v2 post-processing code, while the GPU is busy)
#   4. probability maps of the new members (~20-30 min)
#   5. blend-weight search, best weights picked automatically (~15 min)
#   6. extent selector on all members -> submissions/submission_v2.csv (~15 min)
#   7. rubric evaluation of every out-of-fold CSV -> work/eval_v2/ (figures + summary.json)
#   8. 1 px eroded / dilated copies of the recommended file, to submit according to the probes
#
# Options (environment variables):
#   PARALLEL=1           one GPU job at a time (use if a fold runs out of GPU memory)
#   WEIGHTS="1 1 2 2 2"  skip the weight search and use these blend weights (one per member)
#   TTA=4                fewer test-time views when predicting the new members (faster)
#   FORCE=1              redo the selector on the v1 members even if its output exists
# Nothing is submitted to Kaggle: upload the files yourself.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
export FILSEG_WORK=${FILSEG_WORK:-$ROOT/work}
WORK=$FILSEG_WORK
V1_MEMBERS="r34_1024 rgy040_1024 r34_1536"
V1_TAG="r34_1024+rgy040_1024+r34_1536"
V2_ALL="r34_2048 effv2s_1536"
cd "$ROOT"
mkdir -p logs

say() { echo "$(date '+%F %T') | $*"; }
kill_tree() {
  local c
  for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c"; done
  kill "$1" 2>/dev/null || true
}
BG_PIDS=()
cleanup() {
  say "stopping: killing background jobs"
  for p in "${BG_PIDS[@]}"; do kill_tree "$p"; done
  exit 130
}
trap cleanup INT TERM
folds_done() {  # all 5 fold checkpoints of a member exist (and are real files, not Git LFS pointers)
  local f
  for f in 0 1 2 3 4; do
    [ -f "$WORK/models/$1/fold$f.pt" ] && [ "$(stat -c %s "$WORK/models/$1/fold$f.pt")" -gt 1000000 ] || return 1
  done
}

# ---------------------------------------------------------------- 1. checks and resources
say "run_all_v2 started (pid $$; stop everything with: kill $$)"
if command -v nvidia-smi > /dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
elif command -v rocm-smi > /dev/null 2>&1; then
  rocm-smi --showproductname --showmeminfo vram 2> /dev/null | grep -E "Card Series|VRAM Total Memory" || true
fi
GPU_GIB=$("$PY" -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 2**30) if torch.cuda.is_available() else 0)")
[ "$GPU_GIB" -gt 0 ] || { say "no GPU visible to PyTorch ($PY): aborting"; exit 1; }
NCPU=$(nproc)
if [ -z "${PARALLEL:-}" ]; then
  if [ "$GPU_GIB" -ge 160 ]; then PARALLEL=2; else PARALLEL=1; fi
fi
export PARALLEL
CPU_WHILE_TRAINING=$(( NCPU - PARALLEL - 1 )); [ "$CPU_WHILE_TRAINING" -ge 1 ] || CPU_WHILE_TRAINING=1
CPU_ALL=$(( NCPU - 1 )); [ "$CPU_ALL" -ge 1 ] || CPU_ALL=1
say "GPU ${GPU_GIB} GiB, ${NCPU} CPU cores -> PARALLEL=${PARALLEL} GPU jobs, ${CPU_WHILE_TRAINING} CPU workers during training, ${CPU_ALL} after"
FREE_GB=$(df -BG --output=avail "$WORK" | tail -n 1 | tr -dc '0-9')
[ "$FREE_GB" -ge 60 ] || say "WARNING: only ${FREE_GB} GB free under $WORK; v2 writes about 30-40 GB of probability maps"

if [ ! -f "$WORK/cache/train_meta.pkl" ]; then
  say "image cache missing: running prepare_data.py (sizes 1024 1536 2048)"
  "$PY" src/prepare_data.py --sizes 1024 1536 2048
fi
for m in $V1_MEMBERS; do
  folds_done "$m" || { say "v1 model $m is incomplete (missing or Git LFS pointer files): run 'git lfs pull' or scripts/run_pipeline.sh first"; exit 1; }
  if [ ! -f "$WORK/preds/$m/oof.npy" ] || [ ! -f "$WORK/preds/$m/test_folds.npy" ]; then
    say "probability maps of v1 member $m missing: predicting (log logs/predict_v1_$m.out)"
    "$PY" src/predict.py --name "$m" --folds 0 1 2 3 4 --tta 8 > "logs/predict_v1_$m.out" 2>&1
  fi
done
scripts/run_v2.sh targets

# ---------------------------------------------------------------- 2. leaderboard probe files
if [ -f submissions/submission_v2_probe_erode1.csv ] && [ -f submissions/submission_v2_probe_dilate1.csv ]; then
  say "probe files exist"
elif [ -f submissions/submission.csv ]; then
  say "writing probe files from submissions/submission.csv"
  scripts/run_v2.sh probes
else
  say "no submissions/submission.csv (v1): no probe files"
fi

# ---------------------------------------------------------------- 3. GPU training + CPU selector on v1 members
say "training $V2_ALL on the GPU (log logs/train_v2.out)"
MEMBERS_V2="$V2_ALL" scripts/run_v2.sh train > logs/train_v2.out 2>&1 &
TRAIN_PID=$!
BG_PIDS+=("$TRAIN_PID")

if [ -f submissions/submission_v2_v1members.csv ] && [ -f "$WORK/selectors_v2/${V1_TAG}_config.json" ] && [ -z "${FORCE:-}" ]; then
  say "selector on the v1 members already done (FORCE=1 to redo)"
else
  say "selector on the v1 members, on the CPU while the GPU trains (log logs/selector_v1_v2.out)"
  WORKERS=$CPU_WHILE_TRAINING MORPH=0 scripts/run_v2.sh selector-v1 > logs/selector_v1_v2.out 2>&1 &
  SEL_PID=$!
  BG_PIDS+=("$SEL_PID")
  if wait "$SEL_PID"; then
    say "selector on the v1 members done: $(grep -h 'nested CV' "logs/final_v2_${V1_TAG}.txt" | tail -n 2 | tr -s ' ' | tr '\n' ';')"
  else
    say "WARNING: selector on the v1 members failed (logs/selector_v1_v2.out). Training continues, but the final stage uses the same code: send me that log."
  fi
fi

say "waiting for training (progress: tail -f logs/train_v2.out)"
wait "$TRAIN_PID" || true
DONE_V2=""
for m in $V2_ALL; do
  if folds_done "$m"; then DONE_V2="$DONE_V2 $m"; else say "WARNING: $m is missing folds (logs/train_v2.out); leaving it out"; fi
done
DONE_V2=$(echo "$DONE_V2" | xargs)
if [ -z "$DONE_V2" ]; then
  say "no v2 member finished training: the v2 result is submissions/submission_v2_v1members.csv"
  scripts/run_v2.sh evaluate || true
  exit 1
fi
export MEMBERS_V2="$DONE_V2"
read -r -a NEW <<< "$DONE_V2"
NAMES="$V1_MEMBERS $DONE_V2"
TAG=$(echo "$NAMES" | tr ' ' '+')

# ---------------------------------------------------------------- 4. probability maps of the new members
say "probability maps for: $DONE_V2"
scripts/run_v2.sh predict
for m in $DONE_V2; do
  [ -f "$WORK/preds/$m/test_folds.npy" ] || { say "prediction of $m failed (logs/predict_v2_$m.out)"; exit 1; }
done

# ---------------------------------------------------------------- 5. blend weights
if [ -n "${WEIGHTS:-}" ]; then
  say "using WEIGHTS=$WEIGHTS"
else
  GRID=""  # empty = the 5-member default grid of final_ensemble_v2.sh
  [ ${#NEW[@]} -eq 2 ] || GRID="1 1 2 0,1 1 2 1,1 1 2 2,1 1 2 4,0 0 1 1,0 0 1 2,1 1 1 1"
  say "blend-weight search over: $NAMES"
  WEIGHT_GRID="$GRID" scripts/run_v2.sh search
  WEIGHTS=$(awk -F': ' '/^weights /{ n = split($2, a, " ")
              if (n >= 3 && (best == "" || a[n-1] + 0 > best + 0)) { best = a[n-1]; w = substr($1, 9) } }
            END { print w }' "logs/final_v2_${TAG}.txt")
  [ -n "$WEIGHTS" ] || { say "the weight search produced no result (logs/final_v2_${TAG}.txt)"; exit 1; }
  say "best blend weights: $WEIGHTS"
fi

# ---------------------------------------------------------------- 6. selector on all members
say "final: extent selector on $NAMES with weights $WEIGHTS"
WEIGHTS="$WEIGHTS" MORPH=0 WORKERS=$CPU_ALL scripts/run_v2.sh final

# ---------------------------------------------------------------- 7. rubric evaluation
say "rubric evaluation of every out-of-fold CSV -> $WORK/eval_v2/"
scripts/run_v2.sh evaluate || say "WARNING: evaluation failed (see above); the submission files are unaffected"

# ---------------------------------------------------------------- 8. recommendation + probe-dependent copies
say "nested out-of-fold PQ (honest estimates):"
REC=$("$PY" - "$WORK/selectors_v2/${V1_TAG}_config.json" "$WORK/selectors_v2/${TAG}_config.json" <<'EOF'
import json
import sys

best = None
for path, out in [(sys.argv[1], "submissions/submission_v2_v1members.csv"), (sys.argv[2], "submissions/submission_v2.csv")]:
    try:
        with open(path) as f:
            c = json.load(f)
    except OSError:
        continue
    print(f"    {'+'.join(c['names'])}: v2 selector {c['nested_pq']:.4f}  (same models, v1 extent {c['nested_pq_v1_extent']:.4f})  -> {out}",
          file=sys.stderr)
    if best is None or c["nested_pq"] > best[0]:
        best = (c["nested_pq"], out)
print(best[1] if best else "")
EOF
)
[ -n "$REC" ] && [ -f "$REC" ] || { say "no final submission file found"; exit 1; }
for r in -1 1; do
  n=$([ "$r" -lt 0 ] && echo erode1 || echo dilate1)
  "$PY" tools/morph_submission_v2.py "$REC" --radius "$r" --out "${REC%.csv}_$n.csv" 2>&1 | grep -v -i warn
done
trap - INT TERM

cat <<EOF

$(date '+%F %T') | v2 finished. Nothing was submitted. Which file to upload depends on the two probes
(submissions/submission_v2_probe_erode1.csv and _dilate1.csv; the original v1 file scored 0.38):
    neither probe scored above 0.38    -> $REC
    the erode1 probe scored above 0.38 -> ${REC%.csv}_erode1.csv
    the dilate1 probe scored above 0.38 -> ${REC%.csv}_dilate1.csv
Rubric figures and numbers for the report: $WORK/eval_v2/   Stage timings: logs/timing_v2.txt
EOF
tail -n 12 logs/timing_v2.txt || true
