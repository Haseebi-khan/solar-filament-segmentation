#!/usr/bin/env bash
# v3 on a credit budget (the instance bills $/hour whether or not the GPU is busy).
#
#   scripts/run_budget_v3.sh plan      cost table for your credit (nothing runs)
#   scripts/run_budget_v3.sh stagec    step 1, ~$2-4: Stage C (pool + linker + scorer) on the EXISTING v2 maps.
#                                      No training. Tells you whether v3's post-processing pays off.
#   scripts/run_budget_v3.sh train     step 2, ~$25-45: M3 (ResNet-34) + M4 (DINOv3 ViT-B) + Stage C on everything,
#                                      with a hard stop when the budget is used up
#   scripts/run_budget_v3.sh stop      stop the run started by 'train' (all processes of its group)
#
# Environment: CREDIT_USD=58.46 (balance when you start), USD_PER_HOUR=1.99, RESERVE_USD=8 (never spent),
#   GPU_ONLY=1 to keep CPU stages out of the run (default here: CPU stages overlap the GPU, because idle GPU
#   time is billed), SKIP_ABLATIONS=1 (saves ~1.5-2 h, uses the blueprint's losses untested).
# The budget counts wall-clock time of THIS run; time the instance sits idle before/after is billed too:
# stop the instance when it is done. Nothing is submitted to Kaggle.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export FILSEG_WORK=${FILSEG_WORK:-$ROOT/work}
STATE=$FILSEG_WORK/v3_state
CREDIT=${CREDIT_USD:-58.46}
RATE=${USD_PER_HOUR:-1.99}
RESERVE=${RESERVE_USD:-8}
BUDGET_H=$(awk -v c="$CREDIT" -v r="$RATE" -v v="$RESERVE" 'BEGIN{printf "%.2f", (c - v) / r}')
cd "$ROOT"
mkdir -p logs "$STATE"
cmd=${1:-plan}

plan() {
  cat <<EOF
Credit \$$CREDIT at \$$RATE/h = $(awk -v c="$CREDIT" -v r="$RATE" 'BEGIN{printf "%.1f", c / r}') h in total; reserve \$$RESERVE -> hard stop after ${BUDGET_H} h.

  step 1  stagec   Stage C on the existing v2 maps (CPU)                          ~0.5-1 h   \$1-2
  step 2  train    benchmarks of the three members (+ fallbacks)                    0.3 h
                   mini loss ablation (abl_full; abl_dice / abl_jml already done)    0.6 h
                   M3 ResNet-34 @1536, 5 folds x 0.28 h                              1.2-1.4 h
                   M4 DINOv3 ViT-B @1536, 5 folds x 1.33 h                           6.7-7.5 h
                   M5 DINOv3 ConvNeXt-S @1536, 5 folds x 0.98 h                      4.9-5.5 h
                   maps of the three members                                         0.5 h
                   last CPU round (blend search, v2 selector, Stage C)               0.7-1.5 h
                   total                                                             15-18 h   \$30-36 (worst case \$45)
(times for M3/M4/M5 are measured 20-step benchmarks of this machine, 60 epochs; the rest are estimates)

Expected honest gain: Stage C alone +0.00-0.02 PQ, plus new members +0.005-0.015 (OOF; v2 is 0.4438).
The biggest lever for the PUBLIC score is not in the budget at all: the OOF -> public gap (~0.06) matches
1 px thinner test masks. Upload submissions/submission_v2_probe_erode1.csv and _dilate1.csv (free) first.
EOF
}

case "$cmd" in
plan)
  plan
  ;;

stagec)
  PY=${PY:-$ROOT/.venv/bin/python}
  "$PY" tests/test_v3.py > logs/budget_tests.out 2>&1 || { echo "unit tests failed: logs/budget_tests.out"; exit 1; }
  echo "$(date '+%F %T') | Stage C on the v2 maps (log logs/final_v3_stagec_*.txt)"
  scripts/run_v3.sh stagec
  scripts/run_v3.sh diag || true
  echo "$(date '+%F %T') | done. Read the last lines of logs/final_v3_stagec_r34_2048+effv2s_1536_w1-1_linker.txt (bootstrap vs v2:"
  echo "  ADOPT = worth keeping) and logs/final_v3_diag.txt (where the PQ is lost)."
  ;;

train)
  if [ -f "$STATE/budget_pgid.txt" ] && kill -0 "$(cut -d' ' -f1 "$STATE/budget_pgid.txt")" 2>/dev/null; then
    echo "a budget run is already active (pid $(cut -d' ' -f1 "$STATE/budget_pgid.txt")); '$0 stop' ends it"; exit 1
  fi
  plan
  echo
  export GPU_ONLY=${GPU_ONLY:-0}
  export SKIP_GONG=1 SKIP_STUDENTS=1
  export MEMBERS="m3_r34_1536_v3 m4_vitb_1536_v3 m5_cnxs_1536_v3"
  export EXTRA_ALL=${EXTRA_ALL:---epochs 60}
  export MAX_HOURS_M3=${MAX_HOURS_M3:-0.6} MAX_HOURS_M4=${MAX_HOURS_M4:-2.0} MAX_HOURS_M5=${MAX_HOURS_M5:-1.5}
  export BG_WORKERS=${BG_WORKERS:-12}
  if [ -z "${SKIP_ABLATIONS:-}" ]; then
    export VARIANTS="abl_dice abl_jml abl_full"
    # keep the epoch count of any ablation folds that already exist (variants must be compared at equal epochs)
    if [ ! -f "$FILSEG_WORK/models/abl_dice/fold0.pt" ]; then export ABL_EPOCHS=40; fi
  fi
  : > logs/run_all_v3.out
  setsid nohup "$ROOT/scripts/run_all_v3.sh" >> logs/run_all_v3.out 2>&1 < /dev/null &
  RUN_PID=$!
  sleep 3
  kill -0 "$RUN_PID" 2>/dev/null || { echo "the run did not start; see logs/run_all_v3.out"; tail -n 5 logs/run_all_v3.out; exit 1; }
  PGID=$(ps -o pgid= -p "$RUN_PID" | tr -d ' ')
  echo "$RUN_PID $PGID $(date +%s)" > "$STATE/budget_pgid.txt"
  END=$(( $(date +%s) + $(awk -v h="$BUDGET_H" 'BEGIN{printf "%d", h * 3600}') ))
  setsid nohup bash -c '
    while kill -0 '"$RUN_PID"' 2>/dev/null; do
      if [ "$(date +%s)" -ge '"$END"' ]; then
        echo "$(date "+%F %T") | BUDGET REACHED ('"$BUDGET_H"' h): stopping the run" >> '"$ROOT"'/logs/run_all_v3.out
        kill -TERM -- -'"$PGID"' 2>/dev/null; sleep 30; kill -KILL -- -'"$PGID"' 2>/dev/null
        break
      fi
      sleep 30
    done' > /dev/null 2>&1 < /dev/null &
  echo "started: pid $RUN_PID, hard stop in ${BUDGET_H} h ($(date -d "@$END" '+%F %T'))."
  echo "  watch:  tail -f logs/run_all_v3.out        stop now:  scripts/run_budget_v3.sh stop"
  echo "  When it prints 'run_all_v3 finished', or after a stop, shut the instance down to stop billing."
  ;;

stop)
  [ -f "$STATE/budget_pgid.txt" ] || { echo "no budget run recorded"; exit 1; }
  read -r pid pgid _ < "$STATE/budget_pgid.txt"
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    sleep 20
    kill -KILL -- "-$pgid" 2>/dev/null || true
    echo "stopped process group $pgid"
  else
    echo "the run is not active"
  fi
  ;;

*)
  sed -n '2,17p' "$0"
  exit 1
  ;;
esac
