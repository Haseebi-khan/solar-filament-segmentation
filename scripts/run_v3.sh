#!/usr/bin/env bash
# v3 entry point, one stage per call (each resumable; GPU stages one at a time). See V3.md for the order.
#
#   scripts/run_v3.sh tests                     CPU unit tests (~1 min)
#   scripts/run_v3.sh diag                      P0: taxonomy, ceilings, bootstrap of v1 vs v2 (CPU, ~15 min)
#   scripts/run_v3.sh stagec                    Stage C (pool + linker + scorer) on the v2 members' maps (CPU)
#   scripts/run_v3.sh targets                   T / D / instance targets (CPU, ~10 min)
#   scripts/run_v3.sh bench [members...]        50-step benchmark of each member (GPU, minutes)
#   scripts/run_v3.sh ablate                    P1 loss ablations on the proxy (GPU, ~5-6 h)
#   scripts/run_v3.sh train <members...>        P2 5-fold training (GPU)
#   scripts/run_v3.sh predict <members...>      OOF + test maps (GPU)
#   scripts/run_v3.sh rescorer                  v1 post-processing + v2 rescorer on a blend (CPU) -> submission
#   scripts/run_v3.sh gong                      Stage B data: download, domain check, QC (network + CPU)
#   scripts/run_v3.sh pseudo                    Stage B pseudo-labels from the M1 + M2 teachers (GPU)
#   scripts/run_v3.sh students                  Stage B students M1s, M2s (GPU)
#
# Blends are given by NAMES="a b c" WEIGHTS="1 1 1" (defaults below). LINKER=0 skips the learned linker in
# stagec. Nothing is ever submitted to Kaggle.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
export FILSEG_WORK=${FILSEG_WORK:-$ROOT/work}
WORK=$FILSEG_WORK
export FILSEG_UNLABELED=${FILSEG_UNLABELED:-$WORK/unlabeled}
WORKERS=${WORKERS:-$(( $(nproc) - 1 ))}
V1_CSV=$WORK/oof_sub_r34_1024+rgy040_1024+r34_1536.csv
V2_CSV=$WORK/oof_sub_v2_r34_1024+rgy040_1024+r34_1536+r34_2048+effv2s_1536.csv
V2_NESTED_CSV=$WORK/oof_sub_v2_r34_1024+rgy040_1024+r34_1536.csv
cd "$ROOT"
mkdir -p logs
stage=${1:-}
shift || true
say() { echo "$(date '+%F %T') | $*"; }
timing() { echo "$(date '+%F %T') stage $1: $2 s" >> logs/timing_v3.txt; }
quiet() { grep -v -i -E "warn|^\s*$" || true; }
tag_of() {  # members with weight > 0, joined by +
  local -a n w
  read -r -a n <<< "$1"
  read -r -a w <<< "$2"
  local out="" i
  for i in "${!n[@]}"; do
    if awk "BEGIN{exit !(${w[$i]} > 0)}"; then out="${out:+$out+}${n[$i]}"; fi
  done
  echo "$out"
}
T0=$(date +%s)

case "$stage" in
tests)
  "$PY" tests/test_v3.py
  ;;

diag)
  # v1's OOF CSV in this repo was scored with an in-sample rescorer (optimistic); v2's are nested (honest).
  say "error taxonomy of the v2 OOF predictions (and v1 for comparison)"
  "$PY" src/diagnose_v3.py "$V2_CSV" "$V2_NESTED_CSV" --labels v2_5members v2_v1members --workers "$WORKERS" 2>&1 \
    | quiet | tee logs/final_v3_diag.txt
  say "noise ceiling and annotator-aware oracle"
  "$PY" src/ceilings_v3.py --oof-csv "$V2_CSV" --workers "$WORKERS" 2>&1 | quiet | tee -a logs/final_v3_diag.txt
  say "paired bootstrap: v2 with the v1 members -> v2 with 5 members"
  "$PY" src/compare_v3.py "$V2_NESTED_CSV" "$V2_CSV" 2>&1 | quiet | tee -a logs/final_v3_diag.txt
  timing diag $(( $(date +%s) - T0 ))
  ;;

stagec)
  NAMES=${NAMES:-"r34_1024 rgy040_1024 r34_1536 r34_2048 effv2s_1536"}
  WEIGHTS=${WEIGHTS:-"0 0 0 1 1"}
  BASE_CSV=${BASE_CSV:-$V2_CSV}
  read -r -a N <<< "$NAMES"
  read -r -a W <<< "$WEIGHTS"
  TAG=$(tag_of "$NAMES" "$WEIGHTS")
  LOG=logs/final_v3_stagec_${TAG}.txt
  LK=()
  POOL=$WORK/pools_v3/$TAG
  if [ "${LINKER:-1}" = "1" ]; then
    if [ ! -f "$WORK/linker_v3/$TAG/linker.pkl" ]; then
      say "training the fragment linker on $TAG (log $LOG)"
      "$PY" src/train_linker_v3.py --names "${N[@]}" --weights "${W[@]}" --workers "$WORKERS" 2>&1 | quiet | tee -a "$LOG"
    fi
    LK=(--linker "$WORK/linker_v3/$TAG/linker.pkl" --link-taus ${LINK_TAUS:-0.4 0.6})
    POOL=${POOL}_linker
  fi
  for split in oof test; do
    if [ ! -f "$POOL/$split/features.parquet" ] || [ -n "${FORCE:-}" ]; then
      say "candidate pool ($split)"
      "$PY" src/build_pool_v3.py --names "${N[@]}" --weights "${W[@]}" --split "$split" "${LK[@]}" \
        --workers "$WORKERS" --out "$POOL" 2>&1 | quiet | tee -a "$LOG"
    fi
  done
  say "stacked scorer + overlap-aware selection (nested CV)"
  "$PY" src/train_scorer_v3.py --pool "$POOL" --workers "$WORKERS" 2>&1 | quiet | tee -a "$LOG"
  CONF=$POOL/scorer/scorer_config.json
  OOF=$WORK/oof_sub_v3_${TAG}$( [ ${#LK[@]} -gt 0 ] && echo _linker ).csv
  SUB=submissions/submission_v3_${TAG}$( [ ${#LK[@]} -gt 0 ] && echo _linker ).csv
  "$PY" src/make_submission_v3.py --config "$CONF" --oof --out "$OOF" --workers "$WORKERS" 2>&1 | quiet | sed 's/^/[OOF] /' | tee -a "$LOG"
  "$PY" src/make_submission_v3.py --config "$CONF" --out "$SUB" --workers "$WORKERS" 2>&1 | quiet | tee -a "$LOG"
  "$PY" src/evaluate_v2.py "$OOF" --labels "v3_${TAG}" --out "$WORK/eval_v3" 2>&1 | quiet | sed 's/^/[rubric] /' | tee -a "$LOG"
  "$PY" src/diagnose_v3.py "$OOF" --labels "v3_${TAG}" --workers "$WORKERS" 2>&1 | quiet | sed 's/^/[diag] /' | tee -a "$LOG"
  if [ -f "$BASE_CSV" ]; then
    echo "-- paired bootstrap vs $(basename "$BASE_CSV")" | tee -a "$LOG"
    "$PY" src/compare_v3.py "$BASE_CSV" "$OOF" 2>&1 | quiet | tee -a "$LOG"
  fi
  say "done: $SUB (honest OOF CSV $OOF; log $LOG)"
  timing "stagec $TAG" $(( $(date +%s) - T0 ))
  ;;

targets)
  "$PY" src/prepare_targets_v3.py --sizes 1024 1536 2048 --inst-size 1024 2>&1 | quiet
  timing targets $(( $(date +%s) - T0 ))
  ;;

bench)
  MEMBERS=("$@")
  [ ${#MEMBERS[@]} -gt 0 ] || MEMBERS=(m3_r34_1536_v3 m1_cnxl_2048_v3 m2_vitl_1536_v3)
  for m in "${MEMBERS[@]}"; do
    cfg=configs/$m.yaml
    [ -f "$cfg" ] || cfg=configs/students/$m.yaml
    say "benchmark $m (first steps include MIOpen kernel compilation)"
    "$PY" src/train_v3.py --config "$cfg" --fold 0 --bench-steps "${STEPS:-50}" ${EXTRA:-} 2>&1 | grep -E "bench|model |train images|Error|error" \
      | tee -a logs/final_v3_bench.txt || say "benchmark of $m failed: rerun without the grep to see the error"
  done
  ;;

ablate)
  scripts/ablate_v3.sh
  ;;

train)
  [ $# -ge 1 ] || set -- m3_r34_1536_v3 m1_cnxl_2048_v3 m2_vitl_1536_v3
  scripts/train_queue_v3.sh "$@"
  timing "train $*" $(( $(date +%s) - T0 ))
  ;;

predict)
  [ $# -ge 1 ] || set -- m3_r34_1536_v3 m1_cnxl_2048_v3 m2_vitl_1536_v3
  for m in "$@"; do
    say "maps of $m (log logs/predict_v3_$m.out)"
    "$PY" src/predict_v3.py --name "$m" --tta "${TTA:-8}" > "logs/predict_v3_$m.out" 2>&1
  done
  timing "predict $*" $(( $(date +%s) - T0 ))
  ;;

rescorer)
  # Phase-2 submission: v1 post-processing + v2 extent selector / rescorer on the new blend (existing scripts)
  NAMES=${NAMES:-"r34_2048 effv2s_1536 m3_r34_1536_v3 m1_cnxl_2048_v3 m2_vitl_1536_v3"}
  if [ -z "${WEIGHTS:-}" ]; then
    NAMES="$NAMES" scripts/final_ensemble_v2.sh
    say "pick the best weights from logs/final_v2_$(echo "$NAMES" | tr ' ' '+').txt and rerun with WEIGHTS=\"...\""
  else
    NAMES="$NAMES" WEIGHTS="$WEIGHTS" OUT=submissions/submission_v3_rescorer.csv WORKERS=$WORKERS scripts/final_ensemble_v2.sh
  fi
  ;;

gong)
  say "GONG archive frames of 30 training images + domain check"
  "$PY" src/gong_download_v3.py train-stamps --n 30
  "$PY" src/gong_domain_check_v3.py
  say "unlabelled pool 2023-01-01 .. 2026-09-30 (log logs/gong_download_v3.out)"
  "$PY" src/gong_download_v3.py pool --start "${START:-2023-01-01}" --end "${END:-2026-09-30}" > logs/gong_download_v3.out 2>&1
  say "quality control"
  "$PY" src/gong_qc_v3.py --max-frames "${MAX_FRAMES:-8000}" 2>&1 | quiet | tee logs/final_v3_gong_qc.txt
  timing gong $(( $(date +%s) - T0 ))
  ;;

pseudo)
  TEACHERS=${TEACHERS:-"m1_cnxl_2048_v3 m2_vitl_1536_v3"}
  # shellcheck disable=SC2086
  "$PY" src/pseudo_label_v3.py --teachers $TEACHERS --folds ${FOLDS:-0 1 2 3 4} 2>&1 | quiet | tee logs/pseudo_v3.out
  timing pseudo $(( $(date +%s) - T0 ))
  ;;

students)
  [ $# -ge 1 ] || set -- m1s_cnxl_2048_v3 m2s_vitl_1536_v3
  scripts/train_queue_v3.sh "$@"
  timing "students $*" $(( $(date +%s) - T0 ))
  ;;

*)
  sed -n '2,20p' "$0"
  exit 1
  ;;
esac
