#!/usr/bin/env bash
# v3 Phase 1: loss / head ablations on the fast proxy (ResNet-34 U-Net at 1024 px, 60 epochs, 5 folds each),
# in the order of blueprint section 13. For each variant: train 5 folds (PARALLEL at a time), out-of-fold maps
# (TTA 2, no test), v1 post-processing with the best of 4 thresholds (score_member_v3.py), and a paired
# bootstrap against its baseline (compare_v3.py; gate: mean delta >= +0.003 and P(delta > 0) >= 0.90).
#
#   nohup scripts/ablate_v3.sh > logs/ablate_v3.out 2>&1 &
#   VARIANTS="abl_full_1536 abl_full_hr" scripts/ablate_v3.sh        # HR stem test (needs 1536 targets)
#
# Cheaper set (~1.5 h with PARALLEL=4): VARIANTS="abl_dice abl_jml abl_full" ABL_EPOCHS=40 (src/choose_losses_v3.py copes).
# Results are appended to logs/final_v3_ablations.txt. Rerunning skips finished folds, maps and CSVs.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
WORK=${FILSEG_WORK:-$ROOT/work}
PARALLEL=${PARALLEL:-2}
cd "$ROOT"
mkdir -p logs "$WORK/ablations_v3"
OUT=logs/final_v3_ablations.txt

declare -A ARGS=(
  [abl_dice]="--region dice --w-blob 0 --no-aux-heads --no-deep-sup"          # v1 losses: BCE + soft Dice
  [abl_jml]="--region jml --w-blob 0 --no-aux-heads --no-deep-sup"            # (1) JML1 instead of soft Dice
  [abl_dml]="--region dml --w-blob 0 --no-aux-heads --no-deep-sup"            # (1) DML1 instead of soft Dice
  [abl_jml_blob]="--region jml --w-blob 0.5 --no-aux-heads --no-deep-sup"     # (2) + blob loss
  [abl_full]="--region jml --w-blob 0.5"                                     # (3) + S/D heads + deep supervision
  [abl_full_sr]="--region jml --w-blob 0.5 --w-sr 0.5"                       # (4) + skeleton recall
  [abl_full_1536]="--region jml --w-blob 0.5 --size 1536"                    # baseline for (6), (7)
  [abl_full_hr]="--region jml --w-blob 0.5 --size 1536 --hr-stem"            # (6) HR stem, output 2048
  [abl_full_edges]="--region jml --w-blob 0.5 --size 1536 --hr-stem --edge-inputs"   # (7) Sobel/LoG inputs
)
declare -A BASE=(
  [abl_jml]=abl_dice [abl_dml]=abl_dice [abl_jml_blob]=abl_jml [abl_full]=abl_jml_blob [abl_full_sr]=abl_full
  [abl_full_1536]=abl_full [abl_full_hr]=abl_full_1536 [abl_full_edges]=abl_full_hr
)
read -r -a VARIANTS <<< "${VARIANTS:-abl_dice abl_jml abl_dml abl_jml_blob abl_full abl_full_sr}"

train_variant() {
  local v=$1 f
  for f in 0 1 2 3 4; do
    if [ -f "$WORK/models/$v/fold$f.pt" ]; then continue; fi
    # shellcheck disable=SC2086
    "$PY" src/train_v3.py --config configs/proxy_r34_1024_v3.yaml --name "$v" --fold "$f" --resume ${ARGS[$v]} ${ABL_EPOCHS:+--epochs $ABL_EPOCHS} \
      > "logs/${v}_f${f}.out" 2>&1 || echo "$(date '+%F %T') FAILED $v fold $f (logs/${v}_f${f}.out)" &
    sleep 20
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 20; done
  done
  wait
}

for v in "${VARIANTS[@]}"; do
  [ -n "${ARGS[$v]:-}" ] || { echo "unknown variant $v"; exit 1; }
  echo "$(date '+%F %T') == $v: ${ARGS[$v]}"
  train_variant "$v"
  for f in 0 1 2 3 4; do [ -f "$WORK/models/$v/fold$f.pt" ] || { echo "$v fold $f missing: stopping"; exit 1; }; done
  CSV=$WORK/ablations_v3/$v.csv
  if [ ! -f "$CSV" ]; then
    "$PY" src/predict_v3.py --name "$v" --tta 2 --no-test --no-aux > "logs/predict_${v}.out" 2>&1
    "$PY" src/score_member_v3.py --names "$v" --out "$CSV" 2>&1 | grep -v -i warn | tee -a "$OUT"
  fi
  b=${BASE[$v]:-}
  if [ -n "$b" ] && [ -f "$WORK/ablations_v3/$b.csv" ]; then
    echo "-- $v vs $b" | tee -a "$OUT"
    "$PY" src/compare_v3.py "$WORK/ablations_v3/$b.csv" "$CSV" 2>&1 | grep -v -i warn | tee -a "$OUT"
  fi
done
echo "$(date '+%F %T') ablations finished; summary in $OUT"
