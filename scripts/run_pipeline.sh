#!/usr/bin/env bash
# End-to-end reproduction: data prep -> 5-fold training of every ensemble member ->
# out-of-fold/test prediction -> blend + post-processing + rescorer -> submission CSV.
#
#   FILSEG_DATA=/path/to/MAGFiLO_1.0_Kaggle_2026 FILSEG_WORK=$PWD/work scripts/run_pipeline.sh
#
# Wall time on one AMD MI300X: ~5 h of training (all members), ~10 min of inference and tuning.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
export FILSEG_WORK=${FILSEG_WORK:-$ROOT/work}

# name | train.py arguments  (every member uses the same 5 image-level folds)
MEMBERS=(
  "r34_1024|--encoder resnet34 --size 1024 --epochs 60"
  "rgy040_1024|--encoder tu-regnety_040 --size 1024 --epochs 80"
  "r34_1536|--encoder resnet34 --size 1536 --epochs 80"
)
WEIGHTS=${WEIGHTS:-"1 1 2"}

"$PY" "$ROOT/src/prepare_data.py" --sizes 1024 1536

NAMES=()
for m in "${MEMBERS[@]}"; do
  name=${m%%|*}; args=${m#*|}
  NAMES+=("$name")
  # shellcheck disable=SC2086
  PARALLEL=${PARALLEL:-2} "$ROOT/scripts/train_folds.sh" "$name" "0 1 2 3 4" $args --eval-every 20
  "$PY" "$ROOT/src/predict.py" --name "$name" --folds 0 1 2 3 4 --tta 8
done

# blend-weight search (logs only), then rescorer + out-of-fold score + submissions/submission.csv
NAMES="${NAMES[*]}" "$ROOT/scripts/final_ensemble.sh"
NAMES="${NAMES[*]}" WEIGHTS="$WEIGHTS" "$ROOT/scripts/final_ensemble.sh"
