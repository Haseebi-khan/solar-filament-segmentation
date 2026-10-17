#!/usr/bin/env bash
# Train several CV folds of one configuration, PARALLEL jobs at a time on the GPU.
#   scripts/train_folds.sh <name> "<folds>" [extra train.py args...]
#   e.g. scripts/train_folds.sh r34_1024 "0 1 2 3 4" --encoder resnet34 --size 1024 --epochs 60
set -euo pipefail
NAME=$1; FOLDS=$2; shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
PARALLEL=${PARALLEL:-2}
mkdir -p "$ROOT/logs"
for f in $FOLDS; do
  "$PY" "$ROOT/src/train.py" --name "$NAME" --fold "$f" "$@" > "$ROOT/logs/${NAME}_f${f}.out" 2>&1 &
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 10; done
done
wait
echo "all folds done: $NAME"
