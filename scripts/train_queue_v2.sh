#!/usr/bin/env bash
# v2: train new ensemble members one fold at a time (one job saturates the MI300X).
#   scripts/train_queue_v2.sh            -> all v2 members, folds 0-4
#   MEMBERS_V2="r34_2048" FOLDS="0" scripts/train_queue_v2.sh
# Each fold writes work/models/<name>/fold<k>.pt and .log; finished folds are skipped.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
FOLDS=${FOLDS:-"0 1 2 3 4"}
declare -A ARGS=(
  [r34_2048]="--encoder resnet34 --size 2048 --epochs 80"
  [effv2s_1536]="--encoder tu-tf_efficientnetv2_s --size 1536 --epochs 80"
)
read -r -a MEMBERS <<< "${MEMBERS_V2:-r34_2048 effv2s_1536}"
mkdir -p "$ROOT/logs"
for name in "${MEMBERS[@]}"; do
  for f in $FOLDS; do
    if [ -f "$ROOT/work/models/$name/fold$f.pt" ]; then echo "skip $name fold $f"; continue; fi
    echo "$(date '+%F %T') start $name fold $f"
    # shellcheck disable=SC2086
    "$PY" "$ROOT/src/train.py" --name "$name" --fold "$f" ${ARGS[$name]} --eval-every 20 \
      > "$ROOT/logs/${name}_f${f}.out" 2>&1
    echo "$(date '+%F %T') done  $name fold $f: $(tail -n 2 "$ROOT/work/models/$name/fold$f.log" | head -n 1)"
  done
done
