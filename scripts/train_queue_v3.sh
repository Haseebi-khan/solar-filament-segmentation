#!/usr/bin/env bash
# v3: train members from configs/<member>.yaml (or configs/students/<member>.yaml), fold by fold.
#   scripts/train_queue_v3.sh m3_r34_1536_v3 m1_cnxl_2048_v3            -> folds 0-4 of each, one job at a time
#   FOLDS="0" PARALLEL=2 scripts/train_queue_v3.sh m3_r34_1536_v3        -> two folds at once (small members only)
#   EXTRA="--epochs 60" scripts/train_queue_v3.sh m2_vitl_1536_v3        -> extra train_v3.py arguments
# Finished folds (work/models/<name>/fold<k>.pt) are skipped; an interrupted fold resumes from fold<k>_last.pt.
# M1/M2 need most of the GPU: keep PARALLEL=1 for them.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
WORK=${FILSEG_WORK:-$ROOT/work}
FOLDS=${FOLDS:-"0 1 2 3 4"}
PARALLEL=${PARALLEL:-1}
EXTRA=${EXTRA:-}
cd "$ROOT"
mkdir -p logs
[ $# -ge 1 ] || { echo "usage: $0 <member> [<member> ...]   (configs/<member>.yaml)"; exit 1; }

config_of() {
  if [ -f "configs/$1.yaml" ]; then echo "configs/$1.yaml"
  elif [ -f "configs/students/$1.yaml" ]; then echo "configs/students/$1.yaml"
  else echo ""; fi
}

run_fold() {
  local name=$1 cfg=$2 f=$3
  echo "$(date '+%F %T') start $name fold $f"
  # shellcheck disable=SC2086
  if "$PY" src/train_v3.py --config "$cfg" --fold "$f" --resume $EXTRA > "logs/${name}_f${f}.out" 2>&1; then
    echo "$(date '+%F %T') done  $name fold $f: $(grep '^epoch' "$WORK/models/$name/fold$f.log" | tail -n 1 | grep -o 'pq@.*' || true)"
  else
    echo "$(date '+%F %T') FAILED $name fold $f (see logs/${name}_f${f}.out)"
  fi
}

for name in "$@"; do
  cfg=$(config_of "$name")
  [ -n "$cfg" ] || { echo "no config for $name"; exit 1; }
  for f in $FOLDS; do
    if [ -f "$WORK/models/$name/fold$f.pt" ]; then echo "skip $name fold $f (done)"; continue; fi
    run_fold "$name" "$cfg" "$f" &
    sleep 30
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 20; done
  done
done
wait
echo "$(date '+%F %T') queue finished; missing folds (rerun to retry):"
for name in "$@"; do
  for f in $FOLDS; do [ -f "$WORK/models/$name/fold$f.pt" ] || echo "  $name fold $f"; done
done
