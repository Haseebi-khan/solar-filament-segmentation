#!/usr/bin/env bash
# v2: train the new ensemble members, PARALLEL folds at a time on the GPU.
#   scripts/train_queue_v2.sh                                  -> all v2 members, folds 0-4, 2 folds at once
#   MEMBERS_V2="r34_2048" FOLDS="0" PARALLEL=1 scripts/train_queue_v2.sh
# One job already keeps the MI300X compute at 100% but uses only ~25% of its 192 GiB, so a second
# concurrent fold adds ~18% total throughput (measured on v1). If a fold runs out of memory, rerun
# with PARALLEL=1: finished folds (work/models/<name>/fold<k>.pt) are skipped.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
FOLDS=${FOLDS:-"0 1 2 3 4"}
PARALLEL=${PARALLEL:-2}
declare -A ARGS=(
  [r34_2048]="--encoder resnet34 --size 2048 --epochs 80"
  [effv2s_1536]="--encoder tu-tf_efficientnetv2_s --size 1536 --epochs 80"
)
read -r -a MEMBERS <<< "${MEMBERS_V2:-r34_2048 effv2s_1536}"
mkdir -p "$ROOT/logs"

run_fold() {
  local name=$1 f=$2
  echo "$(date '+%F %T') start $name fold $f"
  # shellcheck disable=SC2086
  if "$PY" "$ROOT/src/train.py" --name "$name" --fold "$f" ${ARGS[$name]} --eval-every 20 \
       > "$ROOT/logs/${name}_f${f}.out" 2>&1; then
    echo "$(date '+%F %T') done  $name fold $f: $(tail -n 2 "$ROOT/work/models/$name/fold$f.log" | head -n 1)"
  else
    echo "$(date '+%F %T') FAILED $name fold $f (see logs/${name}_f${f}.out)"
  fi
}

for name in "${MEMBERS[@]}"; do
  [ -n "${ARGS[$name]:-}" ] || { echo "unknown member $name"; exit 1; }
  for f in $FOLDS; do
    if [ -f "$ROOT/work/models/$name/fold$f.pt" ]; then echo "skip $name fold $f"; continue; fi
    run_fold "$name" "$f" &
    sleep 60  # stagger starts so the two jobs do not load data / compile kernels at the same moment
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 20; done
  done
done
wait
echo "$(date '+%F %T') queue finished; missing folds (rerun to retry):"
for name in "${MEMBERS[@]}"; do
  for f in $FOLDS; do [ -f "$ROOT/work/models/$name/fold$f.pt" ] || echo "  $name fold $f"; done
done
