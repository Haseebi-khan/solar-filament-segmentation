#!/usr/bin/env bash
# v3: the whole pipeline in one command. Resumable: after an interruption run it again and finished work
# (targets, ablations, benchmarks, trained folds, maps, pools, pseudo-labels) is skipped.
#
#   nohup scripts/run_all_v3.sh > logs/run_all_v3.out 2>&1 &
#   tail -f logs/run_all_v3.out           # stop everything: kill <pid printed on the first line>
#
# GPU jobs run strictly one at a time; CPU and network jobs run alongside. On one MI300X:
#   0. checks, DINOv3 weight download, unit tests (1 min), v3 targets (10 min)
#   1. background CPU: Phase-0 diagnostics + ceilings, Stage C on the v2 maps          (~2-3 h)
#      background network: GONG download + domain check + QC                           (~2-4 h)
#   2. GPU: loss ablations on the proxy -> loss flags (src/choose_losses_v3.py)         (~5-6 h)
#   3. GPU: 50-step benchmark of every member; out of memory or over the per-fold budget -> next fallback
#   4. GPU: 5-fold training + maps of M3, M1, M2                                        (~35-45 h)
#      background CPU after M3 and after all teachers: blend search, v2-selector and v3 Stage C submissions
#   5. GPU: Stage B pseudo-labels, students M1s / M2s, their maps                       (~45-55 h)
#      CPU: blend search with the students, final Stage C
#   6. summary: honest nested out-of-fold PQ of every candidate and the file to upload
# About 4-5 days in total; STOP_AFTER lets you stop at a phase boundary.
#
# GPU_ONLY=1 (default): only GPU work runs. No diagnostics, Stage C, blend search or GONG download/QC in the
# background (those are CPU: LightGBM, connected components, JPEG decoding); GPU jobs run several at a time to
# fill the 192 GB (see PARALLEL below). Run the CPU parts later with GPU_ONLY=0 (finished GPU work is skipped).
# Unavoidable CPU use: target building (10 min, once), per-fold validation post-processing in training.
#
# Options (environment variables):
#   GPU_ONLY=0          also run the background CPU/network jobs (diagnostics, Stage C rounds, GONG)
#   ABL_PARALLEL=4      proxy folds trained at once during the ablations (small models, ~25 GB each)
#   PARALLEL=n          force n folds at once for M1/M2/M3 (default: from the benchmark's peak memory, max 3)
#   STOP_AFTER=prep|ablate|teachers  stop after that phase (rerun without it to continue)
#   SKIP_ABLATIONS=1    blueprint default losses (JML1 + blob + S/D heads + deep supervision), no proxy runs
#   SKIP_GONG=1         no download and no Stage B;   SKIP_STUDENTS=1   download, but no pseudo-labels/students
#   MEMBERS="m3_r34_1536_v3 m1_cnxl_2048_v3 m2_vitl_1536_v3"   v3 members to train, in this order
#   MAX_HOURS_M1=5 MAX_HOURS_M2=5 MAX_HOURS_M3=2   per-fold time budget for the benchmark's fallback choice
#   PL_VIEWS="0 4 1 3"  TTA views for pseudo-labels ("0 4" halves the time);  MIN_U=2000  minimum GONG frames
#   FILSEG_UNLABELED=/scratch/unlabeled   where GONG frames and pseudo-labels go (default work/unlabeled)
#   FORCE=1             redo the CPU rounds (blend search, v2 selector, Stage C) even if they finished before
# Nothing is submitted to Kaggle.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
export FILSEG_WORK=${FILSEG_WORK:-$ROOT/work}
WORK=$FILSEG_WORK
export FILSEG_UNLABELED=${FILSEG_UNLABELED:-$WORK/unlabeled}
STATE=$WORK/v3_state
cd "$ROOT"
mkdir -p logs "$STATE"
read -r -a MEMBERS <<< "${MEMBERS:-m3_r34_1536_v3 m1_cnxl_2048_v3 m2_vitl_1536_v3}"
V2_NAMES="r34_2048 effv2s_1536"
PL_VIEWS=${PL_VIEWS:-"0 4 1 3"}
MIN_U=${MIN_U:-2000}
NCPU=$(nproc)
BG_WORKERS=$(( NCPU > 6 ? NCPU - 4 : 1 ))
GPU_ONLY=${GPU_ONLY:-1}
T_ALL=$(date +%s)

say() { echo "$(date '+%F %T') | $*"; }
timing() { echo "$(date '+%F %T') run_all_v3 $1: $2 s" >> logs/timing_v3.txt; }
kill_tree() {
  local c
  for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c"; done
  kill "$1" 2>/dev/null || true
}
P0_PID="" GONG_PID="" CPU_PID=""
cleanup() {
  say "stopping: killing background jobs"
  local p
  for p in $P0_PID $GONG_PID $CPU_PID; do kill_tree "$p"; done
  exit 130
}
trap cleanup INT TERM
fail() { say "FAILED: $*"; say "background jobs still running: ${P0_PID:-} ${GONG_PID:-} ${CPU_PID:-} (they finish on their own)"; exit 1; }
wait_job() {  # wait_job <pid or empty> <label>
  [ -n "$1" ] || return 0
  if wait "$1"; then say "$2 finished"; else say "WARNING: $2 failed (logs/run_all_v3_$2.out)"; fi
}
wait_all() {
  wait_job "$P0_PID" p0; P0_PID=""
  wait_job "$GONG_PID" gong; GONG_PID=""
  wait_job "$CPU_PID" round; CPU_PID=""
}
stop_if() {
  if [ "${STOP_AFTER:-}" = "$1" ]; then
    say "STOP_AFTER=$1: waiting for the background jobs, then stopping (rerun without STOP_AFTER to continue)"
    wait_all
    exit 0
  fi
}
folds_done() {  # all 5 fold checkpoints exist and are real files (not Git LFS pointers)
  local f
  for f in 0 1 2 3 4; do
    [ -f "$WORK/models/$1/fold$f.pt" ] && [ "$(stat -c %s "$WORK/models/$1/fold$f.pt")" -gt 1000000 ] || return 1
  done
}
preds_done() { [ -f "$WORK/preds/$1/oof.npy" ] && [ -f "$WORK/preds/$1/test_folds.npy" ]; }
config_of() { if [ -f "configs/$1.yaml" ]; then echo "configs/$1.yaml"; else echo "configs/students/$1.yaml"; fi; }
budget_of() {
  case $1 in
    m1*) echo "${MAX_HOURS_M1:-5}" ;;
    m2*) echo "${MAX_HOURS_M2:-5}" ;;
    *) echo "${MAX_HOURS_M3:-2}" ;;
  esac
}
fallbacks_of() {  # first line = config as is; then the blueprint's fallbacks (memory first, then speed)
  case $1 in
    m1*) printf '%s\n' "" "--ckpt-stages 0 1 2" "--bs 2 --accum 4 --ckpt-stages 0 1 2" \
                       "--backbone convnext_base.dinov3_lvd1689m" "--backbone convnext_base.dinov3_lvd1689m --size 1536" ;;
    m2*) printf '%s\n' "" "--bs 4 --accum 2" "--size 1280" "--size 1280 --bs 4 --accum 2" ;;
    *) printf '%s\n' "" "--bs 4 --accum 2" ;;
  esac
}
parallel_of() {  # folds at once: as many as fit in ~88 % of GPU memory (max 3; compute is shared, so gains are modest)
  if [ -n "${PARALLEL:-}" ]; then echo "$PARALLEL"; return; fi
  local pk
  pk=$(cat "$STATE/peak_$1.txt" 2>/dev/null || echo 0)
  awk -v pk="$pk" -v gib="$GPU_GIB" 'BEGIN{ if (pk <= 0) { print 1; exit } n = int(gib * 0.88 / (pk + 4)); print (n < 1 ? 1 : (n > 3 ? 3 : n)) }'
}
student_of() {
  case $1 in
    m1_*) echo "m1s_${1#m1_}" ;;
    m2_*) echo "m2s_${1#m2_}" ;;
    *) echo "" ;;
  esac
}

choose_setting() {  # benchmark <member> with its fallbacks; writes $STATE/extra_<member>.txt (flags or FAILED)
  local m=$1 f=$STATE/extra_$1.txt cfg budget fb out h pk best_h="" best_fb="" best_pk="" ran=0
  if [ -f "$f" ]; then return 0; fi
  cfg=$(config_of "$m")
  budget=$(budget_of "$m")
  while IFS= read -r fb; do
    say "benchmark $m: ${fb:-config as is} (first steps include MIOpen kernel compilation)"
    # shellcheck disable=SC2086
    if out=$("$PY" src/train_v3.py --config "$cfg" --fold 0 --bench-steps 50 $LOSS $fb < /dev/null 2>&1); then
      h=$(echo "$out" | sed -n 's/.*~\([0-9.]*\) h for .*/\1/p' | tail -n 1)
      pk=$(echo "$out" | sed -n 's/.*peak memory \([0-9.]*\) GiB.*/\1/p' | tail -n 1)
    else
      h=""
    fi
    echo "$out" >> "logs/bench_v3_$m.out"
    if [ -z "$h" ]; then
      say "  did not run (out of memory or error; logs/bench_v3_$m.out)"
      continue
    fi
    ran=1
    echo "$(date '+%F %T') $m [${fb:-default}] $(echo "$out" | grep '^bench: [0-9]' | tail -n 1)" >> logs/final_v3_bench.txt
    say "  ~$h h per fold (budget $budget h)"
    if [ -z "$best_h" ] || awk "BEGIN{exit !($h < $best_h)}"; then best_h=$h; best_fb=$fb; best_pk=$pk; fi
    if awk "BEGIN{exit !($h <= $budget)}"; then echo "${pk:-0}" > "$STATE/peak_$m.txt"; echo "$fb" > "$f"; return 0; fi
  done < <(fallbacks_of "$m")
  if [ "$ran" = 1 ]; then
    say "WARNING: no setting of $m fits $budget h per fold; using the fastest (~$best_h h): ${best_fb:-config as is}"
    echo "${best_pk:-0}" > "$STATE/peak_$m.txt"
    echo "$best_fb" > "$f"
  else
    say "WARNING: $m does not run in any setting; it is left out (logs/bench_v3_$m.out)"
    echo FAILED > "$f"
  fi
}

cpu_round() {  # cpu_round <label> "<names>" "<heavy names>": blend search -> v2 selector + v3 Stage C (background)
  local label=$1 names=$2 heavy=$3
  if [ "$GPU_ONLY" = 1 ]; then say "GPU_ONLY: CPU round '$1' deferred (run: GPU_ONLY=0 scripts/run_all_v3.sh)"; return 0; fi
  local done_file
  done_file="$STATE/round_${label}_$(echo "$names" | xargs | tr ' ' '+').done"
  wait_job "$CPU_PID" round
  CPU_PID=""
  if [ -f "$done_file" ] && [ -z "${FORCE:-}" ]; then
    say "CPU round '$label' on: $names already done"
    return 0
  fi
  say "CPU round '$label' on: $names (log logs/run_all_v3_round_$label.out)"
  (
    set +e
    # shellcheck disable=SC2086
    best=$("$PY" src/blend_search_v3.py --names $names --heavy $heavy --workers "$BG_WORKERS" \
           --out "logs/final_v3_blend_$label.txt" 2>&1 | tee /dev/stderr | grep '^BEST ' | tail -n 1 | sed 's/^BEST //')
    if [ -z "$best" ]; then echo "blend search failed"; exit 1; fi
    echo "best blend weights: $best"
    NAMES="$names" WEIGHTS="$best" OUT="submissions/submission_v3_rescorer_$label.csv" WORKERS=$BG_WORKERS \
      scripts/run_v3.sh rescorer
    NAMES="$names" WEIGHTS="$best" WORKERS=$BG_WORKERS scripts/run_v3.sh stagec && touch "$done_file"
  ) > "logs/run_all_v3_round_$label.out" 2>&1 &
  CPU_PID=$!
}

# =============================================================================================== 0. checks
say "run_all_v3 started (pid $$; stop everything with: kill $$)"
GPU_GIB=$("$PY" -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 2**30) if torch.cuda.is_available() else 0)")
[ "$GPU_GIB" -gt 0 ] || fail "no GPU visible to PyTorch ($PY)"
say "GPU ${GPU_GIB} GiB, ${NCPU} CPU cores (${BG_WORKERS} workers for background CPU jobs)"
FREE_GB=$(df -BG --output=avail "$WORK" | tail -n 1 | tr -dc '0-9')
[ "$FREE_GB" -ge 200 ] || say "WARNING: only ${FREE_GB} GB free under $WORK; v3 needs ~150 GB (+ ~200 GB for Stage B)"
if [ ! -f "$WORK/cache/train_meta.pkl" ] || [ ! -f "$WORK/cache/test_images.npy" ]; then
  say "image cache missing: running prepare_data.py"
  "$PY" src/prepare_data.py --sizes 1024 1536 2048
fi
if ! curl -sfI -m 20 https://gong2.nso.edu/ftp/HA/hag/ > /dev/null; then
  if [ -z "${SKIP_GONG:-}" ]; then say "WARNING: GONG archive unreachable: no Stage B in this run"; fi
  SKIP_GONG=1
fi
if [ "$GPU_ONLY" = 1 ] && [ ! -f "$FILSEG_UNLABELED/U_images.npy" ]; then
  SKIP_GONG=1
  say "GPU_ONLY: no GONG frames yet ($FILSEG_UNLABELED/U_images.npy), so no Stage B in this run (download with GPU_ONLY=0 later)"
fi
DINO=()
for m in "${MEMBERS[@]}"; do
  case $m in
    m1*) DINO+=(convnext_large.dinov3_lvd1689m) ;;
    m2*) DINO+=(vit_large_patch16_dinov3.lvd1689m) ;;
  esac
done
if [ ${#DINO[@]} -gt 0 ]; then
  say "DINOv3 weights (download once into the Hugging Face cache): ${DINO[*]}"
  "$PY" -c "import sys, timm; [timm.create_model(n, pretrained=True, num_classes=0) for n in sys.argv[1:]]" \
    "${DINO[@]}" > logs/run_all_v3_weights.out 2>&1 || fail "could not load the DINOv3 weights (logs/run_all_v3_weights.out)"
fi
say "unit tests"
"$PY" tests/test_v3.py > logs/run_all_v3_tests.out 2>&1 || fail "unit tests failed: logs/run_all_v3_tests.out"
if [ ! -f "$WORK/cache/targets_v3.pkl" ]; then
  say "v3 targets (skeleton, core, instance maps; ~10 min)"
  T=$(date +%s)
  scripts/run_v3.sh targets > logs/run_all_v3_targets.out 2>&1 || fail "targets (logs/run_all_v3_targets.out)"
  timing targets $(( $(date +%s) - T ))
fi

# =============================================================================================== 1. background jobs
V2_SUB="submissions/submission_v3_r34_2048+effv2s_1536_w1-1_linker.csv"
if [ "$GPU_ONLY" = 1 ]; then
  say "GPU_ONLY: Phase-0 diagnostics and Stage C on the v2 maps deferred"
elif preds_done r34_2048 && preds_done effv2s_1536; then
  if [ ! -f logs/final_v3_diag.txt ] || [ ! -f "$V2_SUB" ]; then
    say "background: Phase-0 diagnostics, then Stage C on the v2 maps (log logs/run_all_v3_p0.out)"
    (
      set +e
      [ -f logs/final_v3_diag.txt ] || WORKERS=$BG_WORKERS scripts/run_v3.sh diag
      [ -f "$V2_SUB" ] || NAMES="$V2_NAMES" WEIGHTS="1 1" WORKERS=$BG_WORKERS scripts/run_v3.sh stagec
    ) > logs/run_all_v3_p0.out 2>&1 &
    P0_PID=$!
  fi
else
  say "WARNING: maps of the v2 members missing (work/preds/r34_2048, effv2s_1536): no Phase 0 / Stage C on v2 maps"
fi
if [ -z "${SKIP_GONG:-}" ] && [ ! -f "$FILSEG_UNLABELED/U_images.npy" ]; then
  say "background: GONG download + domain check + QC into $FILSEG_UNLABELED (log logs/run_all_v3_gong.out)"
  scripts/run_v3.sh gong > logs/run_all_v3_gong.out 2>&1 &
  GONG_PID=$!
fi
stop_if prep

# =============================================================================================== 2. loss ablations
LOSS_FILE=$STATE/losses.txt
if [ ! -s "$LOSS_FILE" ]; then
  if [ -n "${SKIP_ABLATIONS:-}" ]; then
    echo "--region jml --w-blob 0.5 --w-sr 0 --aux-heads --deep-sup" > "$LOSS_FILE"
  else
    say "loss ablations on the proxy (GPU, ~5-6 h; log logs/ablate_v3.out, results logs/final_v3_ablations.txt)"
    T=$(date +%s)
    PARALLEL=${ABL_PARALLEL:-4} scripts/ablate_v3.sh > logs/ablate_v3.out 2>&1 || fail "ablations (logs/ablate_v3.out)"
    "$PY" src/choose_losses_v3.py > logs/run_all_v3_losses.out 2>&1 || fail "loss choice (logs/run_all_v3_losses.out)"
    grep '^FLAGS ' logs/run_all_v3_losses.out | tail -n 1 | sed 's/^FLAGS //' > "$LOSS_FILE"
    timing ablations $(( $(date +%s) - T ))
  fi
fi
LOSS=$(cat "$LOSS_FILE")
case $LOSS in --region*) ;; *) rm -f "$LOSS_FILE"; fail "no loss flags chosen (logs/run_all_v3_losses.out)" ;; esac
say "loss flags of the v3 members: $LOSS"
stop_if ablate

# =============================================================================================== 3. benchmarks
for m in "${MEMBERS[@]}"; do
  choose_setting "$m"
  say "$m: $( [ "$(cat "$STATE/extra_$m.txt")" = FAILED ] && echo 'left out' || echo "flags $LOSS $(cat "$STATE/extra_$m.txt")")"
done

# =============================================================================================== 4. teachers
READY=()
for m in "${MEMBERS[@]}"; do
  X=$(cat "$STATE/extra_$m.txt")
  if [ "$X" = FAILED ]; then continue; fi
  if ! folds_done "$m"; then
    say "training $m, 5 folds (log logs/run_all_v3_train_$m.out; per fold logs/${m}_f<k>.out)"
    T=$(date +%s)
    say "  $m: $(parallel_of "$m") fold(s) at once (peak $(cat "$STATE/peak_$m.txt" 2>/dev/null || echo ?) GiB each)"
    PARALLEL=$(parallel_of "$m") EXTRA="$LOSS $X" scripts/train_queue_v3.sh "$m" > "logs/run_all_v3_train_$m.out" 2>&1 || true
    timing "train $m" $(( $(date +%s) - T ))
  fi
  if ! folds_done "$m"; then
    say "WARNING: $m is missing folds after training (logs/run_all_v3_train_$m.out): left out"
    continue
  fi
  if ! preds_done "$m"; then
    say "maps of $m (OOF + test, 8 TTA views)"
    T=$(date +%s)
    "$PY" src/predict_v3.py --name "$m" --tta 8 > "logs/predict_v3_$m.out" 2>&1 || say "WARNING: prediction of $m failed"
    timing "predict $m" $(( $(date +%s) - T ))
  fi
  if preds_done "$m"; then READY+=("$m"); fi
  if [ ${#READY[@]} -eq 1 ] && [ "$m" = "${READY[0]}" ] && [ ${#MEMBERS[@]} -gt 1 ]; then
    cpu_round "early" "$V2_NAMES ${READY[*]}" ""   # an early v3 submission while the GPU trains the next member
  fi
done
[ ${#READY[@]} -gt 0 ] || fail "no v3 member finished"
HEAVY=""
for m in "${READY[@]}"; do case $m in m1*|m2*) HEAVY="$HEAVY $m" ;; esac; done
cpu_round "teachers" "$V2_NAMES ${READY[*]}" "$HEAVY"
stop_if teachers

# =============================================================================================== 5. Stage B
if [ -z "${SKIP_GONG:-}" ] && [ -z "${SKIP_STUDENTS:-}" ]; then
  wait_job "$GONG_PID" gong
  GONG_PID=""
  NU=$("$PY" -c "import sys, numpy as np; print(len(np.load(sys.argv[1], mmap_mode='r')))" \
       "$FILSEG_UNLABELED/U_images.npy" 2>/dev/null || echo 0)
  TEACH=()
  for m in "${READY[@]}"; do case $m in m1_*|m2_*) TEACH+=("$m") ;; esac; done
  if [ "$NU" -lt "$MIN_U" ] || [ ${#TEACH[@]} -eq 0 ]; then
    say "Stage B skipped: $NU unlabelled frames (minimum $MIN_U), teachers: ${TEACH[*]:-none}"
  else
    TT=$(IFS=+; echo "${TEACH[*]}")
    say "pseudo-labels of $NU GONG frames by ${TEACH[*]}, folds 0-4, views $PL_VIEWS (log logs/run_all_v3_pseudo.out)"
    T=$(date +%s)
    # shellcheck disable=SC2086
    "$PY" src/pseudo_label_v3.py --teachers "${TEACH[@]}" --views $PL_VIEWS > logs/run_all_v3_pseudo.out 2>&1 \
      || fail "pseudo-labels (logs/run_all_v3_pseudo.out)"
    timing pseudo $(( $(date +%s) - T ))
    for t in "${TEACH[@]}"; do
      s=$(student_of "$t")
      X=$(cat "$STATE/extra_$t.txt")
      if ! folds_done "$s"; then
        say "training student $s (teacher setting: ${X:-as is})"
        T=$(date +%s)
        PARALLEL=$(parallel_of "$t") EXTRA="$LOSS $X --unlabeled $FILSEG_UNLABELED --pseudo $FILSEG_UNLABELED/pl/$TT/fold{fold}.npy" \
          scripts/train_queue_v3.sh "$s" > "logs/run_all_v3_train_$s.out" 2>&1 || true
        timing "train $s" $(( $(date +%s) - T ))
      fi
      if ! folds_done "$s"; then say "WARNING: student $s is missing folds: left out"; continue; fi
      if ! preds_done "$s"; then
        "$PY" src/predict_v3.py --name "$s" --tta 8 > "logs/predict_v3_$s.out" 2>&1 || say "WARNING: prediction of $s failed"
      fi
      if preds_done "$s"; then
        READY+=("$s")
        HEAVY="$HEAVY $s"
        mkdir -p "$WORK/ablations_v3"
        if [ "$GPU_ONLY" = 1 ]; then continue; fi   # the student-vs-teacher check is CPU post-processing
        {
          echo "== student $s vs teacher $t (v1 post-processing, best threshold; gate: student OOF > teacher OOF)"
          "$PY" src/score_member_v3.py --names "$t" --out "$WORK/ablations_v3/gate_$t.csv" --workers "$BG_WORKERS" | tail -n 1
          "$PY" src/score_member_v3.py --names "$s" --out "$WORK/ablations_v3/gate_$s.csv" --workers "$BG_WORKERS" | tail -n 1
          "$PY" src/compare_v3.py "$WORK/ablations_v3/gate_$t.csv" "$WORK/ablations_v3/gate_$s.csv"
        } 2>&1 | grep -v -i warn | tee -a logs/final_v3_students.txt || true
      fi
    done
    cpu_round "students" "$V2_NAMES ${READY[*]}" "$HEAVY"
  fi
fi

# =============================================================================================== 6. summary
say "waiting for the background CPU jobs"
wait_all
timing total $(( $(date +%s) - T_ALL ))
if [ "$GPU_ONLY" = 1 ]; then
  say "GPU work finished (members ready: ${READY[*]}). The CPU stages were deferred; run them now with:"
  echo "    GPU_ONLY=0 nohup scripts/run_all_v3.sh > logs/run_all_v3_cpu.out 2>&1 &"
  echo "  (skips all finished GPU work; runs diagnostics, blend search, v2 selector, Stage C and the summary table)"
  exit 0
fi
"$PY" - "$STATE/candidates.tsv" "$WORK/selectors_v2/r34_1024+rgy040_1024+r34_1536+r34_2048+effv2s_1536_config.json" <<'EOF'
import json
import os
import sys

rows = {}
if os.path.exists(sys.argv[2]):
    rows["v2 (before v3)"] = (json.load(open(sys.argv[2]))["nested_pq"], "submissions/submission_v2.csv")
if os.path.exists(sys.argv[1]):
    for line in open(sys.argv[1]):
        label, pq, path = line.rstrip("\n").split("\t")
        rows[label] = (float(pq), path)   # a rerun of the same candidate replaces the older line
print("\nhonest nested out-of-fold PQ of every candidate (organisers' metric, 707 training images):")
for label, (pq, path) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
    print(f"  {pq:.4f}  {label:<70s} {path}")
if rows:
    label, (pq, path) = max(rows.items(), key=lambda kv: kv[1][0])
    print(f"\nrecommended upload: {path}  (nested OOF PQ {pq:.4f}; nothing was submitted)")
EOF
cat <<EOF

$(date '+%F %T') | run_all_v3 finished. For the report: logs/final_v3_diag.txt (taxonomy, ceilings),
logs/final_v3_ablations.txt, logs/final_v3_bench.txt, logs/final_v3_students.txt, logs/final_v3_stagec_*.txt,
work/eval_v3/ and work/diag_v3/ (rubric figures and tables), logs/timing_v3.txt.
EOF
