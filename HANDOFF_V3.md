# Handoff: Solar Filament Segmentation 2026, state at 2026-10-18 18:05

Start the next session with: "Read HANDOFF_V3.md, V3.md and DECISIONS.md, then check the v3 run."

## 1. Goal and rules

- Kaggle `filament-segmentation-2026` (IEEE BigData Cup, MAGFiLO, GONG H-alpha 2048x2048). Metric: Panoptic Quality,
  every annotation set scored separately. Deadline **2026-11-15 23:59 CDT** (report, public repo, notebook, Google form).
- Winners are chosen by the organisers, who reproduce code. 70 % quantitative (PQ, IoU/Dice distributions,
  1:n / n:1 relations, efficiency) and 30 % report + code quality.
- **Never** use leaked test labels (public MAGFiLO v1.0 Kaggle dataset `esairlab/magfilo-v1-0-...` contains 139/180
  test images with labels), `spine`/`bbox`/`area`/`category_id`, metric exploits (duplicate masks), or LB tuning.
  The 0.55-0.64 public scores are leaked-label submissions; the honest frontier is ~0.38-0.41 public.
- Human annotators agree at PQ 0.340; PQ 0.65+ is not reachable honestly. The user keeps asking for 0.65-0.80;
  explain this plainly, don't promise it.

## 2. User preferences

- The user runs long jobs; Claude writes code, checks it statically, and gives exact commands. New files get the
  current version suffix (`_v3`). Never submit to Kaggle; the user submits.
- Budget-conscious: credit was **\$58.46 at \$1.99/h** (stated ~10:00 on Oct 18). The instance bills while idle.
- Wants GPU used fully; but CPU stages overlap GPU training in the budget run because idle GPU is billed.
- Short, plain answers; the user is not a native English speaker.

## 3. Results so far (nested out-of-fold PQ, organisers' metric, 707 training images)

| system | OOF PQ | public LB |
|---|---|---|
| human vs human | 0.340 | |
| v1: 3 U-Nets + LightGBM rescorer | 0.4381 | 0.38 (`submissions/submission.csv`) |
| v2: + r34_2048, effv2s_1536, extent selector | **0.4438** | 0.38 (`submission_v2.csv`, selected) |
| v2 masks eroded 1 px / dilated 1 px | | 0.37 / 0.35 |
| v3 Stage C (pool + linker + scorer) on v2 maps | 0.4445 | not submitted |
| v3 Stage C on v2 + M3 | 0.4443 | not submitted |
| v2 selector on v2 + M3 (`submission_v3_rescorer_after_m3.csv`) | 0.4449 | 0.38 (submitted 18:11) |

Findings:
- **1 px mask-width hypothesis rejected** (erode/dilate both worse publicly). The OOF -> LB gap (~0.06) is unexplained.
- Public LB shows 2 decimals (also via the Kaggle API); OOF gains < ~0.01 are invisible there.
- Stage C alone gives ~+0.001 (not significant). Pool oracle rises 0.497 -> 0.519, but the scorer cannot capture it
  (annotator disagreement makes u unpredictable).
- Proxy ablations (r34 @1024, 60 epochs, v1 post-processing): dice 0.3861, JML1 0.3864 (same), **full v3 loss
  (JML1 + blob + S/D heads + deep supervision) 0.3956, +0.009, significant**. Chosen flags:
  `--region jml --w-blob 0.5 --w-sr 0 --aux-heads --deep-sup` (`work/v3_state/losses.txt`).
- Per-fold validation PQ (no TTA, pq@0.6l8): M3 ResNet-34 @1536 0.367-0.392; **M4 DINOv3 ViT-B @1536 0.397-0.406**
  (folds 0-1), clearly the strongest single member so far.

## 4. What is running (started by the user at 13:06 with `scripts/run_budget_v3.sh train`)

- Orchestrator `scripts/run_all_v3.sh`, pid/pgid **957303**, watchdog hard stop **2026-10-19 14:28** (credit - \$8).
- Env: `GPU_ONLY=0 SKIP_GONG=1 SKIP_STUDENTS=1 MEMBERS="m3_r34_1536_v3 m4_vitb_1536_v3 m5_cnxs_1536_v3"
  EXTRA_ALL="--epochs 60" BG_WORKERS=12 VARIANTS="abl_dice abl_jml abl_full"`.
- Done: benchmarks, ablations, M3 (5 folds + maps), CPU round after M3.
- Running at 18:03: M4 fold 2 (1.19 h per fold, one fold at a time, 80 GB).
- ETA: M4 done ~21:30, CPU round after M4 ~22:30, M5 (ConvNeXt-S, ~1 h/fold) done ~02:30, final CPU round
  (blend search, v2 selector, Stage C, summary table) ~03:30-05:00 on Oct 19. Total run ~15-17 h, ~\$32-36.

Check:
```bash
cd /root/kaggle
tail -n 30 logs/run_all_v3.out                     # phases; last lines = summary + recommended file
cat work/v3_state/candidates.tsv                   # every candidate: label, nested OOF PQ, submission file
for f in work/models/*/fold*.log; do echo "$f: $(grep '^epoch' $f | tail -1 | cut -c1-120)"; done
amd-smi metric | grep -E "GFX_ACTIVITY|USED_VRAM"
scripts/run_budget_v3.sh stop                      # stop everything (resumable: rerun 'train' to continue)
```

When it finishes: upload the recommended file **without** `_erode1` / `_dilate1` (rejected). Then shut the instance
down. Commit results (weights are Git LFS; `.gitignore` excludes resume checkpoints, pools, ablation models).

## 5. Code map (v3; v1/v2 unchanged)

- Runbook `V3.md`; decisions `DECISIONS.md`; budget wrapper `scripts/run_budget_v3.sh` (plan/stagec/train/stop);
  orchestrator `scripts/run_all_v3.sh` (resumable; state in `work/v3_state/`); stages `scripts/run_v3.sh`;
  fold queue `scripts/train_queue_v3.sh`; ablations `scripts/ablate_v3.sh`.
- Training `src/train_v3.py` (YAML `configs/*.yaml` + CLI overrides, `--bench-steps N`, `--resume`), inference
  `src/predict_v3.py`, models `src/filseg/models_v3.py` (smp_unet / cnx_unet / vit_dpt, HR stem with
  `stem_native` and `stem_ch`), losses `src/filseg/losses_v3.py`, targets `src/prepare_targets_v3.py` (built).
- Stage C: `src/build_pool_v3.py`, `src/train_linker_v3.py`, `src/train_scorer_v3.py`, `src/make_submission_v3.py`,
  `src/blend_search_v3.py`; diagnostics `src/diagnose_v3.py`, `src/ceilings_v3.py`, `src/compare_v3.py` (bootstrap gate).
- Stage B (not run): `src/gong_download_v3.py` (source https://gong2.nso.edu/ftp/HA/hag/), `gong_domain_check_v3.py`,
  `gong_qc_v3.py`, `pseudo_label_v3.py`, `configs/students/`.
- Tests: `tests/test_v3.py` (11 unit tests, CPU), `tests/smoke_stagec_v3.py` (Stage C on 8 real images, ~10 s).

## 6. Measured costs on this MI300X (20-step benchmarks, 60 epochs per fold, bs 8)

proxy r34_1024 0.13 h (19 GB); M3 r34_1536 0.28 h (38 GB); M4 ViT-B @1536 1.19-1.33 h (80 GB); M5 ConvNeXt-S @1536
0.98 h (82 GB); ViT-S 0.77 h; ConvNeXt-T 0.75 h. The 2048 px HR stem costs ~2x a ResNet-34 -> budget members use a
1536 px stem. M1 (ConvNeXt-L @2048) / M2 (ViT-L @1536 + 2048 stem) were not affordable.

## 7. Pitfalls learned

- Never edit a bash script in place while it runs (bash reads by byte offset). This broke one run.
- `dataclasses`: `f.default_factory is MISSING`, not None (crashed train_v3.py once).
- No `cudnn.benchmark` on MIOpen; first steps of a new shape compile kernels (M4: 5.5 min, cached afterwards).
- Don't `pkill -f` patterns that appear in the shell's own command line; kill by explicit PID / process group.
- Kaggle LB: only 2 decimals; probes must show > 0.02 to mean anything.

## 8. Next steps (in order of value)

1. After the run: compare the final candidates (`candidates.tsv`); expect ~0.445-0.455 OOF. Upload the best (user).
2. If credit remains: M4 is the strongest member; a second ViT-B seed or 80 epochs is the most likely gain.
3. Report (4 pages, Overleaf template): honest numbers table, ceilings (`logs/final_v3_diag.txt`,
   `work/diag_v3/`), rubric figures (`work/eval_v3/`), efficiency table, the rejected width probe.
4. Repo hygiene for the deadline: commit code + configs + weights (LFS), notebook reproducing test predictions,
   pinned requirements, DINOv3 licence note.
5. Not done (blueprint cut list): FilaRefine, M4/EoMT proposals, DenseCRF, supervoxel/edge losses, Stage B students,
   efficiency student.
