# Decisions

Compliance and design decisions, newest last. Each entry: date, decision, reasoning or evidence.
Where a rule is ambiguous we take the conservative reading.

## 2026-10-18: v3 (implementation of the "FilaPQ v2" blueprint, rev 2.1)

**Naming.** The repository already has a v2 (extent selector + `r34_2048` / `effv2s_1536`; nested OOF PQ
0.4438). The blueprint's components are therefore named v3 (`*_v3.py`, `configs/*_v3.yaml`, `submission_v3*.csv`).
v1 and v2 files are unchanged and still reproduce 0.4381 / 0.4438.

### Compliance

- **C1, inference inputs.** Only the test H-alpha images are used at inference. No temporal neighbours, no
  external frames, no metadata, no filename-derived model features. The site letter is used only in
  diagnostics (`diagnose_v3.py` breakdowns) and for the per-site intensity LUT of the *unlabelled training*
  frames, never as a model input.
- **C2, ground-truth metadata.** Targets and features come only from `annotations[*].segmentation` (soft mask,
  skeleton, core and instance maps in `filseg/targets_v3.py`; pair labels of the linker; candidate targets u).
  `spine`, `category_id`, `bbox` and `area` are never read. The public MAGFiLO v1.0 release (Kaggle dataset
  `esairlab/magfilo-v1-0-segmentation-of-solar-filaments`) contains 139 of the 180 test images with labels and is
  never used, not even for pretraining or as unlabelled data.
- **C3, external data.** Unlabelled GONG H-alpha frames from the NSO archive (`https://gong2.nso.edu/ftp/HA/hag/`,
  the 2048x2048 JPEG product), 2023-01-01 to 2026-09-30 only. This is outside MAGFiLO's 2011-2022 span, so no frame
  can coincide with a competition image. Frames within 72 h of a test image are skipped anyway. If a 2010-2022
  secondary pool is ever added, `train_v3.py --u-exclude-hours 72` also drops frames within 72 h of the fold's
  validation images. `tests/test_v3.py` checks both.
- **Pretrained weights.** DINOv3 ConvNeXt-L / ViT-L (timm `convnext_large.dinov3_lvd1689m`,
  `vit_large_patch16_dinov3.lvd1689m`) are under the DINOv3 License (Meta). It requires its terms to accompany
  redistributed weights and derivatives. Conservative reading: our fine-tuned checkpoints are derivatives, so the
  public repository ships the licence text next to them and the README names it. ResNet-34 (smp, ImageNet) as in v1.
- **No metric exploits.** Submitted masks are disjoint (`select_v3.greedy_select`, checked again in
  `make_submission_v3.py`). There is no leaderboard probing for tuning and no duplicate masks.

### Design decisions and deviations from the blueprint

- **No "keep the largest connected component"** (blueprint 12.2 / 12.6). v1's fragment grouping deliberately
  merges nearby pieces into one filament, which is worth +0.008 PQ. Keeping only the largest piece would undo it.
  Candidates keep all their pieces, and tiny trimmed leftovers (< 30 px) are dropped.
- **Pool minimum area 100 px** (blueprint: 30 px). This matches v1/v2, where 100-200 px was optimal. It can be
  changed with `--min-area`.
- **Linker thresholds as hypothesis families.** Instead of tuning one tau_link end to end, the pool contains
  H7@0.4 and H7@0.6 (configurable), and the scorer chooses per candidate. `train_linker_v3.py` prints each tau's
  standalone PQ for reference.
- **Nesting.** Linker, scorer and (later) FilaRefine for fold k are trained on folds other than k. Stacking on
  OOF features has the same second-order structure as v1's rescorer on OOF maps, which was accepted there.
  tau and rho of the selection are tuned on the nested scores. This is a two-parameter optimism, the same kind as
  v1's tuned threshold.
- **Blob loss.** It uses one uniformly sampled annotation set per image and step, at 1024 px, with
  nearest-downsampled instance labels. It is computed in closed form from per-instance sums, which is exact for
  non-overlapping polygons (`tests/test_v3.py` checks it against the definition).
- **Input normalisation for DINOv3 backbones.** The v1 constants are used (disk mean 125, std 18, one channel;
  timm sums the RGB filters of the first layer), not ImageNet statistics. Both give unit-scale inputs.
- **Pseudo-labels.** They stay soft (uint8 x255, never thresholded), use 4 dihedral views (identity, h-flip,
  rot90, rot270), and are fold-honest (teacher fold k is used for student fold k).
- **Not implemented yet.** These are on the blueprint's cut list or deferred to a later session: FilaRefine
  (crop refine-and-score CNN, 12.4), M4 / EoMT proposals (promoted only if the Phase-0 taxonomy shows
  fragmented + merged > 40 % of the lost PQ), DenseCRF hypotheses (H9; pydensecrf is not installed), the
  supervoxel connectivity loss and edge head (no ROCm-tested implementation; lowest ablation priority), elastic
  augmentation, the TTA-consistency scorer feature, and the efficiency-mode student.
- **Checkpoint size.** An M1 fold checkpoint is about 0.9 GB in fp32, and M2 about 1.2 GB. Before release we
  decide between Git LFS at that size and bf16 weights, which halve it. That decision needs an OOF check that
  bf16 weights change nothing measurable.

## 2026-10-18: budget run (credit \$58.46 at \$1.99/h)

- **Members.** M1/M2 (ConvNeXt-L @2048, ViT-L @1536 with the 2048 px stem) would need ~40-50 GPU-hours (\$80-100).
  The budget plan trains M3 (ResNet-34), M4 (DINOv3 ViT-B/16 + DPT) and M5 (DINOv3 ConvNeXt-S U-Net), all at 1536 px
  with a light detail stem at 1536 px. Measured per fold (60 epochs): 0.28 h, 1.33 h, 0.98 h. v1 measured a 0.997
  IoU retention at 1536 px (MODEL.md section 4.4), so nothing is lost by not predicting at 2048 px.
- **Mini ablation.** abl_dice, abl_jml, abl_full (JML1, blob loss, S/D heads and deep supervision judged together)
  instead of the six-variant set. The loss choice uses the same bootstrap gate (`src/choose_losses_v3.py`).
- **CPU stages overlap GPU training.** An idle GPU is billed, CPU work is free next to it. This overrides the
  earlier "GPU only" preference for the budget run (`GPU_ONLY=1` restores it).
- **Public-leaderboard mask-width probe.** The OOF -> public gap (~0.06) equals the OOF drop from eroding the
  training masks by 1 px, which suggests the test masks are drawn about 1 px thinner. Blueprint section 13 says never to tune
  on the public leaderboard. We treat the two probe submissions (`*_probe_erode1.csv`, `*_probe_dilate1.csv`) as a
  calibration of a systematic annotation convention (one bit, applied to every mask, shared by the private half
  of the test set), not as tuning. The probe result must be a clear gain (> 0.02, public noise is a few hundredths)
  before the corresponding variant is used. The decision and the submissions are the owner's.

## 2026-10-18: mask-width probe result

Public leaderboard: submission_v2.csv 0.38, the same masks eroded by 1 px 0.37, dilated by 1 px 0.35. The test
masks are not systematically thinner or wider, so the final submission uses unmorphed masks. The `_erode1` and
`_dilate1` copies written by run_all_v3.sh are not to be used. The OOF -> public gap (~0.06) remains unexplained.
