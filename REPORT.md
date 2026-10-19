# Solar Filament Instance Segmentation under Annotator Disagreement: Soft Multi-Annotator Targets, DINOv3 Members and PQ-Additive Instance Selection

*Technical report for the Solar Filament Segmentation Challenge 2026 (Markdown source of the 4-page Overleaf report).*

## Abstract

We segment solar filaments in 2048x2048 GONG H-alpha images. The metric is Panoptic Quality (PQ), computed
separately against every annotator of an image. Two annotators of the same image agree at only PQ = 0.340,
so annotator disagreement, not pixel accuracy, limits every honest system. We train U-Net and DINOv3 networks on
soft consensus targets. We show that PQ is exactly additive over disjoint predicted instances, and we use this to
tune post-processing and to train instance scorers with nested cross-validation. Our best system (ResNet-34,
EfficientNetV2-S and DINOv3 ConvNeXt-S members, a hypothesis pool and a stacked scorer) reaches a nested
out-of-fold PQ of **0.4477** on the 707 training images and **0.39** on the public leaderboard (earlier systems: 0.38). We
report the error breakdown, the ceilings, the leaderboard gap and its most likely cause, and the ideas that did not work.

## 1. Data

The training set (MAGFiLO) has 707 full-disk images with 1,154 annotation sets and 8,199 filament polygons:
411 images have one annotator, 145 have two and 151 have three. Filaments are thin and elongated (median area
1,228 px; 5th percentile 317 px). The test set has 180 images. Train and test images come from the same six
GONG sites (B, C, L, M, T, U) and the same years (2011-2022) in similar proportions. The median gap to the
nearest training image is ~37 h, the same as between training images. We therefore treat the test images as a
random sample of the same observation pool.

Each annotation set is identified by `<annotator>-<image>`. The 37 annotator codes consist of a 4-digit group
and a 2-digit slot (for example group `0104`, slots `01`/`02`/`03` on the same images). Annotation styles
differ strongly between codes: from 3.4 to 10.5 polygons per image and a median filament area from 640 to
2,100 px.

## 2. Metric analysis

**Definition.** For every annotation set, all predicted filaments of the image are compared with its GT
filaments. A pair with IoU > 0.5 is a TP and contributes its IoU. Predictions without a hit are FPs, and GT
filaments without a hit are FNs. Counts are pooled over all sets: PQ = sum IoU_TP / (TP + 0.5 FP + 0.5 FN). An
image with three annotators therefore counts three times. `src/filseg/metric.py` reproduces the organisers'
notebook exactly (checked pair by pair).

**Human agreement and ceilings** (`src/ceilings_v3.py`):

| reference | PQ | SQ | RQ |
|---|---|---|---|
| annotator vs annotator (all pairs of the same image) | 0.340 | 0.634 | 0.536 |
| consensus of two annotators vs the held-out third (151 images) | 0.333 | 0.632 | 0.528 |
| oracle: best single disjoint prediction set with all labels known (296 multi-set images) | 0.666 | 0.825 | 0.807 |
| our system (nested out-of-fold) | **0.448** | 0.683 | 0.655 |

Even an oracle that sees every label cannot exceed 0.666 on multi-annotator images with one set of masks,
because the annotators contradict each other. Our system is above the human consensus because it predicts the
average annotator rather than one particular annotator.

**PQ is additive over disjoint instances.** If predicted masks do not overlap, a GT filament can be hit by at
most one prediction. For any subset S of predicted instances,

    PQ(S) = sum_{i in S} a_i / (0.5 * N_gt + sum_{i in S} d_i)

where a_i is the sum of instance i's TP IoUs over its image's m_i annotation sets and d_i = m_i / 2.
Consequences:
1. Adding instance i raises PQ if and only if its expected per-annotator TP IoU u_i = a_i / m_i exceeds PQ / 2
   (about 0.22 here). This gives a principled keep threshold.
2. Once instances are extracted, every threshold or filter is a sum, so the search is exact and instant.
3. The metric counts every IoU > 0.5 pair, so overlapping duplicate masks inflate it. On our out-of-fold
   predictions, submitting each mask twice raises PQ from 0.4385 to 0.4831. **We submit strictly disjoint masks**
   (checked in every submission writer).

## 3. Method

### 3.1 Targets
Every polygon is rasterised with pycocotools, exactly like the ground truth. The soft target of a pixel is the
fraction of the image's annotators whose polygon covers it. We also derive a soft skeleton T, a soft core D and
one instance map per annotation set (`src/prepare_targets_v3.py`). Only the `segmentation` field is read. The
`spine`, `bbox`, `area` and `category_id` fields are never used.

### 3.2 Networks
All members take the full disk as a 1-channel image (normalised with mean 125 and std 18) and output a
filament probability map.

| member | architecture | encoder (pretraining) | input px | loss | epochs |
|---|---|---|---|---|---|
| r34_1024, r34_1536, r34_2048 | U-Net + scSE | ResNet-34 (ImageNet) | 1024 / 1536 / 2048 | BCE + soft Dice | 60-80 |
| rgy040_1024 | U-Net + scSE | RegNetY-040 (ImageNet) | 1024 | BCE + soft Dice | 80 |
| effv2s_1536 | U-Net + scSE | EfficientNetV2-S (ImageNet) | 1536 | BCE + soft Dice | 80 |
| M3 | U-Net, light detail stem | ResNet-34 (ImageNet) | 1536 | v3 loss | 60 |
| M4 | DPT decoder, light detail stem | DINOv3 ViT-B/16 (LVD-1689M) | 1536 | v3 loss | 60 |
| M5 | U-Net, light detail stem | DINOv3 ConvNeXt-S (LVD-1689M) | 1536 | v3 loss | 60 |

M3-M5 (`src/filseg/models_v3.py`) share a decoder made of GroupNorm/GELU/scSE blocks with a stride-1/2 detail
stem, plus auxiliary skeleton (S) and core (D) heads and deep supervision.

### 3.3 Losses
The v1 members use BCE + soft Dice against the soft target. The v3 members (`src/filseg/losses_v3.py`) use:
- BCE and the Jaccard metric loss JML1 (Wang et al., NeurIPS 2023), which, unlike soft Dice, is minimised at
  P = y for soft labels;
- a blob loss (Kofler et al., IPMI 2023) over the instances of one randomly sampled annotation set, so that
  every filament counts equally, as in PQ (computed in closed form from per-instance sums);
- BCE + Dice semimetric on the S head and SmoothL1 on the D head, and deep supervision at strides 4 and 8.

The losses were chosen by a 5-fold proxy ablation (ResNet-34 at 1024 px) with a paired bootstrap gate (adopt
only if the mean gain is >= 0.003 and P(gain > 0) >= 0.9):

| proxy loss | PQ (best extraction threshold) |
|---|---|
| BCE + soft Dice (v1) | 0.3861 |
| BCE + JML1 | 0.3864 (+0.0003, not adopted) |
| BCE + JML1 + blob + S/D heads + deep supervision | **0.3956** (+0.0092, 90% CI [+0.006, +0.012], adopted) |

### 3.4 Training
AdamW with warm-up then cosine decay, batch 8, bf16, EMA of the weights (0.998) and gradient clipping at 2.
DINOv3 encoders use a lower encoder LR (1e-4) with layer-wise decay 0.9 for the ViT. Images are sampled in
proportion to their number of annotation sets, which mirrors the metric's weighting. All augmentation runs on
the GPU, with the whole training set kept in GPU memory: arbitrary rotation, flips, scale 0.87-1.15, shift,
gamma, contrast, brightness, blur and noise.

### 3.5 Validation
We use 5 image-level folds: all annotation sets of an image share a fold. Every number in this report is
out-of-fold. Every learned post-processing stage (rescorer, selector, linker, scorer) is **nested**: for fold
k it is trained on the other folds' out-of-fold predictions, and only two global thresholds are tuned on the
nested scores.

### 3.6 Inference and instance extraction
- **Probability maps.** 8 dihedral TTA views, averaged over the 5 fold models of a member and then over the
  members with blend weights. Maps are bilinearly upsampled to 2048 px.
- **Instances.**
  1. Seeds are the connected components of p > t_seed.
  2. Seeds closer than ~16 px are grouped into one filament, because annotators draw one polygon around a
     broken filament (+0.008 PQ).
  3. The extent is p > t_mask, grown from the seeds by watershed, with holes filled since GT polygons have none.
  Masks are disjoint by construction.
- **Calibration matters.** The v3-loss members produce softer maps (mean foreground probability 0.80 vs 0.90
  for v1 members), so their best thresholds are 0.4, not 0.6:

| member alone (out-of-fold, best area/score filter) | t = 0.6 | tuned t |
|---|---|---|
| r34_2048 | 0.4331 | |
| effv2s_1536 | 0.4355 | |
| M3 (ResNet-34, v3 loss) | 0.4131 | 0.4303 (t = 0.4) |
| M4 (DINOv3 ViT-B, v3 loss) | 0.4268 | **0.4410** (t = 0.4) |
| M5 (DINOv3 ConvNeXt-S, v3 loss) | 0.4303 | not measured |

### 3.7 Instance selection
- **Rescorer (v1, `src/train_rescorer.py`).** LightGBM regresses u_i from 26 features of each candidate,
  weighted by m_i: probability statistics, contrast against a 12 px ring, shape, position on the disk and
  context. Instances with predicted u > t are kept. The tuned t (0.18-0.26) matches the theoretical PQ/2.
- **Extent selector (v2, `src/train_selector_v2.py`).** Every filament gets extent hypotheses at levels
  0.4/0.5/0.6/0.7, and the selector picks per filament the level with the highest predicted u. Oracle headroom:
  0.494 -> 0.515.
- **Stage C (v3, `src/build_pool_v3.py`, `src/train_linker_v3.py`, `src/train_scorer_v3.py`).** A pool of
  hypotheses (threshold families, a learned fragment linker), a stacked LightGBM scorer and an overlap-aware
  greedy selection. Pool oracle: 0.497 -> 0.524 on the final members.

## 4. Results

### 4.1 Out-of-fold PQ (707 images, organisers' metric, nested where learned)

| system | PQ | SQ | RQ | public LB |
|---|---|---|---|---|
| r34_1024, threshold 0.5, no TTA, no filtering (fold 0) | 0.380 | | | |
| r34_1024 + TTA + tuned thresholds/filters | 0.4205 | | | |
| + rescorer | 0.4272 | | | |
| v1: 3-member ensemble + rescorer | 0.4381 | 0.677 | 0.647 | 0.38 |
| v2: r34_2048 + effv2s_1536 + extent selector | 0.4438 | 0.680 | 0.652 | 0.38 |
| v3 Stage C on the v2 maps | 0.4445 | 0.682 | 0.652 | |
| v2 selector on r34_2048 + effv2s_1536 + M3 | 0.4449 | 0.682 | 0.652 | 0.38 |
| 4-member blend incl. M4, tuned extraction threshold, before selection | 0.4414 | | | |
| v2 selector on r34_2048 + effv2s_1536 + M5 | 0.4455 | | | |
| **final: Stage C on r34_2048 + effv2s_1536 + M5 (weights 1 1 1)** | **0.4477** | 0.683 | 0.655 | **0.39** |

The final file is `submissions/submission_v3_r34_2048+effv2s_1536+m5_cnxs_1536_v3_w1-1-1_linker.csv` (1,067
filaments on 179 of 180 test images).

Per-fold validation PQ of the v3 members (single model, no TTA, t = 0.6, link 8): M3 0.367-0.392,
M4 0.385-0.406, M5 0.371-0.412. The DINOv3 members are the strongest single networks.

Selection stages add only a little:
- the rescorer adds +0.007 to a single member;
- the extent selector adds +0.0004 to +0.003;
- Stage C adds +0.0007 on the v2 maps and +0.0022 on the final members (0.4477 vs 0.4455 for the selector,
  0.4442 for the v1 extent).

Their oracles show 3-8 points of headroom, but the scorers capture little of it, because which extent a given
annotator draws is largely unpredictable from the image.

### 4.2 Rubric metrics (final system, out-of-fold, `src/evaluate_v2.py`)

| metric | value |
|---|---|
| matched pairs with IoU > 0.5 | 75.7 % |
| IoU of matched pairs, mean / median | 0.595 / 0.646 |
| Dice of matched pairs, mean / median | 0.726 / 0.785 |
| semantic Dice per image, mean | 0.676 |
| fragmentation (1:n): GT filaments overlapped by 2+ predictions | 3.0 % |
| over-merging (n:1): predictions overlapping 2+ GT filaments | 2.9 % |

### 4.3 Where PQ is lost (`src/diagnose_v3.py`, v2 system, PQ 0.4438)

| cause | share of the lost PQ |
|---|---|
| IoU deficit of true positives (boundary disagreement) | 56 % |
| missed filaments (no overlap) | 15 % |
| predictions overlapping GT with IoU <= 0.5 (extent) | 8 % FP + 6 % FN |
| spurious predictions | 6 % |
| fragments / merges | 4 % + 4 % + 2 % |

Small filaments are the weakest group (< 500 px: PQ 0.29, recall 0.36), followed by the limb (r/R > 0.8: PQ
0.37). Fragmentation and merging together cost only 9 % of the lost PQ, so query-based instance models were
not pursued. PQ is uniform across observatory sites (0.43-0.45) and across the number of annotators per image
(0.44 / 0.46 / 0.44). It is **not** uniform across annotator groups: 0.40-0.55 per group (0.40 for groups 0401
and 0402, 0.52-0.55 for 0204 and 0304).

### 4.4 Public leaderboard and the out-of-fold gap

Our submissions score 0.38-0.39 on the public leaderboard (shown with two decimals), about 0.06 below the
out-of-fold PQ. Other teams report the same gap. What we checked:
- **Not chance.** Random 90-image subsets of our out-of-fold predictions have SD 0.012, and a subset scores
  <= 0.385 with probability < 0.0003.
- **Not mask width.** The same submission eroded by 1 px scores 0.37 and dilated by 1 px scores 0.35, so the
  test masks are not systematically thinner or wider. We submit unmorphed masks.
- **Not image or calibration shift.** Years and sites are matched, and single-fold and 5-fold-mean test maps
  differ from out-of-fold maps by only about 1 % in foreground fraction.
- **Most likely an annotator shift.** Our PQ varies by 0.15 between annotator groups. A test set labelled
  mainly by harder or unseen groups explains a gap of this size. The test annotators are unknown to us, so we
  did not adapt to them, and we tuned nothing on the leaderboard.

## 5. What did not work

| idea | result |
|---|---|
| 1 px erosion / dilation of the final masks | public 0.37 / 0.35 (vs 0.38) |
| keeping only the largest connected component | removes the +0.008 of fragment grouping |
| per-instance morphological closing, radius 2-6 | -0.003 to -0.016 |
| multi-scale TTA; per-instance adaptive extent | +0.0006; +0.0013 (fold 0, not adopted) |
| JML1 instead of soft Dice alone | +0.0003, not significant |
| learned fragment linker + stacked scorer (Stage C) | +0.0007 on the v2 maps, +0.0022 on the final members (kept) |
| blend search at the v1 threshold (0.6) for v3 members | excluded M4, the best member, from both the after-M4 and the final blend; fixed by tuning the threshold per blend (`src/blend_search_tuned_v3.py`) |

Not run because of the compute budget: semi-supervised training on unlabelled GONG frames (code in
`src/gong_*_v3.py`, `src/pseudo_label_v3.py`), DINOv3 ConvNeXt-L / ViT-L members at 2048 px, and an
annotator-conditioned network (`src/train_rater_v3.py`).

## 6. Compliance

- At inference the model sees only the test image: no metadata, no file-name features, no temporal neighbours.
- Ground-truth fields used: `segmentation` only. The annotator code is used only to group annotation sets
  (sampling weights, per-set instance maps, diagnostics), as the metric itself does.
- The public MAGFiLO v1.0 Kaggle dataset contains 139 of the 180 test images with labels. **It was never
  downloaded or used**, not even as unlabelled data. No other external images were used.
- No metric exploits: predicted masks are disjoint, with no duplicate or overlapping masks. No leaderboard
  tuning: the only submissions that were not model candidates were the one-bit mask-width probe of §4.4,
  which was rejected.
- Pretrained weights: ImageNet (ResNet-34, RegNetY-040, EfficientNetV2-S) and DINOv3 (Meta, DINOv3 License).
  The fine-tuned DINOv3 checkpoints are derivatives, so the public repository must ship the DINOv3 licence
  text next to them **[add the licence file before release]**.

## 7. Efficiency (one AMD Instinct MI300X, 192 GB)

| stage | cost |
|---|---|
| training, 5 folds x 60 epochs: M3 / M5 / M4 | 1.3 h (3 folds in parallel) / 5.1 h / 6.0 h (peak 38 / 82 / 81 GB) |
| inference, 8 TTA views, per image and fold model: M3 / M5 / M4 | 0.46 s / 0.70 s / 0.72 s |
| post-processing + selection for 180 test images | 7 s on 12 CPU workers |
| complete v3 run (ablations, 3 members, maps, Stage C) | 14.2 h |
| v1 training (3 members x 5 folds) | ~5 GPU-hours |

## 8. Reproducibility

- Environment: `SETUP.md` (ROCm 7.2 PyTorch, pinned `requirements.txt`).
- v1: `scripts/run_pipeline.sh`. v2: `scripts/run_v2.sh`. v3: `scripts/run_all_v3.sh` (resumable; every
  candidate with its nested PQ and submission file is listed in `work/v3_state/candidates.tsv`).
- Final submission: `NAMES="r34_2048 effv2s_1536 m3_r34_1536_v3 m4_vitb_1536_v3 m5_cnxs_1536_v3" WEIGHTS="1 1 0 0 1"
  scripts/run_v3.sh stagec` (written by `scripts/run_all_v3.sh`; log
  `logs/final_v3_stagec_r34_2048+effv2s_1536+m5_cnxs_1536_v3_w1-1-1.txt`). Zero-weight members are skipped.
- Reproduction from the trained weights without training: `submission_package_v3/notebook/reproduce_v3.ipynb`
  (SETUP.md §8). The M5 weights are attached to the GitHub release `v3-final`; the linker and the scorer are in
  `submission_package_v3/release/stagec_final/`.
- Design and compliance decisions with dates: `DECISIONS.md`. v3 runbook: `V3.md`.
- Unit tests: `tests/test_v3.py`. Stage C smoke test: `tests/smoke_stagec_v3.py`.

## 9. Limitations

The metric rewards agreement with individual annotators who disagree with each other (PQ 0.34 among
themselves), so out-of-fold gains below ~0.01 are invisible on a two-decimal leaderboard of 90 images. The
remaining error is dominated by boundary disagreement and by faint, small or near-limb filaments, which the
annotators themselves label inconsistently. A test set labelled by different annotators than the training set,
which is the most likely explanation of the leaderboard gap, cannot be corrected without information we do not use.
