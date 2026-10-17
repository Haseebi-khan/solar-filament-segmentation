# Solar Filament Instance Segmentation with Soft Multi-Annotator Targets and PQ-Additive Instance Selection

*Technical report draft for the Solar Filament Segmentation Challenge 2026 (port to the Overleaf template; 4 pages).*

## 1. Problem and data

The task is instance segmentation of solar filaments in 2048x2048 GONG H-alpha full-disk images.
The training set (MAGFiLO) has 707 images with 1,154 annotation sets: 411 images have one
annotator, 145 have two, and 151 have three. There are 8,199 filament polygons in total. Filaments are thin
and elongated (median area 1,228 px; 5th percentile 317 px).

Submissions are scored with Panoptic Quality (PQ). Each annotation set is compared separately
with all predicted filaments of its image, and TP, FP and FN counts are pooled over all sets.

**Human agreement.** We scored every annotator against every other annotator of the same image with
the organisers' metric. The result is **PQ = 0.340** (SQ 0.634, RQ 0.536). Annotators disagree on
which faint structures are filaments, on their extent, and on how fragments are grouped. This
ceiling shapes every design decision below.

## 2. Metric analysis: PQ is additive over disjoint instances

If predicted masks are disjoint, a GT filament can have IoU > 0.5 with at most one prediction, and a
prediction with at most one GT of a set. For any subset S of predicted instances:

    PQ(S) = sum_{i in S} a_i / (0.5 * N_gt + sum_{i in S} d_i)

Here a_i is the sum of instance i's TP-IoUs over its image's m_i annotation sets, and d_i = m_i / 2.
Consequences:

1. Adding instance i raises PQ iff its expected per-annotator TP-IoU u_i = a_i / m_i exceeds PQ/2
   (about 0.21 here). This gives a principled keep-threshold for instances.
2. Threshold and filter search on out-of-fold data is exact and instant: instances are extracted
   once and every filter is then a sum.
3. The metric counts every IoU > 0.5 pair as a TP, so overlapping duplicate masks inflate it. On
   our out-of-fold predictions, submitting each mask twice raises PQ from 0.4385 to 0.4831.
   We submit strictly disjoint masks.

## 3. Method

**Targets.** Every polygon is rasterised with pycocotools, exactly like the ground truth. The
target pixel value is the fraction of the image's annotators that cover it (soft consensus).

**Networks.** U-Nets (segmentation_models_pytorch) with ImageNet encoders, a 1-channel input and
scSE attention in the decoder, applied to the full solar disk:

| member | encoder | input | epochs |
|---|---|---|---|
| r34_1024 | ResNet-34 | 1024x1024 | 60 |
| rgy040_1024 | RegNetY-040 | 1024x1024 | 80 |
| r34_1536 | ResNet-34 | 1536x1536 | 80 |

**Training.** BCE + soft Dice against the soft target; AdamW (lr 5e-4, wd 1e-4); 2-epoch warm-up
then cosine decay; batch 8; bf16; EMA of weights (0.998); gradient clipping at 2. Images are sampled
in proportion to their number of annotation sets, which matches the metric's weighting.
Augmentation runs on the GPU: arbitrary rotation, flips, scale 0.87-1.15, shift, gamma, contrast,
brightness, blur, Gaussian noise. The whole training set is kept on the GPU, so there is no data loader.

**Validation.** 5 image-level folds; all annotation sets of an image share a fold. Test images are
a random sample of the same observation pool (median gap to the nearest training image ~37 h, the
same as between training images), so this split mirrors the train/test relationship.

**Inference.** 8 dihedral TTA transforms; average over the 5 fold models, then a weighted blend of
the members. Probabilities are bilinearly upsampled to 2048x2048.

**Instance extraction.**
1. Seeds are the connected components of p > 0.6.
2. Seeds closer than ~16 px (dilation radius 8) form one filament; this fixes fragmentation, since
   annotators draw one polygon around a broken filament.
3. Extent is p > 0.6, and holes inside each instance are filled (GT polygons have no holes).
   Masks are disjoint by construction.

**Instance rescoring.** Candidates (area >= 100 px) get 26 features: probability statistics; core
fractions above 0.8 and 0.9; contrast against a 12-px ring; ring probability; shape (area,
skeleton length, width, elongation, fill ratio, number of linked parts); position (distance from
disk centre / radius); context (instances per image, rank, nearest-neighbour distance). A LightGBM
regressor predicts u_i, weighted by m_i. Instances with predicted u > t are kept; t is tuned on
out-of-fold data and lands at 0.215-0.22, matching the theoretical PQ/2.

## 4. Results (5-fold out-of-fold, 707 images, organisers' metric)

| configuration | PQ |
|---|---|
| annotator vs annotator (human agreement) | 0.340 |
| r34_1024, threshold 0.5, no TTA, no filtering (fold 0) | 0.380 |
| r34_1024 + TTA + tuned thresholds/filters | 0.4205 |
| + rescorer (nested CV) | 0.4272 |
| ensemble + rescorer (nested CV) | __ENSEMBLE__ |

Fold-0 ablations (tuned): fragment linking +0.008; hole filling +0.0003; closing radius 2-6
-0.003 to -0.016; multi-scale TTA +0.0006; per-instance adaptive extent +0.0013; 1536 px input
+0.009 over 1024 px; no-rotation augmentation +0.005 (single fold, inconclusive).
Performance is uniform across annotator count (0.44 / 0.45 / 0.44 for 1/2/3 sets), observatory
(0.43-0.45) and observing period.

## 5. Efficiency

On one AMD Instinct MI300X, training all members takes about 5 GPU-hours. Inference with 8x TTA
and post-processing takes about 1-2 s per image per member; the rescorer adds about 50 ms per image.

## 6. Reproducibility

`scripts/run_pipeline.sh` runs everything from the raw competition folder to `submissions/submission.csv`.
`notebooks/pipeline.ipynb` illustrates each stage. Package versions are pinned in `requirements.txt`.
