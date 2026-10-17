# Solar Filament Segmentation Challenge 2026: U-Net ensemble with PQ-aware instance extraction

Instance segmentation of solar filaments in GONG H-alpha full-disk images (MAGFiLO),
scored by Panoptic Quality (PQ) against every annotator separately.

## Key observations

* **Annotators disagree strongly.** 296 of the 707 training images have 2-3 independent
  annotation sets. Scoring one annotator against another with the organisers' metric
  gives **PQ = 0.340**, so honest models are bounded by human consistency, not by pixel accuracy.
* **The metric scores each annotation set separately** against all predictions of the
  image (`<annotator>-<image>` grouping). An image with three annotators counts three times.
* **With disjoint predicted masks PQ is additive over instances.** A GT filament can be hit
  (IoU > 0.5) by at most one prediction, so for any subset S of predicted instances

      PQ(S) = sum_{i in S} a_i / (0.5 * N_gt + sum_{i in S} d_i)

  where a_i is the sum of instance i's TP IoUs over the image's annotation sets and
  d_i = m_i / 2 for an image with m_i sets. An instance therefore improves PQ iff its
  expected per-annotator TP-IoU exceeds PQ / 2. This gives an exact, instant objective for
  tuning thresholds and for the second-stage instance rescorer.

## Pipeline

1. `src/prepare_data.py` decodes the images once and builds **soft targets**: for every
   pixel, the fraction of annotators whose polygon covers it (rasterised with pycocotools,
   exactly as the ground truth is).
2. `src/train.py` trains a U-Net (segmentation_models_pytorch, 1-channel input, ImageNet
   encoder) on the full disk with BCE + soft-Dice against the soft targets. Images are sampled
   in proportion to their number of annotation sets, mirroring the metric's weighting.
   Augmentation runs on the GPU: arbitrary rotation, flips, scale, shift, gamma, contrast,
   blur and noise. Weights are averaged with an EMA. Training uses 5 image-level folds.
3. `src/predict.py` writes out-of-fold and test probability maps with 8-fold dihedral TTA.
4. `src/tune_postprocess.py` converts probabilities to disjoint instances: connected
   components above `t_seed`, fragments within `link` px grouped into one filament, extent
   at `t_mask`. It then searches area/score filters with the exact PQ formula above.
5. `src/train_rescorer.py` (optional) trains a LightGBM model on out-of-fold candidate
   instances to predict each one's expected per-annotator TP-IoU, and keeps instances above
   a tuned threshold (about PQ/2). Its gain is measured with nested CV.
6. `src/make_submission.py` writes the CSV (`filament_id`, `segmentation_rle`) and checks
   it: unique ids, valid 2048x2048 RLEs, no overlapping masks. With `--oof` it also scores an
   out-of-fold CSV against a ground-truth CSV in the organisers' format.

`src/filseg/metric.py` reproduces the organisers' self-evaluation notebook exactly
(verified pair by pair against its dense implementation).

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm7.2
export FILSEG_DATA=/path/to/MAGFiLO_1.0_Kaggle_2026 FILSEG_WORK=$PWD/work
.venv/bin/python src/prepare_data.py
scripts/run_pipeline.sh            # trains all folds/models, tunes, writes submissions/
```
