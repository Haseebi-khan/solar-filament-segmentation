# Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments in GONG H-alpha full-disk images (MAGFiLO), scored by Panoptic Quality
(PQ) against every annotator separately.

**Final submission:** [`submission_package_v3/kaggle/submission_v3_final.csv`](submission_package_v3/kaggle/submission_v3_final.csv),
nested out-of-fold PQ **0.4477** on the 707 training images, public leaderboard **0.39** (Kaggle team "Umer Gulzar").

| you want to | go to |
|---|---|
| read the method and the results | [REPORT.md](REPORT.md); the submitted 4-page report (organisers' ACM template) is [submission_package_v3/report/](submission_package_v3/report/) |
| reproduce the final CSV from the trained weights (~40 min on one GPU; verified, see below) | [submission_package_v3/notebook/reproduce_v3.ipynb](submission_package_v3/notebook/reproduce_v3.ipynb), [SETUP.md §8](SETUP.md#8-reproduce-the-final-submission-from-the-weights) |
| retrain everything | [SETUP.md §6](SETUP.md#6-train-the-final-system) |
| check data use and design decisions | [DECISIONS.md](DECISIONS.md) |
| see every candidate and its score | [`work/v3_state/candidates.tsv`](work/v3_state/candidates.tsv), `logs/final_*.txt` |

## Key observations

* **Annotators disagree strongly.** 296 of the 707 training images have 2-3 independent annotation sets. Scoring
  one annotator against another with the organisers' metric gives **PQ = 0.340**. An oracle that knows every label
  still reaches only 0.666 with one set of masks on these images, because the annotators contradict each other.
* **The metric scores each annotation set separately** against all predictions of the image
  (`<annotator>-<image>` grouping), so an image with three annotators counts three times.
* **With disjoint predicted masks, PQ is additive over instances.** A GT filament can be hit (IoU > 0.5) by at most
  one prediction. Hence for any subset S of predicted instances

      PQ(S) = sum_{i in S} a_i / (0.5 * N_gt + sum_{i in S} d_i)

  where a_i is the sum of instance i's TP IoUs over the image's annotation sets and d_i = m_i / 2 for an image with
  m_i sets. An instance improves PQ if and only if its expected per-annotator TP IoU exceeds PQ / 2. This gives an
  exact, instant objective for tuning thresholds and for training instance scorers.

## Final system

1. **Soft targets.** For every pixel, the fraction of annotators whose polygon covers it, rasterised with
   pycocotools exactly like the ground truth (`src/prepare_data.py`, `src/prepare_targets_v3.py`).
2. **Three members, 5 image-level folds each, full-disk input:**

   | member | network | input | loss |
   |---|---|---|---|
   | `r34_2048` | U-Net, ResNet-34 (ImageNet) | 2048 px | BCE + soft Dice (`src/train.py`) |
   | `effv2s_1536` | U-Net, EfficientNetV2-S (ImageNet) | 1536 px | BCE + soft Dice (`src/train.py`) |
   | `m5_cnxs_1536_v3` | U-Net, DINOv3 ConvNeXt-S | 1536 px | BCE + Jaccard metric loss + blob loss + skeleton/core heads + deep supervision (`src/train_v3.py`) |

3. **Inference.** 8 dihedral TTA views, mean over the fold models, then an equal-weight blend
   (`src/predict.py`, `src/predict_v3.py`).
4. **Stage C.** A pool of instance hypotheses (threshold families and a learned fragment linker), a stacked LightGBM
   scorer that predicts each candidate's expected TP IoU, and an overlap-aware greedy selection that yields
   disjoint masks (`src/build_pool_v3.py`, `src/train_linker_v3.py`, `src/train_scorer_v3.py`,
   `src/make_submission_v3.py`). Every learned stage is trained with nested cross-validation.

`src/filseg/metric.py` reproduces the organisers' self-evaluation notebook exactly (verified pair by pair).

## Results (nested out-of-fold PQ, organisers' metric)

| system | PQ | public LB |
|---|---|---|
| human annotator vs annotator | 0.340 | |
| v1: 3 U-Nets + LightGBM rescorer | 0.4381 | 0.38 |
| v2: r34_2048 + effv2s_1536 + extent selector | 0.4438 | 0.38 |
| v2 selector + M3 (ResNet-34, v3 loss) | 0.4449 | 0.38 |
| **final: Stage C on r34_2048 + effv2s_1536 + M5** | **0.4477** | **0.39** |

**Reproduced:** running the notebook on our server rebuilt 1,061 of the 1,067 submitted masks bit-identically; 1,066 match with IoU >= 0.999 and one differs by 74 px (IoU 0.95), as expected from non-deterministic bf16 GPU inference (`submissions/submission_v3_reproduced.csv`).

REPORT.md explains the ~0.06 gap between out-of-fold and public scores. It is not chance and not mask width; the
most likely cause is an annotator shift.

## Data use and compliance

* At inference the models see only the test images. Of the ground-truth fields, only `segmentation` is read.
* The public MAGFiLO v1.0 Kaggle dataset, which contains labelled test images, was never downloaded or used. No
  other external data were used.
* Masks are disjoint (checked by every submission writer); there are no duplicate masks and no leaderboard tuning.
* Competition data: CC BY-NC 4.0, not included (download instructions in SETUP.md). Pretrained weights: ImageNet
  (timm / segmentation_models_pytorch) and DINOv3 (Meta). The fine-tuned DINOv3 checkpoints are covered by the
  [DINOv3 License](LICENSE_DINOv3.md) (Meta, official text). It applies to the DINOv3-based checkpoints (M4, M5)
  and is also included in the M5 release zip, as the licence requires for redistributed derivatives.

## Weights

`work/models/<member>/fold<k>.pt` holds the checkpoints of every member under 100 MB, including `r34_2048` and
`effv2s_1536`. The DINOv3 members are larger than GitHub's file limit. The M5 checkpoints used by the final
submission are attached to the GitHub release `v3-final` as `m5_cnxs_1536_v3_weights.zip`. The linker and the
scorer of the final Stage C are in [submission_package_v3/release/stagec_final/](submission_package_v3/release/stagec_final/).

## Repository map

```
src/                 training, inference, post-processing, Stage C (filseg/ = shared package)
configs/             YAML configs of the v3 members
scripts/             pipelines: run_pipeline.sh (v1), run_v2.sh / run_all_v2.sh (v2), run_v3.sh / run_all_v3.sh (v3)
tests/               unit tests (tests/test_v3.py) and a Stage C smoke test
tools/               notebook builders, mask-width probe tool
work/                weights (<100 MB), training logs, selectors, out-of-fold CSVs, rubric figures, diagnostics
logs/final_*.txt     final scores of every stage
submissions/         every submission file (the final one is also in submission_package_v3/kaggle/)
submission_package_v3/   everything handed in: CSV, LaTeX report, reproduction notebook, release tools
REPORT.md  SETUP.md  DECISIONS.md  V3.md (v3 runbook)
LICENSE_DINOv3.md    Meta's DINOv3 License (DINOv3-based checkpoints)
```

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm7.2
export FILSEG_DATA=/path/to/MAGFiLO_1.0_Kaggle_2026 FILSEG_WORK=$PWD/work
# reproduce the final CSV from the weights (needs the M5 release zip, see SETUP.md §8)
.venv/bin/jupyter nbconvert --to notebook --execute submission_package_v3/notebook/reproduce_v3.ipynb
```

Full instructions (environment, Kaggle token, dataset, training, troubleshooting): [SETUP.md](SETUP.md).
