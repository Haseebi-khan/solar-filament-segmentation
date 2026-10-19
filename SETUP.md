# Setup and usage

How to install the environment, download the competition data, reproduce the final submission from the trained
weights, or retrain everything.

- [1. Requirements](#1-requirements)
- [2. Get the code and the weights](#2-get-the-code-and-the-weights)
- [3. Python environment](#3-python-environment)
- [4. Kaggle API token](#4-kaggle-api-token)
- [5. Download the dataset](#5-download-the-dataset)
- [6. Train the final system](#6-train-the-final-system)
- [7. v1 baseline, step by step](#7-v1-baseline-step-by-step)
- [8. Reproduce the final submission from the weights](#8-reproduce-the-final-submission-from-the-weights)
- [9. Submit to Kaggle](#9-submit-to-kaggle)
- [10. Notebooks](#10-notebooks)
- [11. Repository layout](#11-repository-layout)
- [12. Troubleshooting](#12-troubleshooting)

## 1. Requirements

| | tested | minimum |
|---|---|---|
| GPU | 1x AMD Instinct MI300X (192 GB, ROCm) | ~24 GB for 1024 px; ~48 GB for 1536 px at batch 8 (or lower `--bs`); M5 training peaks at 82 GB at batch 8 |
| CPU / RAM | 20 cores / 235 GB | 8 cores / 32 GB (64 GB for the 2048 px members) |
| Disk | | ~60 GB free: data 1.5 GB, cache ~25 GB, probability maps ~12 GB per member, weights ~3 GB |
| Software | Ubuntu 24.04, Python 3.12 | Python 3.10+ |

NVIDIA GPUs work too; only the PyTorch install command differs (section 3).

## 2. Get the code and the weights

```bash
git clone https://github.com/umerkang66/solar-filament-segmentation.git
cd solar-filament-segmentation
```

All commands below run from the repository root.

The checkpoints of every member under 100 MB are tracked in git as ordinary files (`work/models/<member>/fold<k>.pt`,
no Git LFS needed). This includes `r34_2048` and `effv2s_1536` of the final blend. The DINOv3 ConvNeXt-S member
(`m5_cnxs_1536_v3`, 5 x 240 MB) is larger than GitHub's file limit. Download it from the release `v3-final` and
unpack it into `work/models/`:

```bash
# m5_cnxs_1536_v3_weights.zip from https://github.com/umerkang66/solar-filament-segmentation/releases/tag/v3-final
sha256sum -c m5_cnxs_1536_v3_weights.zip.sha256
(cd work/models && python3 -m zipfile -e ../../m5_cnxs_1536_v3_weights.zip .)
ls work/models/m5_cnxs_1536_v3/          # fold0.pt ... fold4.pt, LICENSE_DINOv3.md
```

The DINOv3 checkpoints are covered by the DINOv3 License (`submission_package_v3/release/LICENSE_DINOv3.md`).

## 3. Python environment

```bash
python3 -m venv .venv          # Ubuntu: sudo apt-get install -y python3.12-venv if this fails
.venv/bin/pip install --upgrade pip
```

AMD GPU (ROCm):

```bash
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm7.2
```

NVIDIA GPU (CUDA): install the CUDA build of PyTorch first, then the rest:

```bash
.venv/bin/pip install torch==2.14.1 torchvision==0.29.1
grep -v -E '^torch(vision)?==' requirements.txt > /tmp/req.txt
.venv/bin/pip install -r /tmp/req.txt
```

Check that the GPU is visible:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

(On ROCm PyTorch still uses the `torch.cuda` API.) The DINOv3 encoder weights for training M5 are downloaded once by
timm from the Hugging Face hub (`convnext_small.dinov3_lvd1689m`); inference from our checkpoints does not need them.

## 4. Kaggle API token

1. Sign in at kaggle.com, open the competition page
   <https://www.kaggle.com/competitions/filament-segmentation-2026> and accept the rules.
   Downloads fail with HTTP 403 until you do.
2. Kaggle → *Settings* → *API* → *Create New Token*.
3. Store the token so only you can read it. Use one of:

```bash
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle

# new-style token (starts with KGAT_)
printf '%s' 'KGAT_xxxxxxxxxxxxxxxx' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

# or the legacy kaggle.json file
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
```

Never commit the token; `.gitignore` excludes `kaggle.json` and `access_token`.

## 5. Download the dataset

```bash
mkdir -p data
.venv/bin/kaggle competitions download filament-segmentation-2026 -p data
.venv/bin/python -m zipfile -e data/filament-segmentation-2026.zip data/
```

Expected layout (707 training images, 180 test images):

```
data/MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/                                   707 x 2048x2048 JPEG
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json   COCO annotations
└── test/
    └── test_images/                                    180 x 2048x2048 JPEG
```

The code finds the data and writes its outputs through two environment variables:

| variable | default | meaning |
|---|---|---|
| `FILSEG_DATA` | `/root/kaggle/data/MAGFiLO_1.0_Kaggle_2026` | dataset folder above |
| `FILSEG_WORK` | `/root/kaggle/work` | cache, checkpoints, predictions, selectors |

If your clone is not at `/root/kaggle`, set both (e.g. in `~/.bashrc`):

```bash
export FILSEG_DATA=$PWD/data/MAGFiLO_1.0_Kaggle_2026
export FILSEG_WORK=$PWD/work
```

## 6. Train the final system

The final submission blends `r34_2048`, `effv2s_1536` and `m5_cnxs_1536_v3` (weights 1 1 1) and selects instances
with Stage C. Measured on one MI300X: ~5.6 h for the two v2 members (2 folds at a time), 5.1 h for M5, ~1 h of
inference and ~0.5 h of CPU post-processing.

```bash
# data: decoded images, folds (seed 42), soft targets at 1024/1536/2048 px, skeleton/core/instance targets
.venv/bin/python src/prepare_data.py --sizes 1024 1536
.venv/bin/python src/prepare_targets_v2.py --sizes 2048
.venv/bin/python src/prepare_targets_v3.py --sizes 1024 1536 2048 --inst-size 1024

# members, 5 folds each
scripts/run_v2.sh train                     # r34_2048 (ResNet-34 @2048) and effv2s_1536 (EfficientNetV2-S @1536), 80 epochs
for k in 0 1 2 3 4; do                      # M5: DINOv3 ConvNeXt-S U-Net @1536, v3 loss (flags chosen by the ablation)
  .venv/bin/python src/train_v3.py --config configs/m5_cnxs_1536_v3.yaml --fold $k \
      --region jml --w-blob 0.5 --w-sr 0 --aux-heads --deep-sup --epochs 60
done

# out-of-fold + test probability maps (8 TTA views)
scripts/run_v2.sh predict
.venv/bin/python src/predict_v3.py --name m5_cnxs_1536_v3 --tta 8

# Stage C: fragment linker, candidate pool, nested stacked scorer, out-of-fold score, test CSV, rubric metrics
NAMES="r34_2048 effv2s_1536 m3_r34_1536_v3 m4_vitb_1536_v3 m5_cnxs_1536_v3" WEIGHTS="1 1 0 0 1" scripts/run_v3.sh stagec
```

Members with weight 0 are skipped (their maps are not needed); the five names only fix the configuration the
scorer is checked against. Output: `submissions/submission_v3_r34_2048+effv2s_1536+m5_cnxs_1536_v3_w1-1-1_linker.csv`
and its log `logs/final_v3_stagec_r34_2048+effv2s_1536+m5_cnxs_1536_v3_w1-1-1.txt` (nested out-of-fold PQ 0.4477).

The complete v3 run that produced it, including the loss ablation and the M3/M4 members, is
`scripts/run_all_v3.sh` (runbook: [V3.md](V3.md)). Every candidate it wrote is listed with its nested PQ in
`work/v3_state/candidates.tsv`.

## 7. v1 baseline, step by step

The v1 system (3 U-Nets + rescorer, nested PQ 0.4381) runs end to end with `scripts/run_pipeline.sh` (~5-6 h):

```bash
mkdir -p logs
nohup scripts/run_pipeline.sh > logs/pipeline.out 2>&1 &
tail -f logs/pipeline.out            # Ctrl+C stops watching, not the run
```

Or stage by stage:

**7.1 Prepare data** (decodes all images once, builds the soft targets):

```bash
.venv/bin/python src/prepare_data.py --sizes 1024 1536
```

**7.2 Train** (one fold, or all folds of a member with `PARALLEL` folds at a time):

```bash
.venv/bin/python src/train.py --name r34_1024 --fold 0 --encoder resnet34 --size 1024 --epochs 60
PARALLEL=2 scripts/train_folds.sh r34_1024    "0 1 2 3 4" --encoder resnet34       --size 1024 --epochs 60
PARALLEL=2 scripts/train_folds.sh rgy040_1024 "0 1 2 3 4" --encoder tu-regnety_040 --size 1024 --epochs 80
PARALLEL=2 scripts/train_folds.sh r34_1536    "0 1 2 3 4" --encoder resnet34       --size 1536 --epochs 80
```

Useful `train.py` options: `--bs`, `--lr`, `--crop`, `--no-rotate`, `--dice-weight`, `--eval-every`, `--fold -1`
(all images).

**7.3 Predict** (out-of-fold maps for the training images, test maps for every fold, 8x TTA):

```bash
for m in r34_1024 rgy040_1024 r34_1536; do
  .venv/bin/python src/predict.py --name $m --folds 0 1 2 3 4 --tta 8
done
```

**7.4 Tune post-processing** (optional, diagnostic; exact competition metric):

```bash
.venv/bin/python src/tune_postprocess.py --names r34_1024 rgy040_1024 r34_1536 --weights 1 1 2 \
  --seeds 0.5 0.6 0.7 --masks 0.5 0.6 --links 0 8 --closes -1
```

**7.5 Blend weights, rescorer, submission:**

```bash
scripts/final_ensemble.sh                         # compare blend weights
WEIGHTS="1 1 2" scripts/final_ensemble.sh         # rescorer + OOF score + submissions/submission.csv
```

`final_ensemble.sh` calls `src/train_rescorer.py` (prints its nested-CV PQ, the honest estimate) and
`src/make_submission.py`.

## 8. Reproduce the final submission from the weights

No training; about 40 minutes on one MI300X. Needs the data (section 5) and the M5 weights (section 2). The notebook
`submission_package_v3/notebook/reproduce_v3.ipynb` runs these steps in a separate work directory
(`work_reproduce/`) and compares the result with the submitted file:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace submission_package_v3/notebook/reproduce_v3.ipynb
```

Equivalent commands (writing to `$FILSEG_WORK`):

```bash
.venv/bin/python src/prepare_data.py
.venv/bin/python src/predict.py --name r34_2048 --tta 8
.venv/bin/python src/predict.py --name effv2s_1536 --tta 8
.venv/bin/python src/predict_v3.py --name m5_cnxs_1536_v3 --tta 8
P=submission_package_v3/release/stagec_final
POOL=$FILSEG_WORK/pools_v3/r34_2048+effv2s_1536+m5_cnxs_1536_v3_w1-1-1_linker
.venv/bin/python src/build_pool_v3.py --names r34_2048 effv2s_1536 m3_r34_1536_v3 m4_vitb_1536_v3 m5_cnxs_1536_v3 \
  --weights 1 1 0 0 1 --split test --linker $P/linker.pkl --link-taus 0.4 0.6 --out $POOL
# point the scorer config at this pool and at the shipped scorer, then select and write the CSV
.venv/bin/python -c "import json,sys; c=json.load(open('$P/scorer_config.json')); c['pool']='$POOL'; c['scorer']='$P/scorer.pkl'; json.dump(c, open('/tmp/scorer_config.json','w'))"
.venv/bin/python src/make_submission_v3.py --config /tmp/scorer_config.json --out submissions/submission_v3_reproduced.csv
```

GPU inference in bf16 is not bit-exact, so a few boundary pixels may differ from the submitted file. Our own run of the
notebook (40 min on one MI300X) reproduced 1,061 of the 1,067 submitted masks bit-identically; 1,066 match with IoU >= 0.999 and one differs by 74 px (IoU 0.95), as expected from non-deterministic bf16 GPU inference.

## 9. Submit to Kaggle

Every submission writer validates the file: unique `filament_id`s, valid 2048x2048 RLEs, no empty and no
overlapping masks.

```bash
.venv/bin/kaggle competitions submit filament-segmentation-2026 -f submission_package_v3/kaggle/submission_v3_final.csv -m "v3 final"
.venv/bin/kaggle competitions submissions filament-segmentation-2026      # check the score
```

The competition allows 5 submissions per day; pick up to 2 final submissions on the website.

## 10. Notebooks

- `submission_package_v3/notebook/reproduce_v3.ipynb`: reproduces the final CSV from the weights (section 8).
  Rebuild it with `.venv/bin/python tools/build_notebook_v3.py`.
- `notebooks/pipeline.ipynb` (optional, v1): illustrates every stage of the v1 pipeline (annotator disagreement,
  the metric, soft targets, training curves, instance extraction, the submission). Build and execute it after
  section 7:

  ```bash
  mkdir -p notebooks
  .venv/bin/python tools/build_notebook.py r34_1024,rgy040_1024,r34_1536 \
    work/rescorers/r34_1024+rgy040_1024+r34_1536_config.json notebooks/pipeline.ipynb
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/pipeline.ipynb
  ```

## 11. Repository layout

```
src/
  filseg/            shared package
    data.py            COCO loading, polygon rasterisation, image cache, folds
    metric.py          exact replica of the organisers' PQ + additive per-instance contributions
    augment*.py        GPU augmentation (affine + photometric)
    model.py           v1 U-Net construction / checkpoint loading
    models_v3.py       v3 networks (smp U-Net, timm ConvNeXt U-Net, DINOv3 ViT + DPT), heads, detail stem
    losses_v3.py       BCE, JML1 / DML1, blob loss, skeleton/core losses
    targets_v3.py      soft skeleton, soft core, per-set instance maps
    infer*.py          full-disk inference with dihedral TTA
    postprocess.py     probability map -> disjoint instances, instance features
    hypotheses_v2.py   extent levels per filament (v2 selector)
    pool_v3.py linker_v3.py select_v3.py masks_v3.py taxonomy_v3.py   Stage C and diagnostics
  prepare_data.py prepare_targets_v2.py prepare_targets_v3.py         data
  train.py (v1/v2 members)  train_v3.py (v3 members)  train_rater_v3.py (experimental, not used)
  predict.py predict_v3.py                                             probability maps
  tune_postprocess.py blend_search_v3.py blend_search_tuned_v3.py      thresholds and blend weights
  train_rescorer.py train_selector_v2.py                               v1 rescorer, v2 extent selector
  build_pool_v3.py train_linker_v3.py train_scorer_v3.py               Stage C
  make_submission.py make_submission_v2.py make_submission_v3.py       checked submission CSVs
  evaluate_v2.py diagnose_v3.py ceilings_v3.py compare_v3.py           rubric metrics, error taxonomy, ceilings, bootstrap
  gong_*_v3.py pseudo_label_v3.py                                      semi-supervised stage (not run)
scripts/             run_pipeline.sh (v1), run_v2.sh / run_all_v2.sh (v2), run_v3.sh / run_all_v3.sh (v3), helpers
configs/             v3 member configs (YAML)
tests/               tests/test_v3.py (unit), tests/smoke_stagec_v3.py
tools/               build_notebook.py (v1), build_notebook_v3.py, morph_submission_v2.py
work/                weights (<100 MB), logs, selectors, out-of-fold CSVs, rubric figures (eval_v*/), diagnostics (diag_v3/)
submission_package_v3/   CSV, LaTeX report, reproduction notebook, release tools, final linker + scorer
README.md  REPORT.md  SETUP.md (this file)  DECISIONS.md  V3.md  requirements.txt (pinned)
```

## 12. Troubleshooting

| symptom | fix |
|---|---|
| `403 Forbidden` when downloading | accept the competition rules on the website (section 4) |
| `FileNotFoundError: .../train_soft_1536.npy` | run `src/prepare_data.py --sizes 1024 1536` |
| `FileNotFoundError: .../m5_cnxs_1536_v3/fold0.pt` | download and unpack the M5 release zip (section 2) |
| `... pool was built with names=..., the scorer with ...` | build the pool with exactly the five names and weights `1 1 0 0 1` (section 8) |
| first epoch takes 10+ minutes on AMD | one-time MIOpen kernel compilation for a new resolution; later epochs and runs are fast (kernels are cached in `~/.cache/miopen`) |
| training is extremely slow on AMD | keep `torch.backends.cudnn.benchmark = False` (as in the training scripts); MIOpen's exhaustive search takes minutes per input shape |
| out of GPU memory | lower `--bs` (e.g. 4), or train one fold at a time (`PARALLEL=1`) |
| data not found | set `FILSEG_DATA` / `FILSEG_WORK` (section 5) |
| two jobs barely faster than one | expected: one job nearly saturates the GPU (~18% extra throughput from a second) |
