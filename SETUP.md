# Setup and usage

How to install the environment, download the competition data, reproduce the full pipeline,
or generate a submission from the released weights.

- [1. Requirements](#1-requirements)
- [2. Get the code](#2-get-the-code)
- [3. Python environment](#3-python-environment)
- [4. Kaggle API token](#4-kaggle-api-token)
- [5. Download the dataset](#5-download-the-dataset)
- [6. Run the full pipeline](#6-run-the-full-pipeline)
- [7. Run step by step](#7-run-step-by-step)
- [8. Submission from the released weights (no training)](#8-submission-from-the-released-weights-no-training)
- [9. Submit to Kaggle](#9-submit-to-kaggle)
- [10. Notebook](#10-notebook)
- [11. Repository layout](#11-repository-layout)
- [12. Troubleshooting](#12-troubleshooting)

## 1. Requirements

| | tested | minimum |
|---|---|---|
| GPU | 1x AMD Instinct MI300X (192 GB, ROCm) | any GPU with ~24 GB for 1024 px; ~48 GB for 1536 px at batch 8 (or lower `--bs`) |
| CPU / RAM | 20 cores / 235 GB | 8 cores / 32 GB |
| Disk | | ~35 GB free: data 1.5 GB, cache 6 GB, probability maps ~15 GB, weights ~2 GB |
| Software | Ubuntu 24.04, Python 3.12 | Python 3.10+ |

NVIDIA GPUs work too; only the PyTorch install command differs (section 3).

## 2. Get the code

The model weights are stored with Git LFS. Install it before cloning:

```bash
sudo apt-get install -y git-lfs
git lfs install
git clone https://github.com/<user>/solar-filament-segmentation-2026.git
cd solar-filament-segmentation-2026
```

All commands below run from the repository root.

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

(On ROCm PyTorch still uses the `torch.cuda` API.)

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
| `FILSEG_WORK` | `/root/kaggle/work` | cache, checkpoints, predictions, rescorers |

If your clone is not at `/root/kaggle`, set both (e.g. in `~/.bashrc`):

```bash
export FILSEG_DATA=$PWD/data/MAGFiLO_1.0_Kaggle_2026
export FILSEG_WORK=$PWD/work
```

## 6. Run the full pipeline

One command runs data preparation, 5-fold training of the three ensemble members, inference,
tuning, the rescorer and the submission. It takes about 5-6 hours on one MI300X.

```bash
mkdir -p logs
nohup scripts/run_pipeline.sh > logs/pipeline.out 2>&1 &
tail -f logs/pipeline.out            # Ctrl+C stops watching, not the run
```

Outputs:

| file | content |
|---|---|
| `submissions/submission.csv` | the submission |
| `logs/final_r34_1024+rgy040_1024+r34_1536.txt` | blend-weight comparison, nested-CV PQ of the rescorer, out-of-fold PQ of the CSV |
| `work/models/<member>/fold<k>.pt`, `fold<k>.log` | checkpoints and training logs |
| `work/rescorers/<members>.pkl`, `<members>_config.json` | rescorer and final post-processing configuration |

The pipeline blends the members with weights `1 1 2` (r34_1024, rgy040_1024, r34_1536). If another
line of the blend-weight comparison scores higher, rebuild only the final stage with those weights
(a few minutes):

```bash
WEIGHTS="1 1 1" scripts/final_ensemble.sh
```

Progress check and stopping:

```bash
grep -h "^epoch" work/models/*/fold*.log | tail -3      # latest epochs
ps -eo pid,args | grep -E "[r]un_pipeline|[t]rain_folds|[s]rc/train.py"
kill <PID> ...                                          # stop the listed jobs
```

## 7. Run step by step

Equivalent to `scripts/run_pipeline.sh`, useful to rerun a single stage.

**7.1 Prepare data.** Decodes all images once and builds the soft multi-annotator targets
(about 20 s). Use `--sizes` for every training resolution you plan to use.

```bash
.venv/bin/python src/prepare_data.py --sizes 1024 1536
```

**7.2 Train.** One fold:

```bash
.venv/bin/python src/train.py --name r34_1024 --fold 0 --encoder resnet34 --size 1024 --epochs 60
```

All folds of a member (`PARALLEL` = folds trained at the same time):

```bash
PARALLEL=2 scripts/train_folds.sh r34_1024    "0 1 2 3 4" --encoder resnet34       --size 1024 --epochs 60
PARALLEL=2 scripts/train_folds.sh rgy040_1024 "0 1 2 3 4" --encoder tu-regnety_040 --size 1024 --epochs 80
PARALLEL=2 scripts/train_folds.sh r34_1536    "0 1 2 3 4" --encoder resnet34       --size 1536 --epochs 80
```

Useful `train.py` options: `--bs` (batch size, default 8), `--lr`, `--crop` (train on random crops),
`--no-rotate`, `--dice-weight`, `--eval-every`, `--fold -1` (train on all images).

Approximate time per fold on one MI300X, one job at a time: r34_1024 10 min, rgy040_1024 17 min,
r34_1536 25 min.

**7.3 Predict.** Writes out-of-fold maps for the training images and test maps for every fold, with
8x flip/rotation TTA:

```bash
for m in r34_1024 rgy040_1024 r34_1536; do
  .venv/bin/python src/predict.py --name $m --folds 0 1 2 3 4 --tta 8
done
```

**7.4 Tune post-processing (optional, diagnostic).** Grid search over seed/extent thresholds,
fragment linking and area/score filters with the exact competition metric:

```bash
.venv/bin/python src/tune_postprocess.py --names r34_1024 rgy040_1024 r34_1536 --weights 1 1 2 \
  --seeds 0.5 0.6 0.7 --masks 0.5 0.6 --links 0 8 --closes -1
```

**7.5 Blend weights, rescorer, submission.**

```bash
scripts/final_ensemble.sh                         # compare blend weights
WEIGHTS="1 1 2" scripts/final_ensemble.sh         # rescorer + OOF score + submissions/submission.csv
```

`final_ensemble.sh` calls `src/train_rescorer.py` (instance rescorer; prints its nested-CV PQ) and
`src/make_submission.py`. The latter can also be used on its own:

```bash
.venv/bin/python src/make_submission.py --names r34_1024 rgy040_1024 r34_1536 --weights 1 1 2 \
  --config work/rescorers/r34_1024+rgy040_1024+r34_1536_config.json --out submissions/submission.csv

# same pipeline on the out-of-fold training maps, scored with the competition metric
.venv/bin/python src/make_submission.py --names r34_1024 rgy040_1024 r34_1536 --weights 1 1 2 \
  --config work/rescorers/r34_1024+rgy040_1024+r34_1536_config.json --oof --out work/oof_submission.csv
```

The `--oof` score uses the rescorer that was fit on those same instances, so it is optimistic. The
honest estimate is the nested-CV PQ printed by `train_rescorer.py`.

## 8. Submission from the released weights (no training)

The repository contains the trained checkpoints (`work/models/`) and rescorer (`work/rescorers/`).
To regenerate the submission without training (about 15 minutes):

```bash
git lfs pull                                           # make sure the weights are downloaded, not pointers
.venv/bin/python src/prepare_data.py --sizes 1024 1536
for m in r34_1024 rgy040_1024 r34_1536; do
  .venv/bin/python src/predict.py --name $m --folds 0 1 2 3 4 --tta 8
done
.venv/bin/python src/make_submission.py --names r34_1024 rgy040_1024 r34_1536 --weights 1 1 2 \
  --config work/rescorers/r34_1024+rgy040_1024+r34_1536_config.json --out submissions/submission.csv
```

Use the weights recorded in the `weights` field of the config JSON.

## 9. Submit to Kaggle

`make_submission.py` already validates the file: unique `filament_id`s, valid 2048x2048 RLEs,
no empty and no overlapping masks.

```bash
.venv/bin/kaggle competitions submit filament-segmentation-2026 -f submissions/submission.csv -m "U-Net ensemble + rescorer"
.venv/bin/kaggle competitions submissions filament-segmentation-2026      # check the score
```

The competition allows 5 submissions per day; pick up to 2 final submissions on the website.

## 10. Notebook

`notebooks/pipeline.ipynb` illustrates every stage: annotator disagreement, the metric, soft targets,
training curves, instance extraction and the submission. It reads the artefacts written by the
pipeline. Build and execute it after section 6 or 8:

```bash
mkdir -p notebooks
.venv/bin/python tools/build_notebook.py r34_1024,rgy040_1024,r34_1536 \
  work/rescorers/r34_1024+rgy040_1024+r34_1536_config.json notebooks/pipeline.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/pipeline.ipynb
```

## 11. Repository layout

```
src/
  filseg/
    data.py          COCO loading, polygon rasterisation, image cache, folds
    metric.py        exact replica of the organisers' PQ + additive per-instance contributions
    augment.py       GPU augmentation (affine + photometric)
    model.py         U-Net construction / checkpoint loading
    infer.py         full-disk inference with dihedral TTA
    postprocess.py   probability map -> disjoint instances, instance features
    rescore.py       LightGBM instance rescorer
  prepare_data.py    decode images, soft targets, metadata
  train.py           train one fold
  predict.py         out-of-fold and test probability maps
  tune_postprocess.py
  train_rescorer.py
  make_submission.py
scripts/
  run_pipeline.sh    end to end
  train_folds.sh     several folds of one member
  final_ensemble.sh  blend weights, rescorer, submission
tools/build_notebook.py
README.md            method overview      REPORT.md   technical report draft
requirements.txt     pinned versions      SETUP.md    this file
```

## 12. Troubleshooting

| symptom | fix |
|---|---|
| `403 Forbidden` when downloading | accept the competition rules on the website (section 4) |
| `FileNotFoundError: .../train_soft_1536.npy` | run `src/prepare_data.py --sizes 1024 1536` |
| first epoch takes 10+ minutes on AMD | one-time MIOpen kernel compilation for a new resolution; later epochs and runs are fast (kernels are cached in `~/.cache/miopen`) |
| training is extremely slow on AMD | keep `torch.backends.cudnn.benchmark = False` (as in `src/train.py`); MIOpen's exhaustive search takes minutes per input shape |
| out of GPU memory | lower `--bs` (e.g. 4), or train one fold at a time (`PARALLEL=1`) |
| `.pt` files are ~130 bytes / `invalid load key` | weights are Git LFS pointers: run `git lfs install && git lfs pull` |
| data not found | set `FILSEG_DATA` / `FILSEG_WORK` (section 5) |
| two jobs barely faster than one | expected: one job nearly saturates the GPU (~18% extra throughput from a second) |
