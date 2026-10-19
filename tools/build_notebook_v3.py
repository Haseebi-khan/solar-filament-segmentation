"""Builds submission_package_v3/notebook/reproduce_v3.ipynb: reproduces the final test submission from the trained
weights (inference only; GPU, about 1 h). Run the built notebook with jupyter nbconvert --execute or cell by cell.

  .venv/bin/python tools/build_notebook_v3.py
"""
import os

import nbformat as nbf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission_package_v3", "notebook",
                   "reproduce_v3.ipynb")
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Solar Filament Segmentation 2026: reproduce the final submission

This notebook recreates `submission_package_v3/kaggle/submission_v3_final.csv` (nested out-of-fold PQ 0.4477) from
the trained weights. It uses only the competition images; no labels of the test set exist anywhere in the pipeline.

| step | script | time (1 GPU) |
|---|---|---|
| 1. decode the images, folds and metadata | `src/prepare_data.py` | ~10 min (CPU) |
| 2. probability maps, 5 fold models x 8 TTA views per member | `src/predict.py`, `src/predict_v3.py` | ~45 min |
| 3. candidate pool with the learned fragment linker | `src/build_pool_v3.py` | ~2 min (CPU) |
| 4. stacked scorer + overlap-aware selection, checked CSV | `src/make_submission_v3.py` | ~1 min (CPU) |

Members of the final blend (weights 1 1 1): `r34_2048` (U-Net ResNet-34 @2048), `effv2s_1536` (U-Net
EfficientNetV2-S @1536), `m5_cnxs_1536_v3` (U-Net on DINOv3 ConvNeXt-S @1536). The weights of the first two are in the
repository (`work/models/`). The M5 weights are larger than GitHub's 100 MB limit and come from the release asset
`m5_cnxs_1536_v3_weights.zip`. The linker and the scorer are in `submission_package_v3/release/stagec_final/`.

Everything is written to a separate work directory (`work_reproduce/`), so the original `work/` stays untouched.
The predictions scripts also write out-of-fold maps for the training images; they are not used here.
""")

code("""
import json, os, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "src" / "train_v3.py").exists():          # find the repository root from any sub-folder
    if ROOT.parent == ROOT:
        raise SystemExit("run this notebook inside the repository")
    ROOT = ROOT.parent
PKG = ROOT / "submission_package_v3"
DATA = Path(os.environ.get("FILSEG_DATA", ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"))
WORK = Path(os.environ.get("FILSEG_REPRO_WORK", ROOT / "work_reproduce"))
PY = sys.executable
ENV = dict(os.environ, FILSEG_DATA=str(DATA), FILSEG_WORK=str(WORK))
NAMES = ["r34_2048", "effv2s_1536", "m3_r34_1536_v3", "m4_vitb_1536_v3", "m5_cnxs_1536_v3"]
WEIGHTS = [1, 1, 0, 0, 1]                                    # exactly as the scorer was trained (zeros are skipped)
USED = [n for n, w in zip(NAMES, WEIGHTS) if w > 0]

def run(*args):
    print("$", " ".join(map(str, args)), flush=True)
    subprocess.run([str(a) for a in args], cwd=ROOT, env=ENV, check=True)

assert (DATA / "test" / "test_images").is_dir(), f"competition data not found in {DATA} (see SETUP.md)"
print("repo", ROOT, "\\ndata", DATA, "\\nwork", WORK)
""")

md("""
## 0. Weights

`r34_2048` and `effv2s_1536` are tracked in git. For M5, download `m5_cnxs_1536_v3_weights.zip` from the GitHub
release of the repository into the repository root (or set `M5_ZIP`); the cell unpacks it into `work/models/`.
The work directory gets links to the model folders, so nothing is copied.
""")

code("""
models = ROOT / "work" / "models"
m5_zip = Path(os.environ.get("M5_ZIP", ROOT / "m5_cnxs_1536_v3_weights.zip"))
if not all((models / "m5_cnxs_1536_v3" / f"fold{k}.pt").exists() for k in range(5)):
    assert m5_zip.exists(), f"M5 weights missing: download m5_cnxs_1536_v3_weights.zip to {m5_zip}"
    with zipfile.ZipFile(m5_zip) as z:
        z.extractall(models)
(WORK / "models").mkdir(parents=True, exist_ok=True)
for n in USED:
    folds = sorted((models / n).glob("fold?.pt"))
    assert len(folds) == 5, f"{n}: expected 5 fold checkpoints, found {len(folds)}"
    link = WORK / "models" / n
    if not link.exists():
        link.symlink_to(models / n, target_is_directory=True)
    print(n, "ok:", ", ".join(f.name for f in folds))
""")

md("## 1. Decode the images (folds and metadata are deterministic, seed 42)")
code("""
if not (WORK / "cache" / "test_meta.pkl").exists():
    run(PY, "src/prepare_data.py")
""")

md("## 2. Probability maps (8 dihedral TTA views, mean over the 5 fold models)")
code("""
run(PY, "src/predict.py", "--name", "r34_2048", "--tta", "8")
run(PY, "src/predict.py", "--name", "effv2s_1536", "--tta", "8")
run(PY, "src/predict_v3.py", "--name", "m5_cnxs_1536_v3", "--tta", "8")
""")

md("## 3. Candidate pool on the test images (hypothesis families H1-H6 + linker H7 at 0.4 / 0.6)")
code("""
tag = "+".join(USED) + "_w1-1-1_linker"
pool = WORK / "pools_v3" / tag
run(PY, "src/build_pool_v3.py", "--names", *NAMES, "--weights", *WEIGHTS, "--split", "test",
    "--linker", PKG / "release" / "stagec_final" / "linker.pkl", "--link-taus", "0.4", "0.6", "--out", pool)
""")

md("## 4. Stacked scorer, overlap-aware selection, checked submission")
code("""
conf = json.loads((PKG / "release" / "stagec_final" / "scorer_config.json").read_text())
conf["pool"] = str(pool)
conf["scorer"] = str(PKG / "release" / "stagec_final" / "scorer.pkl")
conf_path = WORK / "scorer_config_reproduce.json"
conf_path.write_text(json.dumps(conf, indent=1))
out = ROOT / "submissions" / "submission_v3_reproduced.csv"
run(PY, "src/make_submission_v3.py", "--config", conf_path, "--out", out)
""")

md("""
## 5. Compare with the submitted file

Every reproduced mask is matched to the submitted mask of the same image with the highest IoU. GPU inference in
bf16 is not bit-exact across machines, so a few boundary pixels can differ; the counts and IoUs should be ~1.
""")
code("""
import numpy as np, pandas as pd
from pycocotools import mask as mu

def load(path):
    df = pd.read_csv(path)
    df["stem"] = df.filament_id.str.rsplit("_", n=1).str[0]
    return {s: [{"size": [2048, 2048], "counts": r} for r in g.segmentation_rle] for s, g in df.groupby("stem")}

a, b = load(out), load(PKG / "kaggle" / "submission_v3_final.csv")
best = []
for s in set(a) | set(b):
    ra, rb = a.get(s, []), b.get(s, [])
    if ra and rb:
        best += list(np.asarray(mu.iou(ra, rb, [0] * len(rb))).reshape(len(ra), len(rb)).max(1))
    else:
        best += [0.0] * len(ra)
best = np.array(best)
print(f"filaments: reproduced {sum(map(len, a.values()))}, submitted {sum(map(len, b.values()))}")
print(f"best-match IoU: mean {best.mean():.4f}, share >= 0.99: {(best >= 0.99).mean():.3f}")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
nbf.write(nb, OUT)
print("wrote", os.path.normpath(OUT))
