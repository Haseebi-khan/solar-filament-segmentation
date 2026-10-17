"""Builds notebooks/pipeline.ipynb (run it with jupyter nbconvert --execute)."""
import json

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Solar filament segmentation: full pipeline

This notebook walks through the complete solution for the *Solar Filament Segmentation
Challenge 2026*, from raw H-alpha images to the submission CSV. Long-running steps (network
training, test-time-augmented inference) are performed by the scripts in `src/`; the notebook
loads their artefacts from `$FILSEG_WORK` and shows every intermediate result.

| step | script |
|---|---|
| decode images, build soft multi-annotator targets | `src/prepare_data.py` |
| train U-Nets on 5 image-level folds | `src/train.py`, `scripts/train_folds.sh` |
| out-of-fold + test probability maps (8x dihedral TTA) | `src/predict.py` |
| tune instance extraction with the exact PQ metric | `src/tune_postprocess.py` |
| second-stage instance rescorer (LightGBM) | `src/train_rescorer.py` |
| write + validate the submission | `src/make_submission.py` |
""")

code("""
import os, sys, pickle, json, re, glob
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt
def find_root():
    d = os.environ.get('FILSEG_ROOT') or os.getcwd()
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, 'src', 'filseg')):
            return d
        d = os.path.dirname(d)
    raise RuntimeError('run from inside the repository or set FILSEG_ROOT')
ROOT = find_root()
sys.path.insert(0, os.path.join(ROOT, 'src'))
from pycocotools import mask as mask_util
from filseg.data import CACHE_DIR, WORK_ROOT, load_train_annotations
from filseg.metric import evaluate, pq_stats, PQStats, instance_contributions, pq_from_contributions
from filseg.postprocess import PPConfig, extract, upsample

MEMBERS = __MEMBERS__          # ensemble members (each trained on the same 5 folds)
FINAL_CONFIG = __CONFIG__      # post-processing + rescorer configuration used for the submission
meta = pickle.load(open(os.path.join(CACHE_DIR, 'train_meta.pkl'), 'rb'))
stems, gt = meta['stems'], meta['gt_rles']
train_images = np.load(os.path.join(CACHE_DIR, 'train_images.npy'), mmap_mode='r')
print(len(stems), 'training images,', sum(len(v) for v in gt.values()), 'annotation sets')
""")

md("""
## 1. Data: several annotators per image

Each image may be annotated independently by up to three annotators. The organisers'
metric scores every annotation set (`<annotator>-<image>`) separately against all predicted
filaments of the image.
""")

code("""
print('annotation sets per image:', pd.Series(meta['n_sets']).value_counts().sort_index().to_dict())
i = [k for k, s in enumerate(stems) if meta['n_sets'][k] == 3 and len(gt[s][0]) >= 6][3]
img = np.asarray(train_images[i])
fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
axes[0].imshow(img, cmap='gray'); axes[0].set_title(stems[i])
for ax, rles, sid in zip(axes[1:], gt[stems[i]], meta['set_ids'][stems[i]]):
    ax.imshow(img, cmap='gray')
    m = mask_util.decode(rles).max(2)
    ax.contour(m, levels=[0.5], colors='r', linewidths=0.8)
    ax.set_title(f'{sid}: {len(rles)} filaments')
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()
""")

md("""
## 2. The metric and the human ceiling

`filseg.metric` re-implements the self-evaluation notebook on COCO RLEs (verified identical
to its dense implementation). Scoring one annotator against another shows how much humans agree.
""")

code("""
st = PQStats()
for s in stems:
    sets = gt[s]
    for a in range(len(sets)):
        for b in range(len(sets)):
            if a != b:
                st.add(pq_stats(sets[b], sets[a]))
print('one annotator vs another:', st)
""")

md("""
With disjoint predictions every GT filament can be hit by at most one prediction, which makes
PQ additive over predicted instances:

$$PQ(S)=\\frac{\\sum_{i\\in S} a_i}{\\tfrac12 N_{gt}+\\sum_{i\\in S} d_i},$$

with $a_i$ the summed TP-IoUs of instance $i$ over the image's annotation sets and $d_i=m_i/2$
for an image with $m_i$ sets. An instance improves PQ iff its expected per-annotator TP-IoU
exceeds PQ/2. This drives the threshold search and the rescorer.

## 3. Soft targets

The network is trained to predict, per pixel, the fraction of annotators whose polygon covers it.
""")

code("""
soft = np.load(os.path.join(CACHE_DIR, 'train_soft_1024.npy'), mmap_mode='r')
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA), cmap='gray'); axes[0].set_title('image')
axes[1].imshow(np.asarray(soft[i]) / 255, cmap='magma'); axes[1].set_title('soft target (fraction of annotators)')
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()
""")

md("""
## 4. Training

Each member is a U-Net (segmentation_models_pytorch, ImageNet encoder, 1-channel input,
scSE decoder attention) trained on full-disk images with BCE + soft Dice, AdamW, cosine schedule,
EMA weights and GPU-side augmentation (rotation, flips, scale, shift, gamma/contrast, blur, noise).
Images are sampled in proportion to their number of annotation sets. Validation curves:
""")

code("""
fig, ax = plt.subplots(figsize=(9, 4.5))
for name in MEMBERS:
    for f in range(5):
        path = os.path.join(WORK_ROOT, 'models', name, f'fold{f}.log')
        if not os.path.exists(path): continue
        ep, pq = [], []
        for line in open(path):
            m = re.search(r'epoch (\\d+)/\\d+.*pq@0.5 ([0-9.]+)', line)
            if m: ep.append(int(m.group(1))); pq.append(float(m.group(2)))
        ax.plot(ep, pq, marker='o', ms=3, label=f'{name} fold {f}' if f == 0 else None,
                color=f'C{MEMBERS.index(name)}', alpha=0.7)
ax.set_xlabel('epoch'); ax.set_ylabel('val PQ (no TTA, threshold 0.5)'); ax.legend(); ax.grid(alpha=0.3)
plt.show()
""")

md("## 5. Out-of-fold probability maps and instance extraction")

code("""
oofs = [np.load(os.path.join(WORK_ROOT, 'preds', n, 'oof.npy'), mmap_mode='r') for n in MEMBERS]
weights = np.array(FINAL_CONFIG.get('weights', [1.0] * len(MEMBERS)), float)
weights /= weights.sum()
def blend(i):
    return sum(w * upsample(np.asarray(o[i])) for w, o in zip(weights, oofs))
from filseg.rescore import Rescorer
rescorer = Rescorer.load(FINAL_CONFIG['rescorer']) if FINAL_CONFIG.get('rescorer') else None
cfg = PPConfig(**{k: v for k, v in FINAL_CONFIG.items() if k in PPConfig.__dataclass_fields__})
p = blend(i)
cand, objs = extract(None, cfg, p_full=p, image=img)
keep = rescorer.select(objs) if rescorer else [True] * len(cand)
rles = [r for r, k in zip(cand, keep) if k]
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(p, cmap='magma'); axes[0].set_title('out-of-fold ensemble probability')
axes[1].imshow(img, cmap='gray')
for r, k in zip(cand, keep):
    axes[1].contour(mask_util.decode(r), levels=[0.5], colors='lime' if k else 'orange', linewidths=0.8)
axes[1].set_title(f'{len(rles)} kept (green) of {len(cand)} candidates (orange = dropped by rescorer)')
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()
for k, g in enumerate(gt[stems[i]]):
    print(f'annotator {k + 1}:', pq_stats(g, rles))
""")

md("""
## 6. Final configuration and out-of-fold score

The final configuration (thresholds, fragment linking, rescorer) was selected on the
out-of-fold predictions; `src/train_rescorer.py` reports its nested-CV score.
""")

code("""
print(json.dumps(FINAL_CONFIG, indent=1))
for path in sorted(glob.glob(os.path.join(ROOT, 'logs', 'final_*.txt'))):
    print(open(path).read())
""")

md("## 7. Submission file")

code("""
sub_path = os.path.join(ROOT, 'submissions', 'submission.csv')
sub = pd.read_csv(sub_path)
ids = sub['filament_id'].str.split('_', n=1).str[0]
print(sub.shape, 'rows;', ids.nunique(), 'test images with at least one filament')
print(sub.head())
masks_per_image = ids.value_counts()
print('filaments per image: mean %.2f, max %d' % (masks_per_image.mean(), masks_per_image.max()))
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}

if __name__ == "__main__":
    import sys
    members, config_path, out = sys.argv[1].split(","), sys.argv[2], sys.argv[3]
    with open(config_path) as f:
        cfg = json.load(f)
    for cell in nb.cells:
        cell.source = cell.source.replace("__MEMBERS__", repr(members)).replace("__CONFIG__", repr(cfg))
    nbf.write(nb, out)
    print("wrote", out)
