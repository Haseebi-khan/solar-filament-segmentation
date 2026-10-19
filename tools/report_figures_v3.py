"""Figures and uncertainty numbers for the technical report (CPU, ~2 min).

  .venv/bin/python tools/report_figures_v3.py --out submission_package_v3/report

Writes:
  figures/examples.png   three out-of-fold training images (good / median / poor per-image PQ, 3 annotators each):
                         top row: every annotator's polygons (one colour each); bottom row: our prediction
  stats.json             per-image bootstrap 95 % CIs of PQ for the final and earlier systems, per-fold PQ,
                         paired bootstrap of final - v2
Only out-of-fold predictions on training images are used (test labels are not available to us).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import sys

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pycocotools import mask as mu  # noqa: E402

from filseg.data import CACHE_DIR, WORK_ROOT  # noqa: E402
from filseg.metric import pq_stats  # noqa: E402

SYSTEMS = {
    "v1 members + v2 selector": "oof_sub_v2_r34_1024+rgy040_1024+r34_1536.csv",
    "v2": "oof_sub_v2_r34_1024+rgy040_1024+r34_1536+r34_2048+effv2s_1536.csv",
    "final": "oof_sub_v3_r34_2048+effv2s_1536+m5_cnxs_1536_v3_w1-1-1_linker.csv",
}
ANN_COLORS = ["#e6194b", "#3cb44b", "#4363d8"]


def load_preds(path):
    df = pd.read_csv(path)
    df["stem"] = df.filament_id.str.rsplit("_", n=1).str[0]
    out = collections.defaultdict(list)
    for s, r in zip(df.stem, df.segmentation_rle):
        out[s].append({"size": [2048, 2048], "counts": r})
    return out


def per_image(meta, preds):
    """(N, 4) array of tp_iou_sum, tp, fp, fn per image, pooled over its annotation sets."""
    rows = []
    for s in meta["stems"]:
        st = [pq_stats(g, preds.get(s, [])) for g in meta["gt_rles"][s]]
        rows.append([sum(x.tp_iou_sum for x in st), sum(x.tp for x in st), sum(x.fp for x in st),
                     sum(x.fn for x in st)])
    return np.array(rows, float)


def pq(a):
    return a[:, 0].sum() / (a[:, 1].sum() + 0.5 * a[:, 2].sum() + 0.5 * a[:, 3].sum())


def bootstrap(a, b=None, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    v = np.array([pq(a[i]) - (pq(b[i]) if b is not None else 0.0) for i in idx])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(np.std(v))


def contours(rles):
    out = []
    for r in rles:
        m = mu.decode(r).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out += [c[:, 0, :] for c in cs if len(c) > 2]
    return out


def crop_box(rles, size=900):
    """Square crop centred on the centroid of all annotated filaments of the image."""
    m = np.zeros((2048, 2048), bool)
    for r in rles:
        m |= mu.decode(r).astype(bool)
    ys, xs = np.nonzero(m)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y0 = int(np.clip(cy - size // 2, 0, 2048 - size))
    x0 = int(np.clip(cx - size // 2, 0, 2048 - size))
    return y0, x0, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stats = {"systems": {}, "n_images": len(meta["stems"])}
    arrays = {}
    for name, fn in SYSTEMS.items():
        a = per_image(meta, load_preds(os.path.join(WORK_ROOT, fn)))
        arrays[name] = a
        lo, hi, sd = bootstrap(a)
        folds = {int(k): float(pq(a[meta["folds"] == k])) for k in range(5)}
        stats["systems"][name] = {"pq": float(pq(a)), "ci95": [lo, hi], "boot_sd": sd, "per_fold": folds,
                                  "fold_mean": float(np.mean(list(folds.values()))),
                                  "fold_sd": float(np.std(list(folds.values()), ddof=1))}
        print(name, json.dumps(stats["systems"][name]))
    lo, hi, sd = bootstrap(arrays["final"], arrays["v2"])
    stats["final_minus_v2"] = {"delta": float(pq(arrays["final"]) - pq(arrays["v2"])), "ci95": [lo, hi]}
    print("final - v2", stats["final_minus_v2"])

    # examples: 3-annotator images, per-image PQ at the 85th / 50th / 15th percentile
    final = arrays["final"]
    img_pq = final[:, 0] / np.maximum(final[:, 1] + 0.5 * final[:, 2] + 0.5 * final[:, 3], 1e-9)
    three = np.where(meta["n_sets"] == 3)[0]
    order = three[np.argsort(img_pq[three])]
    picks = [order[int(q * (len(order) - 1))] for q in (0.85, 0.5, 0.15)]
    images = np.load(os.path.join(CACHE_DIR, "train_images.npy"), mmap_mode="r")
    preds = load_preds(os.path.join(WORK_ROOT, SYSTEMS["final"]))
    fig, grid = plt.subplots(2, 3, figsize=(10.5, 7.4))
    stats["examples"] = []
    for j, i in enumerate(picks):
        s = meta["stems"][i]
        sets = meta["gt_rles"][s]
        y0, x0, size = crop_box([r for g in sets for r in g])
        img = np.asarray(images[i][y0:y0 + size, x0:x0 + size])
        for k, (ax, title) in enumerate(zip(grid[:, j], ("3 annotators", "ours"))):
            ax.imshow(img, cmap="gray", vmin=60, vmax=200)
            groups = [(c, contours(g)) for c, g in zip(ANN_COLORS, sets)] if k == 0 else [("#ffd400", contours(preds.get(s, [])))]
            for color, cs in groups:
                for c in cs:
                    c = c - [x0, y0]
                    ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=color, lw=0.8)
            ax.set_xlim(0, size)
            ax.set_ylim(size, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            label = f"{s[:8]} (site {s[14]}), image PQ {img_pq[i]:.2f}"
            ax.set_title(f"{title}: {label}" if k == 0 else title, fontsize=9)
        stats["examples"].append({"stem": s, "image_pq": float(img_pq[i])})
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(args.out, "figures", "examples.png"), dpi=200, bbox_inches="tight", pad_inches=0.05)
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print("wrote", os.path.join(args.out, "figures", "examples.png"), "and stats.json")


if __name__ == "__main__":
    main()
