"""v3: paired bootstrap of the PQ difference between two out-of-fold CSVs (blueprint section 13).

  python src/compare_v3.py work/oof_sub_v2_<members>.csv work/oof_sub_v3_<members>.csv [--b 2000]

Images are resampled with replacement together with all their annotation sets, the global PQ of both systems
is recomputed on every resample, and the mean difference (B - A), its 90 % interval and P(delta > 0) are
reported. Gate: adopt B if mean delta >= +0.003 and P(delta > 0) >= 0.90; for architecture or loss changes with
delta < 0.008 repeat with a second seed before deciding.
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from filseg.data import CACHE_DIR
from filseg.metric import pq_stats


def per_image_stats(csv: str, meta: dict) -> np.ndarray:
    """(N_images, 4): tp_iou_sum, tp, fp, fn summed over each image's annotation sets."""
    df = pd.read_csv(csv)
    stems = df["filament_id"].str.split("_", n=1).str[0]
    preds = {k: [{"size": [2048, 2048], "counts": c.encode()} for c in g["segmentation_rle"]]
             for k, g in df.groupby(stems)}
    out = np.zeros((len(meta["stems"]), 4))
    for i, s in enumerate(meta["stems"]):
        for g in meta["gt_rles"][s]:
            st = pq_stats(g, preds.get(s, []))
            out[i] += (st.tp_iou_sum, st.tp, st.fp, st.fn)
    return out


def pq_of(stats: np.ndarray) -> np.ndarray:
    """stats (..., 4) -> PQ."""
    den = stats[..., 1] + 0.5 * stats[..., 2] + 0.5 * stats[..., 3]
    return np.where(den > 0, stats[..., 0] / np.maximum(den, 1e-9), 0.0)


def bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(a)
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[k] = pq_of(b[idx].sum(0)) - pq_of(a[idx].sum(0))
    return {"pq_a": float(pq_of(a.sum(0))), "pq_b": float(pq_of(b.sum(0))),
            "delta": float(pq_of(b.sum(0)) - pq_of(a.sum(0))), "delta_mean": float(deltas.mean()),
            "ci90": (float(np.percentile(deltas, 5)), float(np.percentile(deltas, 95))),
            "p_gt0": float((deltas > 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_a", help="baseline")
    ap.add_argument("csv_b", help="candidate")
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="restrict to images of these folds")
    args = ap.parse_args()
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    a, b = per_image_stats(args.csv_a, meta), per_image_stats(args.csv_b, meta)
    if args.folds is not None:
        keep = np.isin(meta["folds"], args.folds)
        a, b = a[keep], b[keep]
    r = bootstrap(a, b, args.b)
    adopt = r["delta_mean"] >= 0.003 and r["p_gt0"] >= 0.90
    print(f"A {os.path.basename(args.csv_a)}: PQ {r['pq_a']:.4f}\nB {os.path.basename(args.csv_b)}: PQ {r['pq_b']:.4f}")
    print(f"delta B-A {r['delta']:+.4f}  bootstrap mean {r['delta_mean']:+.4f}  90% CI [{r['ci90'][0]:+.4f}, "
          f"{r['ci90'][1]:+.4f}]  P(delta>0) {r['p_gt0']:.3f}  ({args.b} resamples of {len(a)} images)")
    verdict = "ADOPT B" if adopt else "keep A"
    if adopt and r["delta_mean"] < 0.008:
        verdict += " (delta < 0.008: confirm with a second seed if this is an architecture/loss change)"
    print("gate (mean delta >= +0.003 and P(delta>0) >= 0.90):", verdict)


if __name__ == "__main__":
    main()
