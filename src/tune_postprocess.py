"""Tune post-processing on out-of-fold probability maps with the exact PQ metric.

For every (t_seed, t_mask) pair all instances are extracted once and each
instance's additive PQ contribution (a_i, d_i) is computed against every
annotation set of its image (see metric.instance_contributions). Any
area/score filter is then evaluated instantly as
    PQ = sum(a) / (0.5 * N_gt + sum(d)).

Example:
  python src/tune_postprocess.py --names r34_1024
  python src/tune_postprocess.py --names r34_1024 r50_1024 --weights 1 1
"""
from __future__ import annotations

import argparse
import itertools
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd

from filseg.data import CACHE_DIR, WORK_ROOT
from filseg.metric import instance_contributions, pq_from_contributions
from filseg.postprocess import encode_instance, instance_table, label_instances, upsample

_G = {}


def _init(probs_path_list, weights, gt_rles, stems):
    _G["probs"] = [np.load(p, mmap_mode="r") for p in probs_path_list]
    _G["w"] = np.asarray(weights, np.float32) / np.sum(weights)
    _G["gt"] = gt_rles
    _G["stems"] = stems


def blended(i: int) -> np.ndarray:
    maps = [np.asarray(p[i], np.float32) for p in _G["probs"]]
    size = max(m.shape[0] for m in maps)
    acc = np.zeros((size, size), np.float32)
    for w, m in zip(_G["w"], maps):
        acc += w * (m if m.shape[0] == size else upsample(m, size))
    return acc


STRUCT = ["t_seed", "t_mask", "link", "close"]


def _image_instances(args):
    i, structs = args
    p = upsample(blended(i))
    stem = _G["stems"][i]
    rows = []
    for t_seed, t_mask, link, close in structs:
        n, lab = label_instances(p, t_seed, t_mask, link, close)
        objs = instance_table(p, lab, n)
        rles = [encode_instance(lab, o) for o in objs]
        a, d, best = instance_contributions(_G["gt"][stem], rles)
        for o, ai, di, bi in zip(objs, a, d, best):
            rows.append(dict(stem=stem, t_seed=t_seed, t_mask=t_mask, link=link, close=close, a=ai, d=di,
                             best_iou=bi, area=o["area"], mean=o["mean"], max=o["max"], p90=o["p90"]))
    return rows


def collect(names, weights, structs, folds=None, workers=18) -> tuple[pd.DataFrame, int]:
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stems = meta["stems"]
    idx = [i for i in range(len(stems)) if folds is None or meta["folds"][i] in folds]
    n_gt_total = sum(len(g) for i in idx for g in meta["gt_rles"][stems[i]])
    paths = [os.path.join(WORK_ROOT, "preds", n, "oof.npy") for n in names]
    with Pool(workers, initializer=_init, initargs=(paths, weights, meta["gt_rles"], stems)) as pool:
        rows = pool.map(_image_instances, [(i, structs) for i in idx], chunksize=4)
    return pd.DataFrame([r for rr in rows for r in rr]), n_gt_total


def search_filters(df: pd.DataFrame, n_gt_total: int) -> pd.DataFrame:
    res = []
    areas = [0, 100, 200, 300, 400, 500, 600, 800, 1000]
    for key, g in df.groupby(STRUCT):
        a, d = g["a"].to_numpy(), g["d"].to_numpy()
        for score in ("mean", "max", "p90"):
            sv = g[score].to_numpy()
            for min_area in areas:
                for min_score in np.round(np.arange(0.0, 0.99, 0.02), 2):
                    keep = (g["area"].to_numpy() >= min_area) & (sv >= min_score)
                    res.append(dict(zip(STRUCT, key), score=score, min_area=min_area, min_score=min_score,
                                    pq=pq_from_contributions(a[keep], d[keep], n_gt_total), n=int(keep.sum())))
    return pd.DataFrame(res).sort_values("pq", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--seeds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6])
    ap.add_argument("--masks", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--links", type=int, nargs="+", default=[0])
    ap.add_argument("--closes", type=int, nargs="+", default=[0])
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="restrict to images of these folds")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    weights = args.weights or [1.0] * len(args.names)
    structs = [(s, m, l, c) for s, m, l, c in itertools.product(args.seeds, args.masks, args.links, args.closes)
               if m <= s]
    df, n_gt = collect(args.names, weights, structs, folds=args.folds)
    out = args.out or os.path.join(WORK_ROOT, "preds", "+".join(args.names) + "_instances.parquet")
    df.to_parquet(out)
    res = search_filters(df, n_gt)
    pd.set_option("display.width", 200)
    print(f"N_gt={n_gt}  instances file: {out}")
    print("best per structure config:")
    print(res.groupby(STRUCT).head(1).head(25).to_string(index=False))
    print("\noverall top 10:")
    print(res.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
