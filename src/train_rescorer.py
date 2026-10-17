"""Train the instance rescorer on out-of-fold candidates and tune its keep-threshold.

  python src/train_rescorer.py --names r34_1024 --t-seed 0.5 --t-mask 0.4 --out work/rescorer.pkl

Evaluation is nested: the rescorer applied to fold k is trained only on the
candidates of the other folds, so the reported PQ is out-of-fold.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd

from filseg.data import CACHE_DIR, WORK_ROOT
from filseg.metric import instance_contributions, pq_from_contributions
from filseg.postprocess import PPConfig, extract, upsample
from filseg.rescore import FEATURES, Rescorer, feature_matrix

_G = {}


def _init(paths, weights, cfg):
    _G["probs"] = [np.load(p, mmap_mode="r") for p in paths]
    _G["w"] = np.asarray(weights, np.float32) / np.sum(weights)
    _G["cfg"] = cfg
    _G["images"] = np.load(os.path.join(CACHE_DIR, "train_images.npy"), mmap_mode="r")
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        _G["meta"] = pickle.load(f)


def _candidates(i):
    parts = [np.asarray(p[i], np.float32) for p in _G["probs"]]
    size = max(p.shape[0] for p in parts)
    prob = sum(w * (p if p.shape[0] == size else upsample(p, size)) for w, p in zip(_G["w"], parts))
    meta = _G["meta"]
    stem = meta["stems"][i]
    rles, objs = extract(prob, _G["cfg"], image=np.asarray(_G["images"][i]))
    if not objs:
        return None
    a, d, best = instance_contributions(meta["gt_rles"][stem], rles)
    x = feature_matrix(objs)
    m = len(meta["gt_rles"][stem])
    return x, a, d, np.full(len(objs), i), np.full(len(objs), m)


def build_table(names, weights, cfg, workers=18):
    paths = [os.path.join(WORK_ROOT, "preds", n, "oof.npy") for n in names]
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    with Pool(workers, initializer=_init, initargs=(paths, weights, cfg)) as pool:
        out = [r for r in pool.map(_candidates, range(len(meta["stems"])), chunksize=4) if r is not None]
    x = np.concatenate([o[0] for o in out])
    a = np.concatenate([o[1] for o in out])
    d = np.concatenate([o[2] for o in out])
    img = np.concatenate([o[3] for o in out])
    m = np.concatenate([o[4] for o in out])
    n_gt = sum(len(g) for s in meta["stems"] for g in meta["gt_rles"][s])
    return x, a, d, img, m, meta["folds"][img], n_gt


PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)


def fit(x, u, w, rounds, seed=0):
    params = dict(PARAMS, seed=seed)
    return lgb.train(params, lgb.Dataset(x, u, weight=w, feature_name=FEATURES), num_boost_round=rounds)


def best_threshold(score, a, d, n_gt):
    best = (0.0, 0.0)
    for t in np.round(np.arange(0.0, 1.0, 0.005), 3):
        keep = score > t
        pq = pq_from_contributions(a[keep], d[keep], n_gt)
        if pq > best[0]:
            best = (pq, float(t))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--t-seed", type=float, default=0.5)
    ap.add_argument("--t-mask", type=float, default=0.4)
    ap.add_argument("--link", type=int, default=8)
    ap.add_argument("--close", type=int, default=-1, help="-1 = fill holes per instance (see PPConfig.close)")
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    weights = args.weights or [1.0] * len(args.names)
    cfg = PPConfig(t_seed=args.t_seed, t_mask=args.t_mask, min_area=args.min_area, link=args.link, close=args.close)
    x, a, d, img, m, folds, n_gt = build_table(args.names, weights, cfg)
    u = a / m
    print(f"candidates {len(u)}  N_gt {n_gt}  PQ(all candidates) {pq_from_contributions(a, d, n_gt):.4f}  "
          f"oracle PQ {best_threshold(u, a, d, n_gt)[0]:.4f}")

    oof = np.zeros(len(u))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        oof[va] = fit(x[tr], u[tr], m[tr], args.rounds).predict(x[va])
    pq, thr = best_threshold(oof, a, d, n_gt)
    print(f"nested-CV rescorer: PQ {pq:.4f} at threshold {thr}")
    for name, col in [("mean", FEATURES.index("mean")), ("p90", FEATURES.index("p90"))]:
        bp, bt = best_threshold(x[:, col], a, d, n_gt)
        print(f"  reference: threshold on {name} prob only -> PQ {bp:.4f} at {bt}")

    boosters = [fit(x, u, m, args.rounds, seed=s) for s in range(5)]
    Rescorer(boosters, thr).save(args.out)
    conf = dict(t_seed=args.t_seed, t_mask=args.t_mask, min_area=args.min_area, min_score=0.0, score="mean",
                link=args.link, close=args.close, rescorer=args.out)
    with open(os.path.splitext(args.out)[0] + "_config.json", "w") as f:
        json.dump(conf, f, indent=1)
    imp = pd.Series(boosters[0].feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print("feature gain:", ", ".join(f"{k} {v:.0f}" for k, v in imp.head(10).items()))
    print("saved", args.out)


if __name__ == "__main__":
    main()
