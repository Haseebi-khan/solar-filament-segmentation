"""v2: train the extent selector on out-of-fold candidates and tune its keep-threshold.

  python src/train_selector_v2.py --names r34_1024 rgy040_1024 r34_1536 --weights 1 1 2 \
      --out work/selectors_v2/r34_1024+rgy040_1024+r34_1536.pkl

For every candidate filament and every extent level (see filseg/hypotheses_v2.py) a LightGBM
regressor predicts u = mean over the image's annotation sets of (IoU if IoU > 0.5 else 0). Each
filament keeps its best-scoring level, and filaments whose best u is below a threshold are dropped.
Evaluation is nested like v1: the model applied to fold k is trained on the other folds only.

Printed for comparison on the same candidates:
  oracle (true u)       best level and keep decision from the ground truth: an upper bound
  v1 extent only        the same model restricted to the v1 level, i.e. the v1 rescorer
  v2 selector           all levels

Writes <out> (5 boosters trained on all rows + threshold), <out stem>_config.json (everything
make_submission_v2.py needs) and <out stem>_nested.parquet (nested out-of-fold scores, so the OOF
CSV can be scored exactly with make_submission_v2.py --oof --nested-scores).
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

from filseg.data import CACHE_DIR
from filseg.hypotheses_v2 import FEATURES_V2, LEVELS, best_rows, encode_row, feature_matrix_v2, hypotheses
from filseg.metric import instance_contributions
from filseg.postprocess import upsample
from make_submission import load_blend
from train_rescorer import PARAMS, best_threshold

_G = {}


def _init(names, weights, cfg):
    _G["probs"] = load_blend(names, weights, "oof")
    _G["images"] = np.load(os.path.join(CACHE_DIR, "train_images.npy"), mmap_mode="r")
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        _G["meta"] = pickle.load(f)
    _G["cfg"] = cfg


def _image_table(i):
    meta = _G["meta"]
    stem = meta["stems"][i]
    p = upsample(_G["probs"][i])
    rows, labs = hypotheses(p, np.asarray(_G["images"][i]), **_G["cfg"])
    if not rows:
        return None
    a = np.zeros(len(rows))
    d = np.zeros(len(rows))
    for t in sorted({r["level"] for r in rows}):
        idx = [j for j, r in enumerate(rows) if r["level"] == t]
        aa, dd, _ = instance_contributions(meta["gt_rles"][stem], [encode_row(labs, rows[j]) for j in idx])
        a[idx], d[idx] = aa, dd
    key = np.array([[i, r["label"], r["level"]] for r in rows], np.float64)
    return feature_matrix_v2(rows), a, d, key, len(meta["gt_rles"][stem])


def build_table(names, weights, cfg, workers):
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    with Pool(workers, initializer=_init, initargs=(names, weights, cfg)) as pool:
        out = [r for r in pool.map(_image_table, range(len(meta["stems"])), chunksize=4) if r is not None]
    x = np.concatenate([o[0] for o in out])
    a = np.concatenate([o[1] for o in out])
    d = np.concatenate([o[2] for o in out])
    key = np.concatenate([o[3] for o in out])
    m = np.concatenate([np.full(len(o[1]), o[4]) for o in out]).astype(np.float64)
    n_gt = sum(len(g) for s in meta["stems"] for g in meta["gt_rles"][s])
    return x, a, d, key, m, meta["folds"][key[:, 0].astype(int)], n_gt


def fit(x, u, w, rounds, seed=0):
    params = dict(PARAMS, seed=seed)
    return lgb.train(params, lgb.Dataset(x, u, weight=w, feature_name=FEATURES_V2), num_boost_round=rounds)


def selected_pq(scores, inst, a, d, n_gt):
    """Best level per filament by score, then the best keep-threshold: (PQ, threshold, chosen rows)."""
    b = best_rows(inst, scores)
    pq, thr = best_threshold(scores[b], a[b], d[b], n_gt)
    return pq, thr, b[scores[b] > thr]


def nested_scores(x, u, m, folds, rows_mask, rounds):
    oof = np.full(len(u), -1.0)
    for k in np.unique(folds):
        tr = (folds != k) & rows_mask
        va = (folds == k) & rows_mask
        oof[va] = fit(x[tr], u[tr], m[tr], rounds).predict(x[va])
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--t-seed", type=float, default=0.6)
    ap.add_argument("--link", type=int, default=8)
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--levels", type=float, nargs="+", default=list(LEVELS))
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    weights = args.weights or [1.0] * len(args.names)
    cfg = dict(t_seed=args.t_seed, link=args.link, min_area=args.min_area, levels=tuple(args.levels))

    x, a, d, key, m, folds, n_gt = build_table(args.names, weights, cfg, args.workers)
    u = a / m
    inst = key[:, 0] * 100000 + key[:, 1]
    level = key[:, 2]
    base = np.isclose(level, args.t_seed)
    print(f"filaments {int(base.sum())}  hypotheses {len(u)}  levels {args.levels}  N_gt {n_gt}")

    pq_or_v1, _ = best_threshold(u[base], a[base], d[base], n_gt)
    pq_or_v2, _, ch = selected_pq(u, inst, a, d, n_gt)
    print(f"oracle (true u): v1 extent only {pq_or_v1:.4f}   best level per filament {pq_or_v2:.4f}")

    oof_v1 = nested_scores(x, u, m, folds, base, args.rounds)
    pq_v1, thr_v1 = best_threshold(oof_v1[base], a[base], d[base], n_gt)
    print(f"nested CV, v1 extent only: PQ {pq_v1:.4f} at threshold {thr_v1}")

    oof = nested_scores(x, u, m, folds, np.ones(len(u), bool), args.rounds)
    pq, thr, ch = selected_pq(oof, inst, a, d, n_gt)
    print(f"nested CV, v2 selector:    PQ {pq:.4f} at threshold {thr}   ({pq - pq_v1:+.4f} vs v1 extent)")
    chosen_levels = pd.Series(level[ch]).value_counts().sort_index()
    print("levels chosen for kept filaments:", ", ".join(f"{k:g}: {v}" for k, v in chosen_levels.items()))

    boosters = [fit(x, u, m, args.rounds, seed=s) for s in range(5)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"boosters": [b.model_to_string() for b in boosters], "threshold": thr,
                     "features": FEATURES_V2}, f)
    stem = os.path.splitext(args.out)[0]
    pd.DataFrame({"image": key[:, 0].astype(int), "label": key[:, 1].astype(int), "level": level,
                  "score": oof}).to_parquet(stem + "_nested.parquet")
    conf = dict(names=args.names, weights=weights, selector=os.path.abspath(args.out), threshold=thr,
                t_seed=args.t_seed, link=args.link, min_area=args.min_area, levels=list(args.levels),
                nested_pq=pq, nested_pq_v1_extent=pq_v1)
    with open(stem + "_config.json", "w") as f:
        json.dump(conf, f, indent=1)
    imp = pd.Series(boosters[0].feature_importance("gain"), index=FEATURES_V2).sort_values(ascending=False)
    print("feature gain:", ", ".join(f"{k} {v:.0f}" for k, v in imp.head(12).items()))
    print("saved", args.out, "and", stem + "_config.json")


if __name__ == "__main__":
    main()
