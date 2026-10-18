"""v3 Stage C4/C5: stacked LightGBM scorer on the candidate pool, then overlap-aware selection; nested CV.

  python src/train_scorer_v3.py --pool work/pools_v3/<members>[_linker]

The scorer predicts u (mean per-annotator TP-IoU) of every pool candidate from the pool features (v1's 26
rescorer features + pool / ensemble / skeleton / linker features, filseg/pool_v3.py), weighted by the image's
number of annotation sets, averaged over --seeds seeds. scorer_k is trained on the candidates of the other
folds and scores fold k (standard stacking). Selection (filseg/select_v3.py) keeps candidates in order of
predicted u while u > tau and they overlap already kept pixels by at most rho; tau and rho are tuned on the
nested scores with the exact organisers' PQ of the final masks (every annotation set separately).

Reported on the same images:
  v1-style    the scorer restricted to H1 candidates and v1 features (= v1's rescorer on the v1 extraction)
  v3 pool     the full pool and all features
  oracle      selection by the true u (from build_pool_v3.py)

Writes <pool>/scorer/scorer.pkl (boosters on all candidates + tau/rho), scorer_config.json (everything
make_submission_v3.py needs) and nested_scores.parquet (honest out-of-fold scores for the OOF CSV).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd

from filseg.data import CACHE_DIR
from filseg.masks_v3 import CropMask, gt_crops
from filseg.metric import PQStats
from filseg.pool_v3 import feature_columns
from filseg.rescore import FEATURES as V1_FEATURES
from filseg.select_v3 import greedy_select, image_stats

PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
_G = {}


def _init(mask_dir, grid, min_keep):
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        _G["meta"] = pickle.load(f)
    _G.update(mask_dir=mask_dir, grid=grid, min_keep=min_keep)


def load_masks(mask_dir: str, i: int) -> list[CropMask]:
    path = os.path.join(mask_dir, f"{i}.pkl")
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        return [CropMask.unpack(t) for t in pickle.load(f)]


def _evaluate(job):
    """job: (image, {variant: (candidate indices, scores)}) -> {variant: {(tau, rho): PQStats}}."""
    i, variants = job
    meta = _G["meta"]
    gts = gt_crops(meta["gt_rles"][meta["stems"][i]])
    masks = load_masks(_G["mask_dir"], i)
    out = {}
    for name, (idx, scores) in variants.items():
        cands = [masks[j] for j in idx]
        out[name] = {(t, r): image_stats(gts, greedy_select(cands, scores, t, r, _G["min_keep"])[0])
                     for t, r in _G["grid"]}
    return out


def fit(x, u, w, cols, rounds, seed):
    return lgb.train(dict(PARAMS, seed=seed), lgb.Dataset(x, u, weight=w, feature_name=cols),
                     num_boost_round=rounds)


def nested(x, u, w, folds, cols, rounds, seeds, rows=None):
    rows = np.ones(len(u), bool) if rows is None else rows
    oof = np.full(len(u), -1.0)
    for k in np.unique(folds):
        tr, va = rows & (folds != k), rows & (folds == k)
        if va.any():
            oof[va] = np.mean([fit(x[tr], u[tr], w[tr], cols, rounds, s).predict(x[va]) for s in range(seeds)], 0)
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="pool directory written by build_pool_v3.py")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--taus", type=float, nargs="+", default=list(np.round(np.arange(0.15, 0.305, 0.01), 2)))
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--min-keep", type=int, default=30, help="drop a trimmed candidate below this many px")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    oof_dir = os.path.join(args.pool, "oof")
    with open(os.path.join(oof_dir, "pool_config.json")) as f:
        pconf = json.load(f)
    df = pd.read_parquet(os.path.join(oof_dir, "features.parquet"))
    cols = [c for c in feature_columns(pconf["families"], pconf["link_taus"]) if c in df.columns]
    x = df[cols].to_numpy(np.float64)
    u, w, folds = df["u"].to_numpy(), df["K"].to_numpy(np.float64), df["fold"].to_numpy()
    h1 = (df["family"].to_numpy() == 0) & (df.get("in_H1", pd.Series(0, index=df.index)).to_numpy() == 1)
    v1_cols = [c for c in V1_FEATURES if c in cols]
    print(f"{len(df)} candidates ({int(h1.sum())} from H1), {len(cols)} features, {df['image'].nunique()} images")

    t0 = time.time()
    oof = nested(x, u, w, folds, cols, args.rounds, args.seeds)
    oof_v1 = nested(df[v1_cols].to_numpy(np.float64), u, w, folds, v1_cols, args.rounds, args.seeds, h1)
    print(f"nested scorers fitted in {time.time() - t0:.0f}s")

    grid = [(float(t), float(r)) for t in args.taus for r in args.rhos]
    jobs = []
    for i, g in df.groupby("image"):
        idx = g["cand"].to_numpy()
        rows = g.index.to_numpy()
        hm = h1[rows]
        jobs.append((int(i), {"pool": (idx, oof[rows]), "v1": (idx[hm], oof_v1[rows][hm])}))
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    covered = set(df["image"].unique())
    missing_fn = sum(len(g) for i, s in enumerate(meta["stems"]) if i not in covered for g in meta["gt_rles"][s])
    with Pool(args.workers, initializer=_init, initargs=(os.path.join(oof_dir, "masks"), grid, args.min_keep)) as pool:
        res = pool.map(_evaluate, jobs, chunksize=2)
    best = {}
    for name in ("v1", "pool"):
        scores = {}
        for cfg in grid:
            tot = PQStats(fn=missing_fn)
            for r in res:
                tot.add(r[name][cfg])
            scores[cfg] = tot
        b = max(scores, key=lambda c: scores[c].pq)
        best[name] = (b, scores[b])
        print(f"nested {name:>4}: {scores[b]}  at tau {b[0]}, rho {b[1]}")
    oracle_path = os.path.join(oof_dir, "oracle.json")
    if os.path.exists(oracle_path):
        with open(oracle_path) as f:
            orc = json.load(f)
        print("oracle (true u): " + "  ".join(f"{k} {v['pq']:.4f}" for k, v in orc.items()))
    print(f"v3 pool + scorer vs v1-style on the same maps: {best['pool'][1].pq - best['v1'][1].pq:+.4f}")

    (tau, rho), st = best["pool"]
    out = os.path.join(args.pool, "scorer")
    os.makedirs(out, exist_ok=True)
    boosters = [fit(x, u, w, cols, args.rounds, s) for s in range(args.seeds)]
    with open(os.path.join(out, "scorer.pkl"), "wb") as f:
        pickle.dump({"boosters": [b.model_to_string() for b in boosters], "features": cols, "tau": tau,
                     "rho": rho, "min_keep": args.min_keep}, f)
    pd.DataFrame({"image": df["image"], "cand": df["cand"], "score": oof}).to_parquet(
        os.path.join(out, "nested_scores.parquet"))
    conf = dict(pool=os.path.abspath(args.pool), scorer=os.path.abspath(os.path.join(out, "scorer.pkl")),
                tau=tau, rho=rho, min_keep=args.min_keep, nested_pq=st.pq, nested_pq_v1_style=best["v1"][1].pq,
                **{k: pconf[k] for k in ("names", "weights", "families", "link_taus", "linker", "min_area")})
    with open(os.path.join(out, "scorer_config.json"), "w") as f:
        json.dump(conf, f, indent=1)
    imp = pd.Series(boosters[0].feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("feature gain:", ", ".join(f"{k} {v:.0f}" for k, v in imp.head(15).items()))
    sel_fams = []
    for i, g in df.groupby("image"):
        masks = load_masks(os.path.join(oof_dir, "masks"), int(i))
        _, kept = greedy_select([masks[j] for j in g["cand"]], oof[g.index.to_numpy()], tau, rho, args.min_keep)
        sel_fams += list(g["family"].to_numpy()[kept])
    names = pconf.get("columns") or []
    counts = pd.Series(sel_fams).map(lambda k: names[k] if k < len(names) else k).value_counts()
    print("families of the selected candidates:", ", ".join(f"{k} {v}" for k, v in counts.items()))
    print("saved", out)


if __name__ == "__main__":
    main()
