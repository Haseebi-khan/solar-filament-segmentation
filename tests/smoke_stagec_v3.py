"""Smoke test of the whole Stage C chain on a few images of the existing v2 maps (single process, no GPU).

  .venv/bin/python tests/smoke_stagec_v3.py [--n 8]

Runs the real code paths of build_pool_v3 (all families + learned-linker families), train_linker_v3 (pair
features and labels, grouping check), train_scorer_v3 (nested fit, selection, PQ) and make_submission_v3
(selection, RLE encoding, format checks) on --n images, with tiny models. The numbers are meaningless; the
point is that nothing crashes before you pay for a long run. ~1-3 min.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_pool_v3 as bp  # noqa: E402
import make_submission_v3 as mv3  # noqa: E402
import train_linker_v3 as tl  # noqa: E402
import train_scorer_v3 as ts  # noqa: E402
from filseg.data import CACHE_DIR, WORK_ROOT  # noqa: E402
from filseg.linker_v3 import Linker  # noqa: E402
from filseg.pool_v3 import feature_columns  # noqa: E402
from make_submission import validate_submission  # noqa: E402


def step(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


def main():
    global T0
    T0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--names", nargs="+", default=["r34_2048", "effv2s_1536"])
    ap.add_argument("--keep", action="store_true", help="keep the temporary directory")
    args = ap.parse_args()
    for n in args.names:
        if not os.path.exists(os.path.join(WORK_ROOT, "preds", n, "oof.npy")):
            raise SystemExit(f"no maps for {n}: work/preds/{n}/oof.npy")
    weights = [1.0] * len(args.names)
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    rng = np.random.RandomState(0)
    imgs = sorted(rng.choice(len(meta["stems"]), args.n, replace=False).tolist())
    tmp = tempfile.mkdtemp(prefix="smoke_stagec_")
    step(f"images {imgs}, folds {meta['folds'][imgs].tolist()}, tmp {tmp}")

    # ---- linker: pair features / labels, tiny model, grouping check
    lconf = dict(t=0.6, min_area=20, max_gap=48)
    tl._init(args.names, weights, lconf)
    parts = [r for r in (tl._pairs(i) for i in imgs) if r is not None]
    x = np.concatenate([q[0] for q in parts]); y = np.concatenate([q[1] for q in parts])
    w = np.concatenate([q[2] for q in parts])
    step(f"linker pairs: {len(y)} edges, {int((w > 0).sum())} labelled, features {x.shape[1]}")
    assert len(y) > 0 and np.isfinite(y).all()
    lab = w > 0
    booster = tl.fit(x[lab] if lab.sum() > 5 else x, y[lab] if lab.sum() > 5 else y,
                     w[lab] if lab.sum() > 5 else np.ones(len(y)), rounds=20)
    linker = Linker({k: [booster] for k in range(5)}, [booster], 0.6, 20, 48)
    lpath = os.path.join(tmp, "linker.pkl")
    linker.save(lpath)
    Linker.load(lpath)
    rows = []
    for i in imgs[:3]:
        el = tl._graph(i)[-1]
        rows += tl._grouping((i, np.random.RandomState(i).rand(len(el))))
    step(f"grouping variants evaluated: {sorted({r[0] for r in rows})}")

    # ---- pool with the linker families
    pconf = dict(names=args.names, weights=weights, families=bp.DEFAULT_FAMILIES, link_taus=[0.4, 0.6],
                 linker=lpath, min_area=100)
    pconf["columns"] = bp.family_names(pconf["families"], pconf["link_taus"])
    out = os.path.join(tmp, "pool", "oof")
    os.makedirs(os.path.join(out, "masks"))
    bp._init(args.names, weights, "oof", lpath, pconf, out)
    recs_all, oracles = [], []
    for i in imgs:
        recs, oracle = bp._image(i)
        recs_all += recs
        oracles.append(oracle)
    df = pd.DataFrame(recs_all)
    step(f"pool: {len(df)} candidates; per family {df['family'].value_counts().sort_index().to_dict()}; "
         f"stability mean {df['stability'].mean():.2f}")
    cols = [c for c in feature_columns(pconf["families"], pconf["link_taus"]) if c in df.columns]
    missing = [c for c in feature_columns(pconf["families"], pconf["link_taus"]) if c not in df.columns]
    assert not missing, f"features missing from the pool table: {missing}"
    assert df["u"].between(0, 1).all() and (df["family"].nunique() >= 6)
    df.to_parquet(os.path.join(out, "features.parquet"))
    with open(os.path.join(out, "pool_config.json"), "w") as f:
        json.dump(pconf, f)

    # ---- scorer: nested fit + selection + PQ + checks
    x = df[cols].to_numpy(np.float64)
    u, wt, folds = df["u"].to_numpy(), df["K"].to_numpy(np.float64), df["fold"].to_numpy()
    ts.PARAMS.update(min_data_in_leaf=3, num_leaves=4)
    oof = ts.nested(x, u, wt, folds, cols, rounds=10, seeds=1)
    assert (oof > -1).all() or len(np.unique(folds)) < len(np.unique(folds))  # every row scored
    step(f"scorer nested scores: mean {oof[oof > -1].mean():.3f}")
    ts._init(os.path.join(out, "masks"), [(0.2, 0.1), (0.3, 0.2)], 30)
    res = []
    for i, g in df.groupby("image"):
        res.append(ts._evaluate((int(i), {"pool": (g["cand"].to_numpy(), oof[g.index.to_numpy()])})))
    pq_sel = sum(r["pool"][(0.2, 0.1)].tp for r in res)
    step(f"selection ran on {len(res)} images (TPs at tau 0.2: {pq_sel})")

    # ---- submission writer and format checks
    conf = dict(pool=os.path.join(tmp, "pool"), tau=0.2, rho=0.1, min_keep=30, **{k: pconf[k] for k in pconf})
    scores = {int(i): g.sort_values("cand").assign(s=oof[g.index.to_numpy()])["s"].to_numpy()
              for i, g in df.groupby("image")}
    mv3._init(conf, "oof", scores)
    rows = [r for i in imgs for r in mv3._image((i, meta["stems"][i]))]
    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    validate_submission(sub)
    step(f"submission rows {len(sub)} over {sub['filament_id'].str.rsplit('_', n=1).str[0].nunique()} images; format OK")
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
