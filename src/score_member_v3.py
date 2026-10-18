"""v3: quick, comparable out-of-fold score of one member (or blend) for ablations and gates.

  python src/score_member_v3.py --names abl_jml --out work/ablations_v3/abl_jml.csv

v1 post-processing (seeds and extent at the same threshold t, fragments within ~16 px grouped, holes filled,
>= 100 px) with t chosen from --thresholds by out-of-fold PQ (one parameter for every variant, so differently
calibrated losses are compared fairly). Writes the out-of-fold CSV of the best t (input to compare_v3.py) and
prints PQ / SQ / RQ for every t.
"""
from __future__ import annotations

import argparse
import os
import pickle
from multiprocessing import Pool

import pandas as pd

from filseg.data import CACHE_DIR, rle_to_string
from filseg.metric import PQStats, pq_stats
from filseg.postprocess import PPConfig, extract
from make_submission import load_blend, validate_submission

_G = {}


def _init(names, weights, thresholds):
    _G["probs"] = load_blend(names, weights, "oof")
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        _G["meta"] = pickle.load(f)
    _G["thr"] = thresholds


def _image(i):
    meta = _G["meta"]
    stem = meta["stems"][i]
    p = _G["probs"][i]
    out = {}
    for t in _G["thr"]:
        rles, _ = extract(p, PPConfig(t_seed=t, t_mask=t, min_area=100, link=8, close=-1))
        st = PQStats()
        for g in meta["gt_rles"][stem]:
            st.add(pq_stats(g, rles))
        out[t] = (st, [(f"{stem}_{k}", rle_to_string(r)) for k, r in enumerate(rles, start=1)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    weights = args.weights or [1.0] * len(args.names)
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        n = len(pickle.load(f)["stems"])
    with Pool(args.workers, initializer=_init, initargs=(args.names, weights, args.thresholds)) as pool:
        res = pool.map(_image, range(n), chunksize=4)
    best = None
    for t in args.thresholds:
        tot = PQStats()
        for r in res:
            tot.add(r[t][0])
        print(f"{'+'.join(args.names)}  t={t}: {tot}")
        if best is None or tot.pq > best[1].pq:
            best = (t, tot)
    t = best[0]
    df = pd.DataFrame([row for r in res for row in r[t][1]], columns=["filament_id", "segmentation_rle"])
    validate_submission(df)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"best t={t}: {best[1]}  -> {args.out}")


if __name__ == "__main__":
    main()
