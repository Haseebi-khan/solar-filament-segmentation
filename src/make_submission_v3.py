"""v3: candidate pool -> stacked scorer -> overlap-aware selection -> submission CSV.

  # test submission (needs build_pool_v3.py --split test with the same members/families/linker first)
  python src/make_submission_v3.py --config work/pools_v3/<tag>/scorer/scorer_config.json \
      --out submissions/submission_v3.csv
  # out-of-fold CSV with the nested scores (honest), scored with the organisers' grouping logic
  python src/make_submission_v3.py --config .../scorer_config.json --oof --out work/oof_sub_v3_<tag>.csv

Checks before writing (blueprint 12.7): RLE decodes to 2048x2048, masks of one image are disjoint, no empty
mask, ids unique and prefixed by the image stem, counts written without quotes.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from filseg.data import CACHE_DIR, rle_to_string
from filseg.masks_v3 import CropMask
from filseg.select_v3 import greedy_select
from make_submission import ground_truth_frame, score_frames, validate_submission

_G = {}


def _init(conf, split, scores):
    _G.update(conf=conf, mask_dir=os.path.join(conf["pool"], split, "masks"), scores=scores)


def _image(args):
    i, stem = args
    sc = _G["scores"].get(i)
    path = os.path.join(_G["mask_dir"], f"{i}.pkl")
    if sc is None or not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        masks = [CropMask.unpack(t) for t in pickle.load(f)]
    c = _G["conf"]
    sel, _ = greedy_select(masks, sc, c["tau"], c["rho"], c["min_keep"])
    return [(f"{stem}_{k}", rle_to_string(m.to_rle())) for k, m in enumerate(sel, start=1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--oof", action="store_true", help="out-of-fold CSV from the nested scores (honest)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    with open(args.config) as f:
        conf = json.load(f)
    split = "oof" if args.oof else "test"
    with open(os.path.join(CACHE_DIR, "train_meta.pkl" if args.oof else "test_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    with open(os.path.join(conf["pool"], split, "pool_config.json")) as f:
        pconf = json.load(f)
    for k in ("names", "weights", "families", "link_taus", "min_area"):
        if pconf[k] != conf[k]:
            raise SystemExit(f"{split} pool was built with {k}={pconf[k]}, the scorer with {conf[k]}: rebuild it")
    df = pd.read_parquet(os.path.join(conf["pool"], split, "features.parquet"))
    if args.oof:
        nd = pd.read_parquet(os.path.join(os.path.dirname(conf["scorer"]), "nested_scores.parquet"))
        df = df.merge(nd, on=["image", "cand"], how="left")
        assert df["score"].notna().all()
    else:
        import lightgbm as lgb
        with open(conf["scorer"], "rb") as f:
            sc = pickle.load(f)
        x = df[sc["features"]].to_numpy(np.float64)
        df["score"] = np.mean([lgb.Booster(model_str=s).predict(x) for s in sc["boosters"]], axis=0) if len(df) else []
    scores = {int(i): g.sort_values("cand")["score"].to_numpy() for i, g in df.groupby("image")}
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(conf, split, scores)) as pool:
        parts = pool.map(_image, list(enumerate(meta["stems"])), chunksize=2)
    print(f"selection: {time.time() - t0:.0f} s for {len(meta['stems'])} images")
    sub = pd.DataFrame([r for rows in parts for r in rows], columns=["filament_id", "segmentation_rle"])
    validate_submission(sub)
    known = set(meta["stems"])
    assert all(fid.rsplit("_", 1)[0] in known for fid in sub["filament_id"]), "id prefix is not an image stem"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sub.to_csv(args.out, index=False)
    with open(args.out) as f:
        assert '"' not in f.read(), "quoted fields in the CSV"
    n_img = sub["filament_id"].str.split("_", n=1).str[0].nunique()
    print(f"wrote {args.out}: {len(sub)} filaments over {n_img}/{len(meta['stems'])} images "
          f"(tau {conf['tau']}, rho {conf['rho']})")
    if args.oof:
        print("OOF score from CSV, nested scorer (honest):", score_frames(ground_truth_frame(meta), sub))


if __name__ == "__main__":
    main()
