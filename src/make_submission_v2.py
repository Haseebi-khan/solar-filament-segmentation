"""v2: probability maps -> extent hypotheses -> selector -> submission CSV.

  # test submission
  python src/make_submission_v2.py --config work/selectors_v2/<members>_config.json --out submissions/submission_v2.csv

  # out-of-fold CSV scored exactly against the training ground truth (organisers' grouping logic);
  # with --nested-scores every image is scored by a selector that never saw its fold (honest number)
  python src/make_submission_v2.py --config work/selectors_v2/<members>_config.json --oof \
      --nested-scores work/selectors_v2/<members>_nested.parquet --out work/oof_sub_v2_<members>.csv

The config is written by train_selector_v2.py (members, blend weights, extraction settings, selector).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd

from filseg.data import CACHE_DIR, rle_to_string
from filseg.hypotheses_v2 import feature_matrix_v2, hypotheses, select_and_assemble
from filseg.postprocess import upsample
from make_submission import ground_truth_frame, load_blend, score_frames, validate_submission

_G = {}


def _init(conf, split, nested_path):
    _G["probs"] = load_blend(conf["names"], conf["weights"], split)
    _G["images"] = np.load(os.path.join(CACHE_DIR, f"{'train' if split == 'oof' else 'test'}_images.npy"),
                           mmap_mode="r")
    _G["cfg"] = dict(t_seed=conf["t_seed"], link=conf["link"], min_area=conf["min_area"],
                     levels=tuple(conf["levels"]))
    _G["threshold"] = conf["threshold"]
    if nested_path:
        nd = pd.read_parquet(nested_path)
        _G["nested"] = {(int(i), int(k), round(float(t), 4)): float(s)
                        for i, k, t, s in zip(nd["image"], nd["label"], nd["level"], nd["score"])}
    else:
        import lightgbm as lgb
        with open(conf["selector"], "rb") as f:
            sel = pickle.load(f)
        _G["boosters"] = [lgb.Booster(model_str=s) for s in sel["boosters"]]


def _image_rows(args):
    i, stem = args
    p = upsample(_G["probs"][i])
    rows, labs = hypotheses(p, np.asarray(_G["images"][i]), **_G["cfg"])
    if not rows:
        return []
    if "nested" in _G:
        scores = np.array([_G["nested"][(i, r["label"], round(r["level"], 4))] for r in rows])
    else:
        x = feature_matrix_v2(rows)
        scores = np.mean([b.predict(x, num_threads=1) for b in _G["boosters"]], axis=0)
    rles = select_and_assemble(rows, scores, _G["threshold"], labs)
    return [(f"{stem}_{k}", rle_to_string(r)) for k, r in enumerate(rles, start=1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--oof", action="store_true", help="build and score an out-of-fold CSV for the training images")
    ap.add_argument("--nested-scores", default=None, help="with --oof: use the nested out-of-fold selector scores")
    ap.add_argument("--workers", type=int, default=18)
    args = ap.parse_args()
    with open(args.config) as f:
        conf = json.load(f)
    if args.nested_scores and not args.oof:
        ap.error("--nested-scores only applies to --oof")
    split = "oof" if args.oof else "test"
    with open(os.path.join(CACHE_DIR, "train_meta.pkl" if args.oof else "test_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    with Pool(args.workers, initializer=_init, initargs=(conf, split, args.nested_scores)) as pool:
        parts = pool.map(_image_rows, list(enumerate(meta["stems"])), chunksize=2)
    df = pd.DataFrame([r for rows in parts for r in rows], columns=["filament_id", "segmentation_rle"])
    validate_submission(df)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    n_img = df["filament_id"].str.split("_", n=1).str[0].nunique()
    print(f"wrote {args.out}: {len(df)} filaments over {n_img}/{len(meta['stems'])} images")
    if args.oof:
        kind = "nested selector (honest)" if args.nested_scores else "selector fit on these images (optimistic)"
        print(f"OOF score from CSV, {kind}:", score_frames(ground_truth_frame(meta), df))


if __name__ == "__main__":
    main()
