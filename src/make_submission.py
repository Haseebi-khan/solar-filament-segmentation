"""Turn probability maps into the competition CSV (filament_id, segmentation_rle).

  python src/make_submission.py --names r34_1024 --config work/pp_config.json --out submissions/sub.csv

With --oof the same pipeline is applied to the out-of-fold training maps and the
resulting CSV is scored against a ground-truth CSV in the organisers' format,
which checks the whole path from probabilities to RLE strings.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from filseg.data import CACHE_DIR, WORK_ROOT, rle_to_string
from filseg.metric import PQStats, pq_stats
from filseg.postprocess import PPConfig, extract, upsample
from filseg.rescore import Rescorer


def load_blend(names: list[str], weights: list[float], split: str) -> np.ndarray:
    maps = []
    for n in names:
        path = os.path.join(WORK_ROOT, "preds", n, "oof.npy" if split == "oof" else "test_folds.npy")
        maps.append(np.load(path, mmap_mode="r"))
    w = np.asarray(weights, np.float32) / np.sum(weights)

    class Blend:
        def __len__(self):
            return len(maps[0])

        def __getitem__(self, i):
            parts = [np.asarray(m[i], np.float32) for m in maps]
            size = max(p.shape[0] for p in parts)
            return sum(wi * (p if p.shape[0] == size else upsample(p, size)) for wi, p in zip(w, parts))

    return Blend()


def predictions_to_rows(stems, probs, cfg: PPConfig, rescorer: Rescorer | None, images=None) -> list[tuple[str, str]]:
    rows = []
    for i, stem in enumerate(stems):
        p_full = upsample(probs[i])
        image = np.asarray(images[i]) if rescorer is not None else None
        rles, objs = extract(None, cfg, p_full=p_full, image=image)
        if rescorer is not None:
            keep = rescorer.select(objs)
            rles = [r for r, k in zip(rles, keep) if k]
        for k, r in enumerate(rles, start=1):
            rows.append((f"{stem}_{k}", rle_to_string(r)))
    return rows


def validate_submission(df: pd.DataFrame) -> None:
    assert list(df.columns) == ["filament_id", "segmentation_rle"]
    assert df["filament_id"].is_unique, "duplicate filament ids"
    image_ids = df["filament_id"].str.split("_", n=1).str[0]
    for img_id, g in df.groupby(image_ids):
        rles = [{"size": [2048, 2048], "counts": c} for c in g["segmentation_rle"]]
        masks = mask_util.decode(rles)
        assert masks.shape[:2] == (2048, 2048)
        assert masks.sum(axis=2).max() <= 1, f"overlapping masks in {img_id}"
        assert (masks.reshape(-1, masks.shape[2]).sum(0) > 0).all(), f"empty mask in {img_id}"


def ground_truth_frame(meta) -> pd.DataFrame:
    rows = []
    for s in meta["stems"]:
        for set_id, rles in zip(meta["set_ids"][s], meta["gt_rles"][s]):
            for k, r in enumerate(rles, start=1):
                rows.append((f"{set_id}_{k}", rle_to_string(r)))
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def score_frames(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> PQStats:
    """Same grouping logic as the organisers' notebook, on RLE strings."""
    gt_keys = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_keys = pred_df["filament_id"].str.split("_", n=1).str[0]
    preds = {k: [{"size": [2048, 2048], "counts": c.encode()} for c in g["segmentation_rle"]]
             for k, g in pred_df.groupby(pred_keys)}
    total = PQStats()
    for key, g in gt_df.groupby(gt_keys):
        image_id = key.split("-", maxsplit=1)[1]
        gt_rles = [{"size": [2048, 2048], "counts": c.encode()} for c in g["segmentation_rle"]]
        total.add(pq_stats(gt_rles, preds.get(image_id, [])))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--config", required=True, help="JSON with PPConfig fields (+ optional 'rescorer' path)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--oof", action="store_true", help="build and score an out-of-fold submission instead")
    args = ap.parse_args()
    weights = args.weights or [1.0] * len(args.names)
    with open(args.config) as f:
        conf = json.load(f)
    rescorer = Rescorer.load(conf.pop("rescorer")) if conf.get("rescorer") else None
    conf.pop("rescorer", None)
    cfg = PPConfig(**{k: v for k, v in conf.items() if k in PPConfig.__dataclass_fields__})

    split = "oof" if args.oof else "test"
    meta_name = "train_meta.pkl" if args.oof else "test_meta.pkl"
    with open(os.path.join(CACHE_DIR, meta_name), "rb") as f:
        meta = pickle.load(f)
    images = np.load(os.path.join(CACHE_DIR, f"{'train' if args.oof else 'test'}_images.npy"), mmap_mode="r")
    probs = load_blend(args.names, weights, split)
    rows = predictions_to_rows(meta["stems"], probs, cfg, rescorer, images)
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    validate_submission(df)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    n_img = df["filament_id"].str.split("_", n=1).str[0].nunique()
    print(f"wrote {args.out}: {len(df)} filaments over {n_img}/{len(meta['stems'])} images")
    if args.oof:
        print("OOF score from CSV:", score_frames(ground_truth_frame(meta), df))


if __name__ == "__main__":
    main()
