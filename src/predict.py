"""Predict probability maps with trained fold models.

For each fold model: out-of-fold maps for its validation images, plus test maps.
Writes to $FILSEG_WORK/preds/<name>/:
  oof.npy   float16 (N_train, S, S)  each image predicted by the model that did not see it
  test.npy  float16 (N_test, S, S)   mean over fold models

Example:
  python src/predict.py --name r34_1024 --folds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

from filseg.data import CACHE_DIR, WORK_ROOT
from filseg.infer import predict_probs
from filseg.model import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--tag", default="", help="suffix for output dir, e.g. to compare TTA settings")
    ap.add_argument("--out-size", type=int, default=0, help="resolution of saved maps (0 = network size)")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--full-model", action="store_true", help="also predict test with the all-data model full.pt")
    args = ap.parse_args()

    model_dir = os.path.join(WORK_ROOT, "models", args.name)
    out_dir = os.path.join(WORK_ROOT, "preds", args.name + args.tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    folds = meta["folds"]
    train_imgs = np.load(os.path.join(CACHE_DIR, "train_images.npy"), mmap_mode="r")
    test_imgs = np.load(os.path.join(CACHE_DIR, "test_images.npy"), mmap_mode="r")

    oof_path = os.path.join(out_dir, "oof.npy")
    oof = None
    for fold in args.folds:
        model, cfg = load_checkpoint(os.path.join(model_dir, f"fold{fold}.pt"))
        size = cfg["size"]
        out_size = args.out_size or size
        if oof is None:
            oof = np.load(oof_path) if os.path.exists(oof_path) else np.zeros((len(folds), out_size, out_size), np.float16)
        va = np.where(folds == fold)[0]
        oof[va] = predict_probs([model], train_imgs[va], net_size=size, tta=args.tta, out_size=out_size,
                                scales=tuple(args.scales))
        np.save(oof_path, oof)
        print(f"fold {fold}: oof done ({len(va)} images)", flush=True)
        if not args.no_test:
            p = predict_probs([model], test_imgs, net_size=size, tta=args.tta, out_size=out_size,
                              scales=tuple(args.scales))
            np.save(os.path.join(out_dir, f"test_fold{fold}.npy"), p)
    # test maps: mean over every fold model predicted so far
    parts = [os.path.join(out_dir, f"test_fold{k}.npy") for k in sorted(set(folds.tolist()))]
    parts = [p for p in parts if os.path.exists(p)]
    if parts:
        mean = sum(np.load(p).astype(np.float32) for p in parts) / len(parts)
        np.save(os.path.join(out_dir, "test_folds.npy"), mean.astype(np.float16))
        print(f"test_folds.npy = mean of {len(parts)} fold models")
    if args.full_model:
        model, cfg = load_checkpoint(os.path.join(model_dir, "full.pt"))
        p = predict_probs([model], test_imgs, net_size=cfg["size"], tta=args.tta, out_size=args.out_size or cfg["size"])
        np.save(os.path.join(out_dir, "test_full.npy"), p)
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
