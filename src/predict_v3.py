"""v3: probability maps (P, and S/D if the member has them) of trained FilaNet fold models.

  python src/predict_v3.py --name m1_cnxl_2048_v3 --folds 0 1 2 3 4 --tta 8

Writes to $FILSEG_WORK/preds/<name>/ in the v1 layout, so every v1/v2 tool (tune_postprocess.py,
train_selector_v2.py, final_ensemble_v2.sh, ...) can use v3 members directly:
  oof.npy                float16 (N_train, S, S)   P of each image by the fold model that did not see it
  test_fold<k>.npy       float16 (N_test, S, S)    P of fold model k on the test images
  test_folds.npy         float16                   mean over fold models
  oof_s.npy, oof_d.npy, test_fold<k>_s.npy, test_folds_s.npy, ...   uint8 x255 (S and D heads)
S is the member's output resolution (2048 for members with the HR stem). Finished folds are kept when the
script is rerun with other --folds (the out-of-fold files are updated in place).
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np

from filseg.data import CACHE_DIR, WORK_ROOT
from filseg.infer_v3 import predict_maps
from filseg.models_v3 import load_checkpoint_v3


def _open(path: str, shape: tuple, dtype) -> np.ndarray:
    if os.path.exists(path):
        a = np.load(path, mmap_mode="r+")
        if a.shape == shape and a.dtype == dtype:
            return a
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _store(arr: np.ndarray, head: str) -> np.ndarray:
    return arr if head == "p" else np.round(arr.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)


def _suffix(head: str) -> str:
    return "" if head == "p" else f"_{head}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--no-aux", action="store_true", help="only the P head")
    args = ap.parse_args()

    model_dir = os.path.join(WORK_ROOT, "models", args.name)
    out_dir = os.path.join(WORK_ROOT, "preds", args.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        folds = pickle.load(f)["folds"]
    train_imgs = np.load(os.path.join(CACHE_DIR, "train_images.npy"), mmap_mode="r")
    test_imgs = np.load(os.path.join(CACHE_DIR, "test_images.npy"), mmap_mode="r")

    heads = None
    for fold in args.folds:
        t0 = time.time()
        model, ck = load_checkpoint_v3(os.path.join(model_dir, f"fold{fold}.pt"))
        if heads is None:
            heads = ("p",) if (args.no_aux or not ck["model_cfg"]["aux_heads"]) else ("p", "s", "d")
        size = model.out_size
        va = np.where(folds == fold)[0]
        maps = predict_maps([model], train_imgs[va], tta=args.tta, batch_size=args.bs, heads=heads)
        for h in heads:
            dtype = np.float16 if h == "p" else np.uint8
            oof = _open(os.path.join(out_dir, f"oof{_suffix(h)}.npy"), (len(folds), size, size), np.dtype(dtype))
            oof[va] = _store(maps[h], h)
            oof.flush()
            del oof
        print(f"fold {fold}: oof done ({len(va)} images, {time.time() - t0:.0f}s)", flush=True)
        if not args.no_test:
            maps = predict_maps([model], test_imgs, tta=args.tta, batch_size=args.bs, heads=heads)
            for h in heads:
                np.save(os.path.join(out_dir, f"test_fold{fold}{_suffix(h)}.npy"), _store(maps[h], h))
            print(f"fold {fold}: test done ({time.time() - t0:.0f}s)", flush=True)
        del model
    for h in heads or ("p",):
        parts = [os.path.join(out_dir, f"test_fold{k}{_suffix(h)}.npy") for k in sorted(set(folds.tolist()))]
        parts = [p for p in parts if os.path.exists(p)]
        if parts:
            mean = sum(np.load(p).astype(np.float32) for p in parts) / len(parts)
            mean = mean.astype(np.float16) if h == "p" else np.round(mean).clip(0, 255).astype(np.uint8)
            np.save(os.path.join(out_dir, f"test_folds{_suffix(h)}.npy"), mean)
            print(f"test_folds{_suffix(h)}.npy = mean of {len(parts)} fold models")
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
