"""Decode images once and build training targets.

Outputs (in $FILSEG_WORK/cache):
  train_images.npy      uint8 (N, 2048, 2048)  full-resolution H-alpha images
  test_images.npy       uint8 (M, 2048, 2048)
  train_soft_<S>.npy    uint8 (N, S, S)        fraction of annotators marking each pixel, x255
  train_meta.pkl        stems, folds, number of annotation sets, GT RLEs per set
  test_meta.pkl         test stems
"""
from __future__ import annotations

import argparse
import os
import pickle
from multiprocessing import Pool

import cv2
import numpy as np
from pycocotools import mask as mask_util

from filseg.data import (CACHE_DIR, TEST_IMG_DIR, TRAIN_IMG_DIR, assign_folds, cache_images,
                         list_test_stems, load_train_annotations)

_BY_STEM = None


def _soft_target(stem: str, sizes: list[int]) -> dict[int, np.ndarray]:
    sets = _BY_STEM[stem]
    acc = np.zeros((2048, 2048), np.float32)
    for s in sets:
        rles = s.rles()
        if rles:
            acc += mask_util.decode(rles).max(axis=2)
    acc /= len(sets)
    return {size: np.round(cv2.resize(acc, (size, size), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
            for size in sizes}


def main():
    global _BY_STEM
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 1536])
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    stems, by_stem = load_train_annotations()
    _BY_STEM = by_stem
    test_stems = list_test_stems()
    print(f"train images {len(stems)}  annotation sets {sum(len(v) for v in by_stem.values())}  test images {len(test_stems)}")

    cache_images(stems, TRAIN_IMG_DIR, "train")
    cache_images(test_stems, TEST_IMG_DIR, "test")
    print("images cached")

    with Pool(16) as pool:
        targets = pool.starmap(_soft_target, [(s, args.sizes) for s in stems])
    for size in args.sizes:
        np.save(os.path.join(CACHE_DIR, f"train_soft_{size}.npy"), np.stack([t[size] for t in targets]))
    print("soft targets written")

    folds = assign_folds(stems, args.folds)
    meta = {
        "stems": stems,
        "folds": np.array([folds[s] for s in stems]),
        "n_sets": np.array([len(by_stem[s]) for s in stems]),
        "set_ids": {s: [a.set_id for a in by_stem[s]] for s in stems},
        "gt_rles": {s: [a.rles() for a in by_stem[s]] for s in stems},
    }
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    with open(os.path.join(CACHE_DIR, "test_meta.pkl"), "wb") as f:
        pickle.dump({"stems": test_stems}, f)
    print("meta written; fold sizes", np.bincount(meta["folds"]))


if __name__ == "__main__":
    main()
