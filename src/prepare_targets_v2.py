"""Build soft training targets at extra resolutions without touching the image cache or train_meta.pkl.

  python src/prepare_targets_v2.py --sizes 2048

Writes $FILSEG_WORK/cache/train_soft_<S>.npy exactly like prepare_data.py (fraction of annotators
covering each pixel, x255, area-resized to S).
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import numpy as np

import prepare_data
from filseg.data import CACHE_DIR, load_train_annotations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    args = ap.parse_args()
    stems, by_stem = load_train_annotations()
    prepare_data._BY_STEM = by_stem
    with Pool(16) as pool:
        targets = pool.starmap(prepare_data._soft_target, [(s, args.sizes) for s in stems])
    for size in args.sizes:
        path = os.path.join(CACHE_DIR, f"train_soft_{size}.npy")
        np.save(path, np.stack([t[size] for t in targets]))
        print("wrote", path)


if __name__ == "__main__":
    main()
