"""v3: build the extra training targets (soft skeleton T, soft core D, per-set instance maps).

  python src/prepare_targets_v3.py --sizes 1024 1536 2048 --inst-size 1024

Writes to $FILSEG_WORK/cache/ (uint8 = value x 255):
  train_soft_<S>.npy   uint8 (N, S, S)        soft mask y (only if missing; same as prepare_data.py)
  train_skel_<S>.npy   uint8 (N, S, S)        soft tubed skeleton T
  train_core_<S>.npy   uint8 (N, S, S)        soft core D
  train_inst_<L>.npy   int16 (N, 3, L, L)     instance label map of each annotation set (slot k >= n_sets: zeros)

Everything is derived from the ``segmentation`` polygons only (see filseg/targets_v3.py). Uses the image
order of train_meta.pkl, so run prepare_data.py first. About 5-10 min on 16 processes.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

from filseg.data import CACHE_DIR, load_train_annotations
from filseg.targets_v3 import area_resize, downsample_labels, image_targets, to_uint8

MAX_SETS = 3
_G = {}


def _init(by_stem, sizes, inst_size, need_soft):
    _G.update(by_stem=by_stem, sizes=sizes, inst_size=inst_size, need_soft=need_soft)


def _work(args):
    i, stem = args
    y, t, d, labs = image_targets(_G["by_stem"][stem])
    out = {}
    for s in _G["sizes"]:
        out[("skel", s)] = to_uint8(area_resize(t, s))
        out[("core", s)] = to_uint8(area_resize(d, s))
        if s in _G["need_soft"]:
            out[("soft", s)] = to_uint8(area_resize(y, s))
    inst = np.zeros((MAX_SETS, _G["inst_size"], _G["inst_size"]), np.int16)
    for k, lab in enumerate(labs[:MAX_SETS]):
        inst[k] = downsample_labels(lab, _G["inst_size"])
    out["inst"] = inst
    return i, out, int(max((lab.max() for lab in labs), default=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 1536, 2048])
    ap.add_argument("--inst-size", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stems = meta["stems"]
    _, by_stem = load_train_annotations()
    assert max(len(by_stem[s]) for s in stems) <= MAX_SETS
    need_soft = [s for s in args.sizes if not os.path.exists(os.path.join(CACHE_DIR, f"train_soft_{s}.npy"))]
    n = len(stems)
    arrays = {}
    for s in args.sizes:
        for kind in ["skel", "core"] + (["soft"] if s in need_soft else []):
            path = os.path.join(CACHE_DIR, f"train_{kind}_{s}.npy")
            arrays[(kind, s)] = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(n, s, s))
    inst_path = os.path.join(CACHE_DIR, f"train_inst_{args.inst_size}.npy")
    arrays["inst"] = np.lib.format.open_memmap(inst_path, mode="w+", dtype=np.int16,
                                               shape=(n, MAX_SETS, args.inst_size, args.inst_size))
    t0 = time.time()
    max_inst = 0
    with Pool(args.workers, initializer=_init, initargs=(by_stem, args.sizes, args.inst_size, need_soft)) as pool:
        for k, (i, out, mi) in enumerate(pool.imap_unordered(_work, list(enumerate(stems)), chunksize=2)):
            for key, a in out.items():
                arrays[key][i] = a
            max_inst = max(max_inst, mi)
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{n} images  {time.time() - t0:.0f}s", flush=True)
    for a in arrays.values():
        a.flush()
    with open(os.path.join(CACHE_DIR, "targets_v3.pkl"), "wb") as f:
        pickle.dump({"sizes": args.sizes, "inst_size": args.inst_size, "max_instances_per_set": max_inst,
                     "max_sets": MAX_SETS}, f)
    print(f"done in {time.time() - t0:.0f}s; max instances in one set {max_inst}; wrote "
          + ", ".join(sorted(f"train_{k[0]}_{k[1]}.npy" if isinstance(k, tuple) else os.path.basename(inst_path)
                             for k in arrays)))


if __name__ == "__main__":
    main()
