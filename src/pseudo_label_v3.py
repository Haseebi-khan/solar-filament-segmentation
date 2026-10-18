"""v3 Stage B: fold-honest soft pseudo-labels of the unlabelled GONG frames (Noisy Student teachers).

  python src/pseudo_label_v3.py --teachers m1_cnxl_2048_v3 m2_vitl_1536_v3 --folds 0 1 2 3 4

For fold k the teacher is the blend of the teachers' fold-k models (trained without fold-k labels), so a student
trained on these labels can still be evaluated out of fold on fold k. Each frame is predicted with 4 dihedral
views (identity, horizontal flip, rot90, rot270; --views to change), P and S are blended with --weights and
stored soft (uint8 x255, never thresholded) at --size:
  <unlabeled>/pl/<teachers>/fold<k>.npy     uint8 (M, 2, size, size)   channels P, S
  <unlabeled>/pl/<teachers>/fold<k>_w.npy   float32 (M,)  reliability clip(1 - 3 * mean disk std of the
                                                          teachers' P, 0.2, 1) (1 with a single teacher)
Resumable: frames already written (marked in fold<k>_done.npy) are skipped.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from filseg.data import WORK_ROOT
from filseg.infer_v3 import predict_maps
from filseg.models_v3 import load_checkpoint_v3
from filseg.postprocess import disk_geometry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--views", type=int, nargs="+", default=[0, 4, 1, 3])
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--unlabeled", default=os.environ.get("FILSEG_UNLABELED", os.path.join(WORK_ROOT, "unlabeled")))
    args = ap.parse_args()
    weights = np.asarray(args.weights or [1.0] * len(args.teachers), np.float32)
    weights = weights / weights.sum()
    images = np.load(os.path.join(args.unlabeled, "U_images.npy"), mmap_mode="r")
    n = len(images)
    out_dir = os.path.join(args.unlabeled, "pl", "+".join(args.teachers))
    os.makedirs(out_dir, exist_ok=True)
    yy, xx = np.ogrid[:args.size, :args.size]
    for fold in args.folds:
        path = os.path.join(out_dir, f"fold{fold}.npy")
        pl = (np.load(path, mmap_mode="r+") if os.path.exists(path)
              else np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(n, 2, args.size, args.size)))
        wpath, dpath = os.path.join(out_dir, f"fold{fold}_w.npy"), os.path.join(out_dir, f"fold{fold}_done.npy")
        rel = np.load(wpath) if os.path.exists(wpath) else np.ones(n, np.float32)
        done = np.load(dpath) if os.path.exists(dpath) else np.zeros(n, bool)
        models = [load_checkpoint_v3(os.path.join(WORK_ROOT, "models", t, f"fold{fold}.pt"))[0] for t in args.teachers]
        heads = [("p", "s") if m.cfg.aux_heads else ("p",) for m in models]
        t0 = time.time()
        todo = np.where(~done)[0]
        print(f"fold {fold}: {len(todo)}/{n} frames to label", flush=True)
        for c0 in range(0, len(todo), args.chunk):
            idx = todo[c0:c0 + args.chunk]
            chunk = images[idx[0]:idx[-1] + 1] if np.all(np.diff(idx) == 1) else images[idx]
            p_acc = np.zeros((len(idx), args.size, args.size), np.float32)
            s_acc = np.zeros_like(p_acc)
            s_w = 0.0
            ps = []
            for m, h, w in zip(models, heads, weights):
                maps = predict_maps([m], chunk, batch_size=args.bs, out_size=args.size, heads=h, views=args.views)
                p = maps["p"].astype(np.float32)
                ps.append(p)
                p_acc += w * p
                if "s" in maps:
                    s_acc += w * maps["s"].astype(np.float32)
                    s_w += w
            if s_w > 0:
                s_acc /= s_w
            pl[idx, 0] = np.round(p_acc * 255).clip(0, 255).astype(np.uint8)
            pl[idx, 1] = np.round(s_acc * 255).clip(0, 255).astype(np.uint8)
            if len(ps) > 1:
                sd = np.std(np.stack(ps), axis=0)
                for k, i in enumerate(idx):
                    cy, cx, r = disk_geometry(np.asarray(images[i]))
                    sc = args.size / 2048
                    disk = (yy - cy * sc) ** 2 + (xx - cx * sc) ** 2 < (0.95 * r * sc) ** 2
                    rel[i] = float(np.clip(1 - 3 * sd[k][disk].mean(), 0.2, 1.0))
            done[idx] = True
            pl.flush()
            np.save(wpath, rel)
            np.save(dpath, done)
            k = c0 + len(idx)
            print(f"  fold {fold}: {k}/{len(todo)}  {time.time() - t0:.0f}s  "
                  f"(~{(time.time() - t0) / k * (len(todo) - k) / 60:.0f} min left)", flush=True)
        del models
    print("pseudo-labels in", out_dir)


if __name__ == "__main__":
    main()
