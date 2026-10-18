"""v3 Stage B: quality-filter the downloaded GONG frames and pack the accepted ones into one uint8 array.

  python src/gong_qc_v3.py [--unlabeled work/unlabeled] [--max-frames 8000]

Reference statistics come from the 707 training images. A frame is rejected if any holds (blueprint 7.4):
  disk mean or disk std outside the training 1st-99th percentile;
  variance of the Laplacian on the disk below the training 1st percentile (blur / bad seeing);
  std of the 64-px-blurred disk above the training 99th percentile (clouds / haze);
  more than 0.5 % saturated (255) pixels on the disk;
  disk centre or radius (filseg.postprocess.disk_geometry) outside the training range +- 4 px.
If <unlabeled>/lut_v3.json says "apply", the site's intensity LUT is applied first.

Writes <unlabeled>/U_images.npy (M, 2048, 2048) uint8, U_meta.csv (stem, site, time, statistics) and
qc_report.txt with the rejection counts per rule.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from filseg.data import CACHE_DIR, WORK_ROOT
from filseg.postprocess import disk_geometry

_G = {}


def frame_stats(img: np.ndarray) -> dict:
    cy, cx, r = disk_geometry(img)
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 < (0.95 * r) ** 2
    f = img.astype(np.float32)
    lap = cv2.Laplacian(f, cv2.CV_32F, ksize=3)
    blur = cv2.blur(f, (64, 64))
    return dict(mean=float(f[disk].mean()), std=float(f[disk].std()), lapvar=float(lap[disk].var()),
                haze=float(blur[disk].std()), sat=float((img[disk] >= 255).mean()), cy=cy, cx=cx, r=r)


def _load(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (2048, 2048):
        return None
    lut = _G.get("lut")
    if lut is not None:
        site = os.path.basename(path)[14]
        table = lut.get(site, lut["global"])
        img = table[img]
    return img


def _stats(path):
    img = _load(path)
    return None if img is None else frame_stats(img)


def _train_stats(i):
    return frame_stats(np.asarray(_G["train"][i]))


def _init(lut_path, train_path):
    if lut_path and os.path.exists(lut_path):
        with open(lut_path) as f:
            d = json.load(f)
        if d.get("apply"):
            _G["lut"] = {k: np.round(np.asarray(v)).clip(0, 255).astype(np.uint8) for k, v in d["luts"].items()}
    if train_path:
        _G["train"] = np.load(train_path, mmap_mode="r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlabeled", default=os.environ.get("FILSEG_UNLABELED", os.path.join(WORK_ROOT, "unlabeled")))
    ap.add_argument("--max-frames", type=int, default=0, help="keep at most this many accepted frames (0 = all)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.unlabeled, "raw", "*", "*.jpg")))
    lut_path = os.path.join(args.unlabeled, "lut_v3.json")
    train_path = os.path.join(CACHE_DIR, "train_images.npy")
    n_train = len(np.load(train_path, mmap_mode="r"))
    with Pool(args.workers, initializer=_init, initargs=(lut_path, train_path)) as pool:
        ref = pd.DataFrame(pool.map(_train_stats, range(n_train), chunksize=8))
        st = pool.map(_stats, files, chunksize=8)
    rows = []
    for path, s in zip(files, st):
        stem = os.path.splitext(os.path.basename(path))[0]
        rows.append(dict(stem=stem, path=path, site=stem[14], ok_read=s is not None, **(s or {})))
    df = pd.DataFrame(rows)
    lo, hi = ref.quantile(0.01), ref.quantile(0.99)
    rules = {
        "unreadable": ~df["ok_read"],
        "disk mean": ~df["mean"].between(lo["mean"], hi["mean"]),
        "disk std": ~df["std"].between(lo["std"], hi["std"]),
        "blur (Laplacian var)": df["lapvar"] < lo["lapvar"],
        "haze (64px-blur std)": df["haze"] > hi["haze"],
        "saturation > 0.5%": df["sat"] > 0.005,
        "geometry": ~(df["cy"].between(ref["cy"].min() - 4, ref["cy"].max() + 4)
                      & df["cx"].between(ref["cx"].min() - 4, ref["cx"].max() + 4)
                      & df["r"].between(ref["r"].min() - 4, ref["r"].max() + 4)),
    }
    bad = np.zeros(len(df), bool)
    report = [f"{len(df)} downloaded frames; training reference from {n_train} images"]
    for name, m in rules.items():
        m = m.fillna(True).to_numpy()
        report.append(f"  rejected by {name}: {int(m.sum())}")
        bad |= m
    acc = df[~bad].copy()
    if args.max_frames and len(acc) > args.max_frames:
        acc = acc.sample(args.max_frames, random_state=0).sort_values("stem")
    acc["time"] = pd.to_datetime(acc["stem"].str[:14], format="%Y%m%d%H%M%S")
    report.append(f"accepted {len(acc)} frames; per site {acc['site'].value_counts().sort_index().to_dict()}")
    print("\n".join(report), flush=True)
    out = np.lib.format.open_memmap(os.path.join(args.unlabeled, "U_images.npy"), mode="w+", dtype=np.uint8,
                                    shape=(len(acc), 2048, 2048))
    with Pool(args.workers, initializer=_init, initargs=(lut_path, "")) as pool:
        for k, img in enumerate(pool.imap(_load, acc["path"].tolist(), chunksize=8)):
            out[k] = img
    out.flush()
    acc.drop(columns=["path", "ok_read"]).to_csv(os.path.join(args.unlabeled, "U_meta.csv"), index=False)
    with open(os.path.join(args.unlabeled, "qc_report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print("wrote", os.path.join(args.unlabeled, "U_images.npy"), out.shape)


if __name__ == "__main__":
    main()
