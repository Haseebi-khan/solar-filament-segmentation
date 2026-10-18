"""v2 leaderboard probe: grow or shrink every predicted filament by r pixels.

  python tools/morph_submission_v2.py submissions/submission.csv --radius -1 --out submissions/submission_v2_probe_erode1.csv

Out-of-fold, the training ground truth is matched best by the unchanged masks (radius 0); shrinking
the ground truth by 1 px instead drops our OOF PQ from 0.45 to 0.37, about the size of the OOF ->
public-leaderboard gap. Submitting radius -1 and +1 tells whether the test masks are drawn thinner
or wider than the training masks. Masks stay disjoint: dilation only claims background pixels that
exactly one instance reaches.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from make_submission import validate_submission  # noqa: E402

H = W = 2048


def morph_image(counts: list[str], r: int) -> list[str]:
    rles = [{"size": [H, W], "counts": c.encode()} for c in counts]
    masks = mask_util.decode(rles)
    if r == 0:
        return counts
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(r) + 1, 2 * abs(r) + 1))
    lab = np.zeros((H, W), np.int32)
    for i in range(masks.shape[2]):
        lab[masks[:, :, i] > 0] = i + 1
    grown = [(cv2.dilate if r > 0 else cv2.erode)(masks[:, :, i], k).astype(bool) for i in range(masks.shape[2])]
    if r > 0:
        reach = np.sum(grown, axis=0)
        grown = [g & ((lab == i + 1) | ((lab == 0) & (reach == 1))) for i, g in enumerate(grown)]
    out = []
    for g in grown:
        if g.any():
            c = mask_util.encode(np.asfortranarray(g.astype(np.uint8)))["counts"]
            out.append(c.decode() if isinstance(c, bytes) else c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--radius", type=int, required=True, help="+r dilates, -r erodes (pixels)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    stems = df["filament_id"].str.rsplit("_", n=1).str[0]
    rows = []
    for stem, g in df.groupby(stems, sort=False):
        for k, c in enumerate(morph_image(list(g["segmentation_rle"]), args.radius), start=1):
            rows.append((f"{stem}_{k}", c))
    out = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    validate_submission(out)
    out.to_csv(args.out, index=False)
    a0 = sum(int(mask_util.area({"size": [H, W], "counts": c.encode()})) for c in df["segmentation_rle"])
    a1 = sum(int(mask_util.area({"size": [H, W], "counts": c.encode()})) for c in out["segmentation_rle"])
    print(f"wrote {args.out}: {len(out)} filaments (was {len(df)}), total area x{a1 / a0:.3f}")


if __name__ == "__main__":
    main()
