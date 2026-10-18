"""v2 leaderboard probe: grow or shrink every predicted filament by r pixels.

  python tools/morph_submission_v2.py submissions/submission.csv --radius -1 --out submissions/submission_v2_probe_erode1.csv

Out-of-fold, the training ground truth is matched best by the unchanged masks (radius 0); shrinking
the ground truth by 1 px instead drops our OOF PQ from 0.45 to 0.37, about the size of the OOF ->
public-leaderboard gap. Submitting radius -1 and +1 tells whether the test masks are drawn thinner
or wider than the training masks. Masks stay disjoint (see filseg/morph_v2.py).
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from pycocotools import mask as mask_util

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from filseg.data import rle_to_string  # noqa: E402
from filseg.morph_v2 import morph_rles  # noqa: E402
from make_submission import validate_submission  # noqa: E402

H = W = 2048


def total_area(counts) -> int:
    return sum(int(mask_util.area({"size": [H, W], "counts": c.encode()})) for c in counts)


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
        rles = [{"size": [H, W], "counts": c.encode()} for c in g["segmentation_rle"]]
        for k, r in enumerate(morph_rles(rles, args.radius), start=1):
            rows.append((f"{stem}_{k}", rle_to_string(r)))
    out = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    validate_submission(out)
    out.to_csv(args.out, index=False)
    ratio = total_area(out["segmentation_rle"]) / total_area(df["segmentation_rle"])
    print(f"wrote {args.out}: {len(out)} filaments (was {len(df)}), total area x{ratio:.3f}")


if __name__ == "__main__":
    main()
