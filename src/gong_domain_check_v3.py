"""v3 Stage B: are archive GONG frames in-domain for the competition images? (blueprint 7.4)

  python src/gong_download_v3.py train-stamps --n 30      # archive copies of 30 training images
  python src/gong_domain_check_v3.py

For every training image whose archive frame (same timestamp and site) was downloaded: MAE, Pearson r, mean and
std on the solar disk. A monotone 256-entry intensity map (LUT) archive -> competition is fitted from the paired
pixels (median competition value per archive value, made non-decreasing) per site (global if a site has < 3
pairs) on half of the pairs and evaluated on the other half. Writes <unlabeled>/lut_v3.json with
"apply": true when the LUT clearly helps (held-out MAE down by > 10 %) and <unlabeled>/domain_check.md. The QC
step (gong_qc_v3.py) applies the LUT to every unlabelled frame when "apply" is true.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import date

import cv2
import numpy as np

from filseg.data import TRAIN_IMG_DIR, WORK_ROOT
from filseg.postprocess import disk_geometry


def disk_mask(img: np.ndarray, frac: float = 0.97) -> np.ndarray:
    cy, cx, r = disk_geometry(img)
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 < (frac * r) ** 2


def fit_lut(pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    a = np.concatenate([p[0] for p in pairs]).astype(np.int64)
    b = np.concatenate([p[1] for p in pairs]).astype(np.float64)
    lut = np.full(256, np.nan)
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    edges = np.searchsorted(a, np.arange(257))
    for v in range(256):
        if edges[v + 1] - edges[v] >= 20:
            lut[v] = np.median(b[edges[v]:edges[v + 1]])
    ok = ~np.isnan(lut)
    if ok.sum() < 2:
        return np.arange(256, dtype=np.float64)
    lut = np.interp(np.arange(256), np.nonzero(ok)[0], lut[ok])
    return np.maximum.accumulate(lut).clip(0, 255)


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return {"mae": float(np.abs(a - b).mean()), "r": float(np.corrcoef(a, b)[0, 1]),
            "mean_diff": float(a.mean() - b.mean()), "std_ratio": float(a.std() / max(b.std(), 1e-9))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlabeled", default=os.environ.get("FILSEG_UNLABELED", os.path.join(WORK_ROOT, "unlabeled")))
    ap.add_argument("--sample", type=int, default=200000, help="pixels per image used for the LUT fit")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.unlabeled, "domain", "raw", "*", "*.jpg")))
    if not files:
        raise SystemExit("no archive frames: run  python src/gong_download_v3.py train-stamps  first")
    rng = np.random.default_rng(0)
    recs = []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        comp = cv2.imread(os.path.join(TRAIN_IMG_DIR, stem + ".jpeg"), cv2.IMREAD_GRAYSCALE)
        arch = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if comp is None or arch is None or comp.shape != arch.shape:
            print("skip", stem)
            continue
        m = disk_mask(comp)
        idx = np.flatnonzero(m.ravel())
        idx = rng.choice(idx, min(args.sample, len(idx)), replace=False)
        recs.append(dict(stem=stem, site=stem[14], a=arch.ravel()[idx], c=comp.ravel()[idx],
                         before=stats(arch[m], comp[m])))
    print(f"{len(recs)} paired frames")
    sites = sorted({r["site"] for r in recs})
    half = {r["stem"]: k % 2 for k, r in enumerate(recs)}
    luts = {"global": fit_lut([(r["a"], r["c"]) for r in recs if half[r["stem"]] == 0])}
    for s in sites:
        fit = [(r["a"], r["c"]) for r in recs if r["site"] == s and half[r["stem"]] == 0]
        luts[s] = fit_lut(fit) if len(fit) >= 3 else luts["global"]
    held = [r for r in recs if half[r["stem"]] == 1]
    mae_before = np.mean([r["before"]["mae"] for r in held])
    mae_after = np.mean([np.abs(luts[r["site"]][r["a"]] - r["c"]).mean() for r in held])
    r_mean = np.mean([r["before"]["r"] for r in recs])
    apply = bool(mae_after < 0.9 * mae_before)
    # refit on all pairs for use
    luts = {"global": fit_lut([(r["a"], r["c"]) for r in recs])}
    for s in sites:
        fit = [(r["a"], r["c"]) for r in recs if r["site"] == s]
        luts[s] = fit_lut(fit) if len(fit) >= 3 else luts["global"]
    out = {"apply": apply, "luts": {k: [float(v) for v in lut] for k, lut in luts.items()},
           "mae_before": float(mae_before), "mae_after_heldout": float(mae_after), "pearson_mean": float(r_mean),
           "pairs": len(recs)}
    with open(os.path.join(args.unlabeled, "lut_v3.json"), "w") as f:
        json.dump(out, f)
    lines = [f"## GONG archive (hag) vs competition JPEGs, {len(recs)} paired frames\n",
             "| stem | site | MAE | Pearson r (disk) | mean diff | std ratio |", "|---|---|---|---|---|---|"]
    for r in recs:
        b = r["before"]
        lines.append(f"| {r['stem']} | {r['site']} | {b['mae']:.2f} | {b['r']:.4f} | {b['mean_diff']:+.2f} | "
                     f"{b['std_ratio']:.3f} |")
    verdict = ("apply the per-site LUT to every unlabelled frame" if apply
               else "in-domain enough: no intensity mapping (differences are compression noise)")
    lines.append(f"\nheld-out MAE without LUT {mae_before:.2f}, with LUT {mae_after:.2f}; mean Pearson r "
                 f"{r_mean:.4f} -> **{verdict}**")
    with open(os.path.join(args.unlabeled, "domain_check.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-1:]))
    print(f"\nadd to DECISIONS.md:\n- {date.today()}: GONG hag domain check on {len(recs)} paired frames: "
          f"MAE {mae_before:.2f} -> {mae_after:.2f} with a per-site LUT, Pearson {r_mean:.3f}; decision: {verdict}.")


if __name__ == "__main__":
    main()
