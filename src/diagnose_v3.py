"""v3 Phase 0: error taxonomy and breakdowns of out-of-fold CSVs ("PQ lost by cause").

  python src/diagnose_v3.py work/oof_sub_v2_r34_1024+rgy040_1024+r34_1536+r34_2048+effv2s_1536.csv --labels v2

For each CSV (submission format, training images) and the training ground truth (every annotation set scored
separately, as the organisers do) it prints and writes to <out>/<label>.md and <out>/summary.json:
  * PQ, SQ, RQ, TP, FP, FN;
  * PQ lost by cause (filseg/taxonomy_v3.py): missed / fragmented / merged / extent FNs, spurious / fragment /
    merged-extent FPs, and the SQ deficit of the matches;
  * PQ, recall and error counts by GT size, radial zone (r/R of the centroid), instrument site and number of
    annotators;
  * the decision rule of blueprint section 14: which lever the losses point to, and whether M4 (query-based
    instance proposals) should be promoted (fragmented + merged > 40 % of the lost PQ).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

from filseg.data import CACHE_DIR, WORK_ROOT
from filseg.masks_v3 import CropMask, gt_crops
from filseg.postprocess import disk_geometry
from filseg.taxonomy_v3 import classify_set, pq_lost, rho_bucket, size_bucket, tally

_G = {}


def _init(preds, meta):
    _G["preds"], _G["meta"] = preds, meta
    _G["images"] = np.load(os.path.join(CACHE_DIR, "train_images.npy"), mmap_mode="r")


def _image(i):
    meta = _G["meta"]
    stem = meta["stems"][i]
    cy, cx, radius = disk_geometry(np.asarray(_G["images"][i]))
    pred = [CropMask.from_rle({"size": [2048, 2048], "counts": c.encode()}) for c in _G["preds"].get(stem, [])]
    pred_rho = []
    for p in pred:
        y, x = p.centroid()
        pred_rho.append(np.hypot(y - cy, x - cx) / radius)
    site = stem[14] if len(stem) > 14 else "?"
    k = len(meta["gt_rles"][stem])
    gts_all, preds_all = [], []
    for gset in gt_crops(meta["gt_rles"][stem]):
        g_recs, p_recs = classify_set(gset, pred)
        for g, r in zip(gset, g_recs):
            y, x = g.centroid()
            r.update(rho=np.hypot(y - cy, x - cx) / radius, site=site, n_sets=k)
        for r, rho in zip(p_recs, pred_rho):
            r.update(rho=rho, site=site, n_sets=k)
        gts_all.extend(g_recs)
        preds_all.extend(p_recs)
    return gts_all, preds_all


def table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def analyse(gt_recs, pred_recs) -> tuple[str, dict]:
    md = []
    t = tally(gt_recs, pred_recs)["all"]
    sq = t.num / t.tp if t.tp else 0.0
    rq = t.tp / t.den if t.den else 0.0
    md.append(f"PQ **{t.pq:.4f}**  SQ {sq:.4f}  RQ {rq:.4f}  TP {t.tp}  FP {t.fp}  FN {t.fn}\n")
    lost = pq_lost(t)
    total_gain = sum(max(g, 0) for _, _, g in lost)
    md.append("### PQ lost by cause\n")
    md.append(table([[c, n, f"{g:+.4f}", f"{100 * g / total_gain:.0f}%"] for c, n, g in lost],
                    ["cause", "count", "PQ if removed", "share"]))
    summary = {"PQ": t.pq, "SQ": sq, "RQ": rq, "TP": t.tp, "FP": t.fp, "FN": t.fn,
               "lost": {c: {"count": n, "gain": g} for c, n, g in lost}}
    breakdowns = {
        "GT / prediction size (px)": (lambda r: size_bucket(r["area"]), lambda r: size_bucket(r["area"])),
        "radial zone r/R": (lambda r: rho_bucket(r["rho"]), lambda r: rho_bucket(r["rho"])),
        "instrument site": (lambda r: r["site"], lambda r: r["site"]),
        "annotation sets per image": (lambda r: str(r["n_sets"]), lambda r: str(r["n_sets"])),
    }
    summary["breakdowns"] = {}
    for title, (kg, kp) in breakdowns.items():
        groups = tally(gt_recs, pred_recs, kg, kp)
        rows = []
        for name in sorted(k for k in groups if k != "all"):
            g = groups[name]
            n_gt = g.tp + g.fn
            rows.append([name, f"{g.pq:.4f}", n_gt, g.tp, g.fp, g.fn, f"{g.tp / max(n_gt, 1):.3f}",
                         g.counts.get("gt_fragmented", 0), g.counts.get("gt_merged", 0),
                         g.counts.get("gt_missed", 0), g.counts.get("pred_spurious", 0)])
            summary["breakdowns"].setdefault(title, {})[name] = {"PQ": g.pq, "TP": g.tp, "FP": g.fp, "FN": g.fn}
        md.append(f"\n### By {title}\n")
        md.append(table(rows, ["group", "PQ", "GT", "TP", "FP", "FN", "recall", "FN frag", "FN merged",
                               "FN missed", "FP spurious"]))
    deg_g = Counter(min(r["n_overlap"], 3) for r in gt_recs)
    deg_p = Counter(min(r["n_overlap"], 3) for r in pred_recs)
    md.append("\n### Overlap degrees (1:n = predictions per GT, n:1 = GTs per prediction)\n")
    md.append(table([[f"{k}{'+' if k == 3 else ''}", deg_g[k], deg_p[k]] for k in range(4)],
                    ["overlapping partners", "GT filaments (1:n)", "predictions (n:1)"]))
    gains = {c: max(g, 0) for c, _, g in lost}
    grouping = gains["FN fragmented"] + gains["FN merged"] + gains["FP fragment"]
    share = grouping / total_gain if total_gain else 0.0
    spurious = gains["FP spurious"] / total_gain if total_gain else 0.0
    sq_share = gains["TP IoU deficit (SQ)"] / total_gain if total_gain else 0.0
    lever = max([("B (hypotheses: grouping / extent)", share + gains["FN extent"] / max(total_gain, 1e-9)),
                 ("C (decisions: FP control)", spurious), ("A (pixel quality / SQ)", sq_share)], key=lambda x: x[1])
    md.append(f"\n### Decision (blueprint section 14)\n\nfragmented + merged share of the lost PQ: "
              f"**{100 * share:.0f}%** -> M4 (query-based proposals) "
              f"{'**promoted** to Phase 2' if share > 0.40 else 'stays optional'}; largest lever: {lever[0]}.")
    summary.update(grouping_share=share, promote_m4=share > 0.40, lever=lever[0])
    return "\n".join(md), summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out", default=os.path.join(WORK_ROOT, "diag_v3"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    labels = args.labels or [os.path.splitext(os.path.basename(c))[0] for c in args.csvs]
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    for path, label in zip(args.csvs, labels):
        df = pd.read_csv(path)
        stems = df["filament_id"].str.split("_", n=1).str[0]
        preds = {s: list(g["segmentation_rle"]) for s, g in df.groupby(stems)}
        with Pool(args.workers, initializer=_init, initargs=(preds, meta)) as pool:
            parts = pool.map(_image, range(len(meta["stems"])), chunksize=4)
        gt_recs = [r for g, _ in parts for r in g]
        pred_recs = [r for _, p in parts for r in p]
        md, s = analyse(gt_recs, pred_recs)
        md = f"## {label}\n\n`{path}`\n\n" + md
        with open(os.path.join(args.out, f"{label}.md"), "w") as f:
            f.write(md + "\n")
        summary[label] = dict(s, csv=path)
        print(md, "\n")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
