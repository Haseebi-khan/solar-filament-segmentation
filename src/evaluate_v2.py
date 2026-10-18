"""v2: score out-of-fold CSVs on every quantitative item of the organisers' rubric.

  python src/evaluate_v2.py work/oof_sub_r34_1024+rgy040_1024+r34_1536.csv work/oof_sub_v2_<members>.csv \
      --labels v1 v2 --out work/eval_v2

For each CSV (same format as a submission, but for the training images) and the training ground
truth, grouped exactly like the organisers' self-evaluation notebook (every annotation set
``<annotator>-<stem>`` against all predictions of ``<stem>``):

  * Panoptic Quality (PQ = SQ x RQ, TP/FP/FN pooled over annotation sets), the leaderboard metric;
  * the distribution of IoU and of Dice over all overlapping GT/prediction pairs (IoU > 0), as
    plotted by the self-evaluation notebook (Dice = 2 IoU / (1 + IoU) for the same pair);
  * one-to-many and many-to-one relations: for every GT filament the number of predictions it
    overlaps (1:n; n >= 2 means fragmentation) and for every prediction the number of GT filaments
    it overlaps (n:1; n >= 2 means over-merging);
  * for reference, the semantic Dice per annotation set (union of predictions vs union of GT),
    the competition's original metric.

Writes <out>/summary.json and one figure per CSV (<out>/<label>.png) for the technical report.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from filseg.data import CACHE_DIR, WORK_ROOT
from filseg.metric import PQStats, iou_matrix, pq_stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

H = W = 2048


def load_predictions(path: str) -> dict[str, list[dict]]:
    df = pd.read_csv(path)
    stems = df["filament_id"].str.split("_", n=1).str[0]
    return {s: [{"size": [H, W], "counts": c.encode()} for c in g["segmentation_rle"]] for s, g in df.groupby(stems)}


def semantic_dice(gt: list[dict], pred: list[dict]) -> float:
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    g = mask_util.merge(gt, intersect=False)
    p = mask_util.merge(pred, intersect=False)
    inter = mask_util.area(mask_util.merge([g, p], intersect=True))
    return float(2 * inter / (mask_util.area(g) + mask_util.area(p)))


def evaluate_csv(meta: dict, preds: dict[str, list[dict]]) -> dict:
    stats = PQStats()
    pair_iou, gt_deg, pred_deg, sem = [], [], [], []
    for stem in meta["stems"]:
        p = preds.get(stem, [])
        for g in meta["gt_rles"][stem]:
            stats.add(pq_stats(g, p))
            sem.append(semantic_dice(g, p))
            if not g:
                pred_deg.extend([0] * len(p))
                continue
            if not p:
                gt_deg.extend([0] * len(g))
                continue
            ious = iou_matrix(g, p)
            pair_iou.extend(ious[ious > 0].tolist())
            hit = ious > 0
            gt_deg.extend(hit.sum(1).tolist())
            pred_deg.extend(hit.sum(0).tolist())
    pair_iou = np.asarray(pair_iou)
    pair_dice = 2 * pair_iou / (1 + pair_iou)

    def degrees(d):
        c = Counter(min(int(x), 3) for x in d)
        n = max(1, len(d))
        return {"0": c[0], "1": c[1], "2": c[2], "3+": c[3], "share_1": round(c[1] / n, 4),
                "share_2plus": round((c[2] + c[3]) / n, 4)}

    return {
        "PQ": round(stats.pq, 4), "SQ": round(stats.sq, 4), "RQ": round(stats.rq, 4),
        "TP": stats.tp, "FP": stats.fp, "FN": stats.fn,
        "pairs_overlapping": int(len(pair_iou)),
        "iou_mean": round(float(pair_iou.mean()), 4) if len(pair_iou) else 0.0,
        "iou_median": round(float(np.median(pair_iou)), 4) if len(pair_iou) else 0.0,
        "dice_mean": round(float(pair_dice.mean()), 4) if len(pair_dice) else 0.0,
        "dice_median": round(float(np.median(pair_dice)), 4) if len(pair_dice) else 0.0,
        "share_pairs_iou_gt_0.5": round(float((pair_iou > 0.5).mean()), 4) if len(pair_iou) else 0.0,
        "gt_to_pred (1:n)": degrees(gt_deg),
        "pred_to_gt (n:1)": degrees(pred_deg),
        "semantic_dice_mean": round(float(np.mean(sem)), 4),
        "_pair_iou": pair_iou, "_gt_deg": gt_deg, "_pred_deg": pred_deg,
    }


def plot(res: dict, label: str, path: str) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    bins = np.linspace(0, 1, 51)
    for a, values, name in [(ax[0], res["_pair_iou"], "IoU"),
                            (ax[1], 2 * res["_pair_iou"] / (1 + res["_pair_iou"]), "Dice")]:
        w = np.ones(len(values)) / max(1, len(values))
        a.hist(values, bins=bins, weights=w, color="navy", alpha=0.75, edgecolor="black")
        if len(values):
            a.axvline(values.mean(), color="crimson", ls="--", label=f"mean = {values.mean():.3f}")
            a.legend(fontsize=8)
        a.set_xlabel(name)
        a.set_ylabel("share of overlapping pairs")
        a.set_title(f"{name} distribution")
        a.grid(axis="y", ls="--")
    gt_c, pr_c = Counter(res["_gt_deg"]), Counter(res["_pred_deg"])
    for deg, cnt in gt_c.items():
        ax[2].bar(deg + 0.5, cnt, width=0.8, color="crimson" if deg == 0 else "forestgreen", alpha=0.75,
                  edgecolor="black")
    for deg, cnt in pr_c.items():
        ax[2].bar(-(deg + 0.5), cnt, width=0.8, color="darkorange" if deg == 0 else "steelblue", alpha=0.75,
                  edgecolor="black")
    ticks = sorted([d + 0.5 for d in gt_c] + [-(d + 0.5) for d in pr_c])
    ax[2].set_xticks(ticks)
    ax[2].set_xticklabels([f"1:{int(t - 0.5)}" if t > 0 else f"{int(-t - 0.5)}:1" for t in ticks], fontsize=8)
    ax[2].axvline(0, color="black", lw=0.8)
    ax[2].set_title("relations (left: prediction -> GT, right: GT -> prediction)")
    ax[2].set_ylabel("count")
    ax[2].grid(axis="y", ls="--")
    fig.suptitle(f"{label}: PQ {res['PQ']:.4f}  SQ {res['SQ']:.4f}  RQ {res['RQ']:.4f}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="out-of-fold CSVs for the training images")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out", default=os.path.join(WORK_ROOT, "eval_v2"))
    args = ap.parse_args()
    labels = args.labels or [os.path.splitext(os.path.basename(c))[0] for c in args.csvs]
    assert len(labels) == len(args.csvs), "one label per CSV"
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for path, label in zip(args.csvs, labels):
        res = evaluate_csv(meta, load_predictions(path))
        plot(res, label, os.path.join(args.out, f"{label}.png"))
        summary[label] = {k: v for k, v in res.items() if not k.startswith("_")}
        summary[label]["csv"] = path
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    table = pd.DataFrame({k: {m: (v if not isinstance(v, dict) else v.get("share_2plus")) for m, v in s.items()
                              if m != "csv"} for k, s in summary.items()})
    print(table.to_string())
    print("(rows '1:n' / 'n:1' show the share of filaments with 2 or more overlapping partners)")
    print("wrote", os.path.join(args.out, "summary.json"), "and one figure per CSV")


if __name__ == "__main__":
    main()
