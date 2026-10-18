"""v3: error taxonomy of a set of predictions (blueprint section 4.5).

For every annotation set of an image, every GT filament and every prediction is classified:

  GT  tp          some prediction has IoU > 0.5
      missed      no prediction covers 10 % of it
      fragmented  2+ predictions overlap it, none with IoU > 0.5
      merged      its best-overlapping prediction overlaps 2+ GT filaments of this set
      extent      otherwise (one overlapping prediction, IoU <= 0.5)
  Pred tp         some GT has IoU > 0.5
      spurious    overlaps no GT
      fragment    its best-overlapping GT is overlapped by 2+ predictions
      merged_extent  otherwise

"Overlap" of a GT g and a prediction p: |g & p| >= 10 % of |g| or >= 50 % of |p| (a small piece lying inside a
long filament counts). "PQ lost" of an error class is the PQ gain if those errors were removed from the
denominator (each FN/FP costs 0.5); for the TPs it is the gain if every matched IoU were 1 (the SQ deficit).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .masks_v3 import CropMask, inter_matrix, iou_from_inter

GT_CLASSES = ("tp", "missed", "fragmented", "merged", "extent")
PRED_CLASSES = ("tp", "spurious", "fragment", "merged_extent")
SIZE_BUCKETS = ((0, 500, "<500"), (500, 2000, "500-2k"), (2000, 8000, "2k-8k"), (8000, 10 ** 9, ">8k"))
RHO_BUCKETS = ((0, 0.5, "r<0.5"), (0.5, 0.8, "0.5-0.8"), (0.8, 10.0, ">0.8"))


def size_bucket(area: float) -> str:
    return next(name for lo, hi, name in SIZE_BUCKETS if lo <= area < hi)


def rho_bucket(rho: float) -> str:
    return next(name for lo, hi, name in RHO_BUCKETS if lo <= rho < hi)


def classify_set(gt: list[CropMask], pred: list[CropMask], thr: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Per-GT and per-prediction records for one (image, annotation set)."""
    inter = inter_matrix(gt, pred).astype(np.float64)
    ious = iou_from_inter(inter, gt, pred)
    ga = np.array([g.area for g in gt], np.float64)
    pa = np.array([p.area for p in pred], np.float64)
    if inter.size:
        ov = (inter >= 0.1 * ga[:, None]) | (inter >= 0.5 * pa[None, :])
        ov &= inter > 0
    else:
        ov = np.zeros(inter.shape, bool)
    hit = ious > thr
    gts, preds = [], []
    for i, g in enumerate(gt):
        rec = dict(area=g.area, best_iou=float(ious[i].max()) if len(pred) else 0.0)
        if len(pred) and hit[i].any():
            rec["cls"] = "tp"
            rec["iou"] = float(ious[i][hit[i]].sum())
            rec["n_tp"] = int(hit[i].sum())
        else:
            cover = inter[i].max() / max(g.area, 1) if len(pred) else 0.0
            n_ov = int(ov[i].sum()) if len(pred) else 0
            if cover < 0.1 and n_ov == 0:
                rec["cls"] = "missed"
            elif n_ov >= 2:
                rec["cls"] = "fragmented"
            else:
                j = int(np.argmax(inter[i]))
                rec["cls"] = "merged" if ov[:, j].sum() >= 2 else "extent"
        rec["n_overlap"] = int(ov[i].sum()) if len(pred) else 0
        gts.append(rec)
    for j, p in enumerate(pred):
        rec = dict(area=p.area)
        if len(gt) and hit[:, j].any():
            rec["cls"] = "tp"
        elif not len(gt) or not ov[:, j].any():
            rec["cls"] = "spurious"
        else:
            i = int(np.argmax(inter[:, j]))
            rec["cls"] = "fragment" if ov[i].sum() >= 2 else "merged_extent"
        rec["n_overlap"] = int(ov[:, j].sum()) if len(gt) else 0
        preds.append(rec)
    return gts, preds


@dataclass
class Tally:
    num: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    counts: dict = field(default_factory=lambda: defaultdict(int))

    @property
    def den(self) -> float:
        return self.tp + 0.5 * self.fp + 0.5 * self.fn

    @property
    def pq(self) -> float:
        return self.num / self.den if self.den else 0.0


def tally(gt_recs: list[dict], pred_recs: list[dict], key_gt=None, key_pred=None) -> dict[str, Tally]:
    """Pools records into PQ tallies, overall ("all") and per group if key functions are given."""
    out: dict[str, Tally] = defaultdict(Tally)
    for r in gt_recs:
        for k in ["all"] + ([key_gt(r)] if key_gt else []):
            t = out[k]
            if r["cls"] == "tp":
                t.num += r["iou"]
                t.tp += r["n_tp"]
            else:
                t.fn += 1
            t.counts["gt_" + r["cls"]] += 1
    for r in pred_recs:
        for k in ["all"] + ([key_pred(r)] if key_pred else []):
            t = out[k]
            if r["cls"] != "tp":
                t.fp += 1
            t.counts["pred_" + r["cls"]] += 1
    return dict(out)


def pq_lost(t: Tally) -> list[tuple[str, int, float]]:
    """(cause, count, PQ gain if removed) sorted by gain."""
    rows = []
    for c in GT_CLASSES[1:]:
        n = t.counts.get("gt_" + c, 0)
        rows.append((f"FN {c}", n, t.num / (t.den - 0.5 * n) - t.pq if t.den - 0.5 * n > 0 else 0.0))
    for c in PRED_CLASSES[1:]:
        n = t.counts.get("pred_" + c, 0)
        rows.append((f"FP {c}", n, t.num / (t.den - 0.5 * n) - t.pq if t.den - 0.5 * n > 0 else 0.0))
    rows.append(("TP IoU deficit (SQ)", t.tp, t.tp / t.den - t.pq if t.den else 0.0))
    return sorted(rows, key=lambda r: -r[2])
