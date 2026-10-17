"""Panoptic Quality exactly as in the organisers' self-evaluation notebook.

For every annotation set (``<annotator>-<stem>``) all of its GT filaments are
compared with *all* predicted filaments of ``<stem>``:

    hit  = IoU > 0.5
    TP   = every hit pair (contributes its IoU)
    FP   = predictions without any hit
    FN   = GT filaments without any hit
    PQ   = sum(IoU_TP) / (|TP| + 0.5 |FP| + 0.5 |FN|), pooled over all sets.

IoUs are computed on COCO RLEs with pycocotools, which is exact and much faster
than the dense matmul used in the notebook.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pycocotools import mask as mask_util


@dataclass
class PQStats:
    tp_iou_sum: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "PQStats") -> "PQStats":
        self.tp_iou_sum += other.tp_iou_sum
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    @property
    def pq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp_iou_sum / d if d > 0 else 0.0

    @property
    def sq(self) -> float:
        return self.tp_iou_sum / self.tp if self.tp else 0.0

    @property
    def rq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / d if d > 0 else 0.0

    def __str__(self) -> str:
        return f"PQ={self.pq:.4f} SQ={self.sq:.4f} RQ={self.rq:.4f} TP={self.tp} FP={self.fp} FN={self.fn}"


def iou_matrix(gt_rles: list[dict], pred_rles: list[dict]) -> np.ndarray:
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)))
    return np.asarray(mask_util.iou(gt_rles, pred_rles, [0] * len(pred_rles))).reshape(len(gt_rles), len(pred_rles))


def pq_stats(gt_rles: list[dict], pred_rles: list[dict], thr: float = 0.5) -> PQStats:
    n_gt, n_pred = len(gt_rles), len(pred_rles)
    if n_gt == 0:
        return PQStats(fp=n_pred)
    if n_pred == 0:
        return PQStats(fn=n_gt)
    ious = iou_matrix(gt_rles, pred_rles)
    hit = ious > thr
    return PQStats(
        tp_iou_sum=float(ious[hit].sum()),
        tp=int(hit.sum()),
        fp=int((hit.sum(0) == 0).sum()),
        fn=int((hit.sum(1) == 0).sum()),
    )


def instance_contributions(gt_sets: list[list[dict]], pred_rles: list[dict], thr: float = 0.5
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-prediction additive PQ terms for one image.

    With disjoint predictions a GT filament can be hit (IoU > 0.5) by at most one
    prediction, so for any subset S of the predictions

        PQ(S) = sum_{i in S} a_i / (0.5 * N_gt + sum_{i in S} d_i)

    where, over the image's annotation sets j,
        a_i = sum_j  sum of IoUs of GT filaments hit by i,
        d_i = sum_j  (0.5 * hits_ij + 0.5 * [hits_ij == 0]),
    and N_gt is the total number of GT filaments. Returns (a, d, best_iou) where
    best_iou is the mean over sets of the best IoU (a diagnostic).
    """
    n = len(pred_rles)
    a = np.zeros(n)
    d = np.zeros(n)
    best = np.zeros(n)
    for g in gt_sets:
        ious = iou_matrix(g, pred_rles)
        if ious.size == 0:
            d += 0.5
            continue
        hit = ious > thr
        h = hit.sum(0)
        a += np.where(hit, ious, 0).sum(0)
        d += 0.5 * h + 0.5 * (h == 0)
        best += ious.max(0)
    return a, d, best / max(1, len(gt_sets))


def pq_from_contributions(a: np.ndarray, d: np.ndarray, n_gt_total: int) -> float:
    den = 0.5 * n_gt_total + d.sum()
    return float(a.sum() / den) if den > 0 else 0.0


def evaluate(gt_sets: dict[str, list[list[dict]]], preds: dict[str, list[dict]]) -> PQStats:
    """gt_sets: {stem: [rles of set 1, rles of set 2, ...]}; preds: {stem: rles}."""
    total = PQStats()
    for stem, sets in gt_sets.items():
        p = preds.get(stem, [])
        for g in sets:
            total.add(pq_stats(g, p))
    return total
