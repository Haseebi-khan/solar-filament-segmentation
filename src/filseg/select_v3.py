"""v3: overlap-aware greedy selection of candidates and exact PQ of crop-mask predictions.

Selection (blueprint 12.6): per image, candidates are visited in order of decreasing score; a candidate with
score <= tau stops the scan; one that overlaps already selected pixels by more than rho_ov of its area is
skipped; otherwise its overlapping pixels are removed (its holes stay filled because they were filled at
extraction) and it is kept if at least ``min_area`` pixels remain. The result is a disjoint set of masks.

The v3 blueprint also asks to "keep the largest connected component" of each candidate. We do not: v1's
fragment grouping deliberately makes one filament out of several nearby pieces (+0.008 PQ), and keeping only
the largest piece would undo it (DECISIONS.md, 2026-10-18).
"""
from __future__ import annotations

import numpy as np

from .masks_v3 import CropMask, H, W, inter_matrix, iou_from_inter, paste, subtract
from .metric import PQStats


def greedy_select(cands: list[CropMask], scores: np.ndarray, tau: float, rho_ov: float = 0.1,
                  min_area: int = 30, shape=(H, W)) -> tuple[list[CropMask], list[int]]:
    occupied = np.zeros(shape, bool)
    out, idx = [], []
    for j in np.argsort(-np.asarray(scores), kind="stable"):
        if scores[j] <= tau:
            break
        c = cands[j]
        if c.area == 0:
            continue
        ov = int(np.count_nonzero(c.m & occupied[c.y0:c.y1, c.x0:c.x1]))
        if ov > rho_ov * c.area:
            continue
        t = subtract(c, occupied) if ov else c
        if t.area < min_area:
            continue
        paste(occupied, t)
        out.append(t)
        idx.append(int(j))
    return out, idx


def pq_stats_crops(gt: list[CropMask], pred: list[CropMask], thr: float = 0.5) -> PQStats:
    """Organisers' PQ terms for one annotation set (same as metric.pq_stats on the RLEs of these masks)."""
    if not gt:
        return PQStats(fp=len(pred))
    if not pred:
        return PQStats(fn=len(gt))
    ious = iou_from_inter(inter_matrix(gt, pred), gt, pred)
    hit = ious > thr
    return PQStats(tp_iou_sum=float(ious[hit].sum()), tp=int(hit.sum()), fp=int((hit.sum(0) == 0).sum()),
                   fn=int((hit.sum(1) == 0).sum()))


def image_stats(gt_sets: list[list[CropMask]], pred: list[CropMask]) -> PQStats:
    total = PQStats()
    for g in gt_sets:
        total.add(pq_stats_crops(g, pred))
    return total


def candidate_targets(gt_sets: list[list[CropMask]], cands: list[CropMask], thr: float = 0.5
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per candidate: a = sum over sets of TP IoUs, m_frac = share of sets with a hit, best = mean best IoU.
    u = a / K is the regression target (adding the candidate to disjoint predictions raises PQ iff u > PQ/2)."""
    n = len(cands)
    a, hits, best = np.zeros(n), np.zeros(n), np.zeros(n)
    k = max(1, len(gt_sets))
    for g in gt_sets:
        if not g or not n:
            continue
        ious = iou_from_inter(inter_matrix(g, cands), g, cands)
        hit = ious > thr
        a += np.where(hit, ious, 0).sum(0)
        hits += hit.any(0)
        best += ious.max(0)
    return a, hits / k, best / k
