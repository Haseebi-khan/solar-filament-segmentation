"""v2: several extent hypotheses per filament, so a learned selector can choose each filament's extent.

v1 gives every filament the same extent, p > 0.6. Out of fold, picking the best threshold per filament
(an oracle) raises the candidate set's PQ from 0.492 to 0.522, so the right extent differs between
filaments. Here every candidate keeps its v1 identity (same seeds, same fragment grouping) and gets
one mask per level:

  level < t_seed : the seeds grown by a watershed over p > level   (larger than v1)
  level = t_seed : the v1 mask
  level > t_seed : the v1 mask restricted to p > level              (smaller than v1)

Holes are filled at every level, as in v1, and the masks of one level are disjoint. The selector
(``src/train_selector_v2.py``) predicts u = expected per-annotator TP-IoU for every (filament, level),
keeps the best level of each filament and drops filaments whose best u is below a tuned threshold.
"""
from __future__ import annotations

import cv2
import numpy as np
from pycocotools import mask as mask_util
from skimage.segmentation import watershed

from .postprocess import _close_instances, _disk, add_features, instance_table
from .rescore import FEATURES, feature_matrix

LEVELS = (0.4, 0.5, 0.6, 0.7)
EXTRA = ["level", "area_ratio", "base_mean", "base_p90", "base_log_area", "base_core80"]
FEATURES_V2 = FEATURES + EXTRA


def level_labels(p: np.ndarray, t_seed: float = 0.6, link: int = 8, levels=LEVELS
                 ) -> tuple[int, dict[float, np.ndarray]]:
    """Instance label maps, one per level, sharing the instance ids of the v1 extraction."""
    seeds = (p > t_seed).astype(np.uint8)
    if link > 0:
        n, groups = cv2.connectedComponents(cv2.dilate(seeds, _disk(link)), connectivity=8)
        base = (groups * seeds).astype(np.int32)
    else:
        n, base = cv2.connectedComponents(seeds, connectivity=8)
        base = base.astype(np.int32)
    n -= 1
    labs = {}
    for t in levels:
        if n == 0:
            labs[t] = base
            continue
        if t < t_seed:
            lab = watershed(-p, markers=base, mask=p > t).astype(np.int32)
        elif t > t_seed:
            lab = np.where(p > t, base, 0).astype(np.int32)
        else:
            lab = base.copy()
        labs[t] = _close_instances(lab, n, -1)
    return n, labs


def hypotheses(p: np.ndarray, image: np.ndarray, t_seed: float = 0.6, link: int = 8, min_area: int = 100,
               levels=LEVELS) -> tuple[list[dict], dict[float, np.ndarray]]:
    """One record per (filament, level) with the v1 rescorer features plus level information.

    A filament is a candidate if its v1 mask (level t_seed) has at least ``min_area`` pixels. A level
    above t_seed can remove a filament entirely; it then has no record at that level.
    """
    levels = tuple(sorted(levels))
    assert t_seed in levels, "the v1 extent (t_seed) must be one of the levels"
    n, labs = level_labels(p, t_seed, link, levels)
    if n == 0:
        return [], labs
    tables = {t: {o["label"]: o for o in instance_table(p, labs[t], n)} for t in levels}
    keep = sorted(k for k, o in tables[t_seed].items() if o["area"] >= min_area)
    if not keep:
        return [], labs
    rows = []
    # the v1 level first, so the base features exist when the other levels refer to them
    for t in (t_seed,) + tuple(t for t in levels if t != t_seed):
        objs = [tables[t][k] for k in keep if k in tables[t]]
        add_features(objs, p, labs[t], image)
        for o in objs:
            b = tables[t_seed][o["label"]]
            o["level"] = float(t)
            o["area_ratio"] = o["area"] / b["area"]
            o["base_mean"] = b["mean"]
            o["base_p90"] = b["p90"]
            o["base_log_area"] = float(np.log1p(b["area"]))
            o["base_core80"] = b["core80"]
        rows.extend(objs)
    return rows, labs


def feature_matrix_v2(rows: list[dict]) -> np.ndarray:
    extra = np.asarray([[float(r[k]) for k in EXTRA] for r in rows], np.float64).reshape(len(rows), len(EXTRA))
    return np.hstack([feature_matrix(rows), extra])


def encode_row(labs: dict[float, np.ndarray], row: dict) -> dict:
    lab = labs[row["level"]]
    m = np.zeros(lab.shape, np.uint8, order="F")
    y0, y1, x0, x1 = row["y0"], row["y1"], row["x0"], row["x1"]
    m[y0:y1, x0:x1] = lab[y0:y1, x0:x1] == row["label"]
    return mask_util.encode(m)


def best_rows(instance_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Index of the highest-scoring row of every instance."""
    order = np.lexsort((-scores, instance_ids))
    first = np.ones(len(order), bool)
    first[1:] = instance_ids[order][1:] != instance_ids[order][:-1]
    return order[first]


def select_and_assemble(rows: list[dict], scores: np.ndarray, threshold: float,
                        labs: dict[float, np.ndarray]) -> list[dict]:
    """Best level per filament, kept if its predicted u > threshold; returns disjoint RLEs.

    Filaments are placed in order of decreasing score; a pixel already taken by a higher-scoring
    filament (possible only where two different levels meet) stays with that filament.
    """
    if not rows:
        return []
    ids = np.array([r["label"] for r in rows])
    chosen = [j for j in best_rows(ids, scores) if scores[j] > threshold]
    chosen.sort(key=lambda j: -scores[j])
    shape = next(iter(labs.values())).shape
    claimed = np.zeros(shape, bool)
    rles = []
    for j in chosen:
        r = rows[j]
        y0, y1, x0, x1 = r["y0"], r["y1"], r["x0"], r["x1"]
        m = (labs[r["level"]][y0:y1, x0:x1] == r["label"]) & ~claimed[y0:y1, x0:x1]
        if not m.any():
            continue
        claimed[y0:y1, x0:x1] |= m
        full = np.zeros(shape, np.uint8, order="F")
        full[y0:y1, x0:x1] = m
        rles.append(mask_util.encode(full))
    return rles
