"""Probability map -> disjoint filament instances.

Instances are connected components of ``p > t_seed``; with ``t_mask < t_seed``
each seed is grown by a watershed over the ``p > t_mask`` region (pixels of a
low-confidence region that touch no seed are dropped). Instances smaller than
``min_area`` (full-resolution pixels) or with a score below ``min_score`` are
removed. Every pixel belongs to at most one instance, so predicted masks never
overlap.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from pycocotools import mask as mask_util
from scipy.ndimage import binary_fill_holes, find_objects
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

FULL = 2048


@dataclass
class PPConfig:
    t_seed: float = 0.5
    t_mask: float = 0.5
    min_area: int = 0
    min_score: float = 0.0
    score: str = "mean"          # instance score: mean | max | p90
    link: int = 0                # seeds closer than ~2*link px form one instance (fragment grouping)
    close: int = 0               # per-instance closing radius (bridges gaps); -1 = fill holes only

    def key(self) -> tuple:
        return (self.t_seed, self.t_mask, self.min_area, self.min_score, self.score, self.link, self.close)


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def upsample(prob: np.ndarray, size: int = FULL) -> np.ndarray:
    """Bilinear upsampling (half-pixel centres, same convention as align_corners=False)."""
    prob = prob.astype(np.float32)
    if prob.shape[0] == size:
        return prob
    return cv2.resize(prob, (size, size), interpolation=cv2.INTER_LINEAR)


def label_instances(p: np.ndarray, t_seed: float, t_mask: float, link: int = 0, close: int = 0
                    ) -> tuple[int, np.ndarray]:
    seeds = (p > t_seed).astype(np.uint8)
    if link > 0:
        n, groups = cv2.connectedComponents(cv2.dilate(seeds, _disk(link)), connectivity=8)
        lab = (groups * seeds).astype(np.int32)
    else:
        n, lab = cv2.connectedComponents(seeds, connectivity=8)
    if t_mask < t_seed and n > 1:
        region = p > t_mask
        lab = watershed(-p, markers=lab, mask=region).astype(np.int32)
    if close != 0 and n > 1:
        lab = _close_instances(lab, n - 1, close)
    return n - 1, lab


def _close_instances(lab: np.ndarray, n: int, r: int) -> np.ndarray:
    """Per-instance closing (r > 0) or hole filling (r < 0); new pixels only come from the background."""
    out = lab.copy()
    h, w = lab.shape
    pad = max(r, 1)
    k = _disk(r) if r > 0 else None
    for i, sl in enumerate(find_objects(lab, max_label=n), start=1):
        if sl is None:
            continue
        y0, y1 = max(0, sl[0].start - pad), min(h, sl[0].stop + pad)
        x0, x1 = max(0, sl[1].start - pad), min(w, sl[1].stop + pad)
        m = lab[y0:y1, x0:x1] == i
        grown = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool) if r > 0 else binary_fill_holes(m)
        sub = out[y0:y1, x0:x1]
        sub[grown & (sub == 0)] = i
    return out


def instance_table(p: np.ndarray, lab: np.ndarray, n: int) -> list[dict]:
    """Per-instance bbox/area and probability statistics."""
    if n == 0:
        return []
    objs = []
    for k, sl in enumerate(find_objects(lab, max_label=n), start=1):
        if sl is None:
            continue
        inside = lab[sl] == k
        pv = p[sl][inside]
        objs.append(dict(label=k, area=int(pv.size), mean=float(pv.mean()), max=float(pv.max()),
                         p90=float(np.percentile(pv, 90)),
                         y0=sl[0].start, y1=sl[0].stop, x0=sl[1].start, x1=sl[1].stop))
    return objs


def encode_instance(lab: np.ndarray, obj: dict) -> dict:
    m = np.zeros(lab.shape, np.uint8, order="F")
    y0, y1, x0, x1 = obj["y0"], obj["y1"], obj["x0"], obj["x1"]
    m[y0:y1, x0:x1] = lab[y0:y1, x0:x1] == obj["label"]
    return mask_util.encode(m)


def disk_geometry(image: np.ndarray) -> tuple[float, float, float]:
    """Centre (cy, cx) and radius of the solar disk, from the largest bright component."""
    small = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if n > 1 else 0
    area = stats[k, cv2.CC_STAT_AREA] * 16.0
    cx, cy = cent[k] * 4.0
    return float(cy), float(cx), float(np.sqrt(area / np.pi))


def add_features(objs: list[dict], p: np.ndarray, lab: np.ndarray, image: np.ndarray, ring: int = 12) -> None:
    """Adds shape, contrast, position and context features to each instance record (in place)."""
    cy, cx, radius = disk_geometry(image)
    img = image.astype(np.float32)
    h, w = lab.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    for o in objs:
        y0, y1 = max(0, o["y0"] - ring), min(h, o["y1"] + ring)
        x0, x1 = max(0, o["x0"] - ring), min(w, o["x1"] + ring)
        inside = lab[y0:y1, x0:x1] == o["label"]
        around = cv2.dilate(inside.astype(np.uint8), kernel).astype(bool) & (lab[y0:y1, x0:x1] == 0)
        crop = img[y0:y1, x0:x1]
        pin = p[y0:y1, x0:x1][inside]
        o["i_in"] = float(crop[inside].mean())
        o["i_min"] = float(np.percentile(crop[inside], 5))
        o["i_ring"] = float(crop[around].mean()) if around.any() else o["i_in"]
        o["contrast"] = o["i_ring"] - o["i_in"]
        o["core80"] = float((pin > 0.8).mean())
        o["core90"] = float((pin > 0.9).mean())
        o["p_ring"] = float(p[y0:y1, x0:x1][around].mean()) if around.any() else 0.0
        o["mass"] = float(pin.sum())
        ys, xs = np.nonzero(inside)
        ys = ys + y0
        xs = xs + x0
        cov = np.cov(np.stack([ys, xs])) if len(ys) > 2 else np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1].clip(1e-3)
        o["major"] = float(4 * np.sqrt(ev[0]))
        o["elong"] = float(np.sqrt(ev[0] / ev[1]))
        o["fill"] = o["area"] / float((o["y1"] - o["y0"]) * (o["x1"] - o["x0"]))
        o["rho"] = float(np.hypot(ys.mean() - cy, xs.mean() - cx) / radius)
        o["n_parts"] = cv2.connectedComponents(inside.astype(np.uint8), connectivity=8)[0] - 1
        o["length"] = float(skeletonize(inside).sum())
        o["width"] = o["area"] / max(o["length"], 1.0)
        o["ring_p30"] = float((p[y0:y1, x0:x1][around] > 0.3).mean()) if around.any() else 0.0
        o["cy"], o["cx"] = float(ys.mean()), float(xs.mean())
    means = np.array([o["mean"] for o in objs])
    cent = np.array([[o["cy"], o["cx"]] for o in objs])
    for k, o in enumerate(objs):
        o["n_inst"] = len(objs)
        o["rank_mean"] = float((means > o["mean"]).sum())
        dist = np.hypot(*(cent - cent[k]).T)
        dist[k] = np.inf
        o["nn_dist"] = float(dist.min()) if len(objs) > 1 else 3000.0


def extract(prob: np.ndarray | None, cfg: PPConfig, p_full: np.ndarray | None = None,
            image: np.ndarray | None = None) -> tuple[list[dict], list[dict]]:
    """Returns (rles, instance records) for one image; records get extra features if ``image`` is given."""
    p = upsample(prob) if p_full is None else p_full
    n, lab = label_instances(p, cfg.t_seed, cfg.t_mask, cfg.link, cfg.close)
    objs = instance_table(p, lab, n)
    keep = [o for o in objs if o["area"] >= cfg.min_area and o[cfg.score] >= cfg.min_score]
    if image is not None and keep:
        add_features(keep, p, lab, image)
    return [encode_instance(lab, o) for o in keep], keep
