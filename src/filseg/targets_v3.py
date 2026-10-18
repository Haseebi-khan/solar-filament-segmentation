"""v3: extra training targets derived only from the annotation polygons (rule C2).

For one image with K annotation sets:

  y  soft mask      fraction of the K sets whose polygons cover a pixel (as v1)
  T  soft skeleton  per set: union over polygons of skeletonize(polygon) dilated by a disk of radius 2;
                    then the mean over the K sets
  D  soft core      per set and polygon: Euclidean distance to the polygon boundary divided by its maximum
                    inside that polygon (0 at the edge, 1 on the medial axis); then the mean over the K sets
  L  instance maps  one int16 label map per set (polygon k of the set -> label k + 1, 0 = background)

Polygons are rasterised with pycocotools exactly as the ground truth is, at the native 2048 px.
``spine``, ``bbox``, ``area`` and ``category_id`` of the annotation file are never read.
"""
from __future__ import annotations

import cv2
import numpy as np
from pycocotools import mask as mask_util
from scipy.ndimage import distance_transform_edt, find_objects
from skimage.morphology import skeletonize

from .data import H, W, AnnotationSet

SKELETON_RADIUS = 2


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def set_label_map(aset: AnnotationSet) -> np.ndarray:
    """int16 (2048, 2048): polygon k -> k + 1. Where polygons of one set overlap, the later one wins."""
    lab = np.zeros((H, W), np.int16)
    for k, rle in enumerate(aset.rles(), start=1):
        lab[mask_util.decode(rle) > 0] = k
    return lab


def set_skeleton_and_core(lab: np.ndarray, radius: int = SKELETON_RADIUS) -> tuple[np.ndarray, np.ndarray]:
    """Tubed skeleton (bool) and normalised core map (float32) of one set's label map."""
    skel = np.zeros(lab.shape, bool)
    core = np.zeros(lab.shape, np.float32)
    pad = radius + 2
    n = int(lab.max())
    for k, sl in enumerate(find_objects(lab, max_label=n), start=1):
        if sl is None:
            continue
        y0, y1 = max(0, sl[0].start - pad), min(lab.shape[0], sl[0].stop + pad)
        x0, x1 = max(0, sl[1].start - pad), min(lab.shape[1], sl[1].stop + pad)
        m = lab[y0:y1, x0:x1] == k
        if not m.any():
            continue
        sk = skeletonize(m)
        if radius > 0:
            sk = cv2.dilate(sk.astype(np.uint8), _disk(radius)).astype(bool)
        skel[y0:y1, x0:x1] |= sk
        dt = distance_transform_edt(m).astype(np.float32)
        mx = float(dt.max())
        if mx > 0:
            sub = core[y0:y1, x0:x1]
            np.maximum(sub, np.where(m, dt / mx, 0.0).astype(np.float32), out=sub)
    return skel, core


def image_targets(sets: list[AnnotationSet]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Returns y, T, D (float32 2048x2048 in [0, 1]) and the per-set label maps."""
    y = np.zeros((H, W), np.float32)
    t = np.zeros((H, W), np.float32)
    d = np.zeros((H, W), np.float32)
    labs = []
    for s in sets:
        lab = set_label_map(s)
        labs.append(lab)
        y += lab > 0
        sk, core = set_skeleton_and_core(lab)
        t += sk
        d += core
    k = max(1, len(sets))
    return y / k, t / k, d / k, labs


def area_resize(a: np.ndarray, size: int) -> np.ndarray:
    if a.shape[0] == size:
        return a
    return cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)


def to_uint8(a: np.ndarray) -> np.ndarray:
    return np.round(np.clip(a, 0, 1) * 255).astype(np.uint8)


def downsample_labels(lab: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour label map at ``size`` (pixel centres, like grid_sample with mode='nearest')."""
    if lab.shape[0] == size:
        return lab
    return cv2.resize(lab, (size, size), interpolation=cv2.INTER_NEAREST_EXACT)
