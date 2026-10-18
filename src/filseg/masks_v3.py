"""v3: compact instance masks (tight bounding box + boolean crop) and exact pixel-set operations on them.

IoUs computed here are identical to pycocotools' IoU of the same masks (both count pixels), but avoid
encoding full 2048x2048 RLEs for every intermediate candidate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pycocotools import mask as mask_util

H = W = 2048


@dataclass
class CropMask:
    y0: int
    x0: int
    m: np.ndarray            # bool (h, w)
    area: int = -1

    def __post_init__(self):
        if self.area < 0:
            self.area = int(self.m.sum())

    @property
    def y1(self) -> int:
        return self.y0 + self.m.shape[0]

    @property
    def x1(self) -> int:
        return self.x0 + self.m.shape[1]

    @classmethod
    def from_full(cls, full: np.ndarray) -> "CropMask":
        ys, xs = np.nonzero(full.any(1))[0], np.nonzero(full.any(0))[0]
        if len(ys) == 0:
            return cls(0, 0, np.zeros((0, 0), bool), 0)
        y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
        return cls(int(y0), int(x0), np.ascontiguousarray(full[y0:y1, x0:x1].astype(bool)))

    @classmethod
    def from_label(cls, lab: np.ndarray, k: int, sl: tuple) -> "CropMask":
        return cls(sl[0].start, sl[1].start, lab[sl] == k).tight()

    @classmethod
    def from_rle(cls, rle: dict) -> "CropMask":
        x, y, w, h = (int(round(v)) for v in mask_util.toBbox(rle))
        if w == 0 or h == 0:
            return cls(0, 0, np.zeros((0, 0), bool), 0)
        full = mask_util.decode(rle)
        return cls(y, x, full[y:y + h, x:x + w].astype(bool)).tight()

    def tight(self) -> "CropMask":
        if self.area == 0:
            return CropMask(0, 0, np.zeros((0, 0), bool), 0)
        rows, cols = np.nonzero(self.m.any(1))[0], np.nonzero(self.m.any(0))[0]
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        if r0 == 0 and c0 == 0 and r1 == self.m.shape[0] and c1 == self.m.shape[1]:
            return self
        return CropMask(self.y0 + int(r0), self.x0 + int(c0), np.ascontiguousarray(self.m[r0:r1, c0:c1]), self.area)

    def full(self, shape=(H, W)) -> np.ndarray:
        out = np.zeros(shape, bool)
        out[self.y0:self.y1, self.x0:self.x1] = self.m
        return out

    def to_rle(self, shape=(H, W)) -> dict:
        out = np.zeros(shape, np.uint8, order="F")
        out[self.y0:self.y1, self.x0:self.x1] = self.m
        return mask_util.encode(out)

    def packed(self) -> tuple:
        return (self.y0, self.x0, self.m.shape, np.packbits(self.m, axis=None), self.area)

    @classmethod
    def unpack(cls, t: tuple) -> "CropMask":
        y0, x0, shape, bits, area = t
        n = int(np.prod(shape))
        return cls(y0, x0, np.unpackbits(bits, count=n).astype(bool).reshape(shape), area)

    def centroid(self) -> tuple[float, float]:
        ys, xs = np.nonzero(self.m)
        return float(ys.mean() + self.y0), float(xs.mean() + self.x0)


def intersection(a: CropMask, b: CropMask) -> int:
    y0, y1 = max(a.y0, b.y0), min(a.y1, b.y1)
    x0, x1 = max(a.x0, b.x0), min(a.x1, b.x1)
    if y0 >= y1 or x0 >= x1:
        return 0
    return int(np.count_nonzero(a.m[y0 - a.y0:y1 - a.y0, x0 - a.x0:x1 - a.x0]
                                & b.m[y0 - b.y0:y1 - b.y0, x0 - b.x0:x1 - b.x0]))


def iou(a: CropMask, b: CropMask) -> float:
    i = intersection(a, b)
    return i / (a.area + b.area - i) if i else 0.0


def _boxes(ms: list[CropMask]) -> np.ndarray:
    return np.array([[m.y0, m.y1, m.x0, m.x1] for m in ms], np.int64).reshape(len(ms), 4)


def inter_matrix(a: list[CropMask], b: list[CropMask]) -> np.ndarray:
    """Pixel intersections |a_i & b_j|, only computed for overlapping bounding boxes."""
    out = np.zeros((len(a), len(b)), np.int64)
    if not a or not b:
        return out
    ba, bb = _boxes(a), _boxes(b)
    ov = ((np.maximum(ba[:, None, 0], bb[None, :, 0]) < np.minimum(ba[:, None, 1], bb[None, :, 1]))
          & (np.maximum(ba[:, None, 2], bb[None, :, 2]) < np.minimum(ba[:, None, 3], bb[None, :, 3])))
    for i, j in zip(*np.nonzero(ov)):
        out[i, j] = intersection(a[i], b[j])
    return out


def iou_from_inter(inter: np.ndarray, a: list[CropMask], b: list[CropMask]) -> np.ndarray:
    if inter.size == 0:
        return np.zeros(inter.shape)
    aa = np.array([m.area for m in a], np.float64)[:, None]
    bb = np.array([m.area for m in b], np.float64)[None, :]
    union = aa + bb - inter
    return np.where(inter > 0, inter / np.maximum(union, 1), 0.0)


def iou_matrix_crops(a: list[CropMask], b: list[CropMask]) -> np.ndarray:
    return iou_from_inter(inter_matrix(a, b), a, b)


def subtract(a: CropMask, occupied: np.ndarray) -> CropMask:
    """a minus the pixels set in the full-size boolean array ``occupied``."""
    m = a.m & ~occupied[a.y0:a.y1, a.x0:a.x1]
    return CropMask(a.y0, a.x0, m).tight()


def paste(full: np.ndarray, a: CropMask) -> None:
    full[a.y0:a.y1, a.x0:a.x1] |= a.m


def masks_from_label(lab: np.ndarray, n: int) -> dict[int, CropMask]:
    from scipy.ndimage import find_objects
    out = {}
    for k, sl in enumerate(find_objects(lab, max_label=n), start=1):
        if sl is not None:
            cm = CropMask.from_label(lab, k, sl)
            if cm.area:
                out[k] = cm
    return out


def gt_crops(gt_rles: list[list[dict]]) -> list[list[CropMask]]:
    """Ground-truth sets of one image (list of RLE lists) as crop masks."""
    return [[CropMask.from_rle(r) for r in rles] for rles in gt_rles]
