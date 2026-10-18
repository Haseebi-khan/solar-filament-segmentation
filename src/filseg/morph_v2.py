"""v2: grow or shrink the final filament masks of one image by r pixels, keeping them disjoint.

Used by the leaderboard probes (tools/morph_submission_v2.py) and, only if a probe shows that the
test masks are drawn systematically thinner or wider than the training masks, by
make_submission_v2.py --morph. Out of fold, r = 0 is best (the training masks are matched as is).
"""
from __future__ import annotations

import cv2
import numpy as np
from pycocotools import mask as mask_util


def morph_rles(rles: list[dict], r: int) -> list[dict]:
    """+r dilates, -r erodes every mask. Dilation only claims background pixels that exactly one
    filament reaches; filaments that erode away are dropped."""
    if r == 0 or not rles:
        return rles
    masks = mask_util.decode(rles)  # (H, W, n)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(r) + 1, 2 * abs(r) + 1))
    lab = np.zeros(masks.shape[:2], np.int32)
    for i in range(masks.shape[2]):
        lab[masks[:, :, i] > 0] = i + 1
    op = cv2.dilate if r > 0 else cv2.erode
    grown = [op(np.ascontiguousarray(masks[:, :, i]), k).astype(bool) for i in range(masks.shape[2])]
    if r > 0:
        reach = np.sum(grown, axis=0)
        grown = [g & ((lab == i + 1) | ((lab == 0) & (reach == 1))) for i, g in enumerate(grown)]
    return [mask_util.encode(np.asfortranarray(g.astype(np.uint8))) for g in grown if g.any()]
