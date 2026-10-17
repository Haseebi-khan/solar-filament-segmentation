"""Full-disk inference with dihedral test-time augmentation."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .augment import normalize


def _dihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, k % 4, dims=(-2, -1))


def _dihedral_inv(x: torch.Tensor, k: int) -> torch.Tensor:
    x = torch.rot90(x, -(k % 4), dims=(-2, -1))
    if k >= 4:
        x = x.flip(-1)
    return x


def resize_uint8(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


@torch.no_grad()
def predict_probs(models: list[torch.nn.Module], images: np.ndarray, net_size: int, tta: int = 8,
                  batch_size: int = 4, out_size: int | None = None, scales: tuple[float, ...] = (1.0,)) -> np.ndarray:
    """images: uint8 (N, 2048, 2048). Returns float16 probabilities (N, out, out), out = net_size by default.

    Probabilities are averaged over models, the first ``tta`` dihedral transforms and the
    given input scales (each rounded to a multiple of 32 px).
    """
    out_size = out_size or net_size
    res = np.zeros((len(images), out_size, out_size), np.float16)
    sizes = [int(round(net_size * s / 32)) * 32 for s in scales]
    for i0 in range(0, len(images), batch_size):
        acc = None
        for size in sizes:
            chunk = np.stack([resize_uint8(np.asarray(im), size) for im in images[i0:i0 + batch_size]])
            x = normalize(torch.from_numpy(chunk).cuda()[:, None])
            part = torch.zeros_like(x)
            for model in models:
                for k in range(tta):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logit = model(_dihedral(x, k).contiguous(memory_format=torch.channels_last))
                    part += _dihedral_inv(torch.sigmoid(logit.float()), k)
            if size != net_size:
                part = F.interpolate(part, size=(net_size, net_size), mode="bilinear", align_corners=False)
            acc = part if acc is None else acc + part
        acc /= len(models) * tta * len(sizes)
        if out_size != net_size:
            acc = F.interpolate(acc, size=(out_size, out_size), mode="bilinear", align_corners=False)
        res[i0:i0 + len(chunk)] = acc[:, 0].cpu().numpy().astype(np.float16)
    return res
