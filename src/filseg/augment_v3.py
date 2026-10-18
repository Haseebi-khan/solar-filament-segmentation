"""v3 batch construction on the GPU: one affine warp shared by the image, the soft targets (y, T, D) and the
instance maps of one randomly sampled annotation set, plus photometric jitter (v1 strength, or the stronger
"student" noise of Stage B with a low-frequency multiplicative haze field).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .augment import IMG_MEAN, IMG_STD, normalize, random_affine, warp

BLACK = (0.0 - IMG_MEAN) / IMG_STD


def _blur(x: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    if not sel.any():
        return x
    k = torch.tensor([1., 4., 6., 4., 1.], device=x.device)
    k = (k[:, None] * k[None, :]) / 256.0
    x = x.clone()
    x[sel] = F.conv2d(x[sel], k[None, None].to(x.dtype), padding=2)
    return x


def photometric_v3(x: torch.Tensor, strong: bool = False) -> torch.Tensor:
    """x: normalised (B, 1, H, W). v1 jitter; ``strong`` widens gamma/contrast/noise and adds haze."""
    b, dev = x.shape[0], x.device
    g_lim, c_lo, c_hi, n_max = (0.5, 0.65, 1.45, 0.25) if strong else (0.35, 0.75, 1.3, 0.15)
    u = (x * IMG_STD + IMG_MEAN).clamp(0, 1)
    u = u.pow(torch.exp(torch.empty(b, 1, 1, 1, device=dev).uniform_(-g_lim, g_lim)))
    if strong:  # low-frequency multiplicative haze, amplitude up to 10 %
        amp = torch.rand(b, 1, 1, 1, device=dev) * 0.10
        field = F.interpolate(torch.rand(b, 1, 6, 6, device=dev) * 2 - 1, size=x.shape[-2:], mode="bicubic",
                              align_corners=False)
        u = (u * (1 + amp * field.clamp(-1, 1))).clamp(0, 1)
    x = (u - IMG_MEAN) / IMG_STD
    x = x * torch.empty(b, 1, 1, 1, device=dev).uniform_(c_lo, c_hi) + torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.4, 0.4)
    x = _blur(x, torch.rand(b, device=dev) < (0.35 if strong else 0.25))
    return x + torch.rand(b, 1, 1, 1, device=dev) * n_max * torch.randn_like(x)


def to_inputs(x_full: torch.Tensor, size: int) -> torch.Tensor:
    """Encoder input: area-resize the (normalised) full-resolution image to ``size``."""
    if x_full.shape[-1] == size:
        return x_full
    return F.interpolate(x_full, size=(size, size), mode="area")


def make_batch_v3(img: torch.Tensor, idx: torch.Tensor, in_size: int, out_size: int, need_full: bool,
                  tgt: torch.Tensor | None = None, inst: torch.Tensor | None = None,
                  n_sets: torch.Tensor | None = None, inst_size: int = 1024, rotate: bool = True,
                  strong: bool = False) -> dict:
    """Builds one augmented batch.

    img   uint8 (N, 1, S_img, S_img): native 2048 images if ``need_full`` (HR stem), else images already
          area-resized to ``in_size`` (as v1)
    tgt   uint8 (N, C, S_out, S_out) soft targets (y, T, D, ...), warped bilinearly to ``out_size``
    inst  int16 (N, K, L, L) instance maps of every annotation set; one set per image is sampled uniformly
          among its ``n_sets`` and warped with nearest sampling to ``inst_size``
    Returns {"x_in", "x_full" (or None), "tgt" (float in [0, 1]), "inst" (int64 (B, L, L))}.
    """
    b = len(idx)
    theta = random_affine(b, img.device, rotate=rotate)
    x = normalize(img[idx])
    work = img.shape[-1] if need_full else in_size
    x = warp(x - BLACK, theta, work) + BLACK
    x = photometric_v3(x, strong=strong)
    out = {"x_full": x if need_full else None, "x_in": to_inputs(x, in_size)}
    if tgt is not None:
        out["tgt"] = warp(tgt[idx].float() / 255.0, theta, out_size)
    if inst is not None:
        k = (torch.rand(b, device=img.device) * n_sets[idx].float()).long().clamp(max=inst.shape[1] - 1)
        lab = inst[idx, k][:, None].float()
        out["inst"] = warp(lab, theta, inst_size, mode="nearest")[:, 0].round().long()
    return out


def warp_unlabeled(img: torch.Tensor, pl: torch.Tensor, in_size: int, out_size: int, need_full: bool,
                   rotate: bool = True, strong: bool = True) -> dict:
    """Unlabelled half of a student batch. img uint8 (B, 1, 2048, 2048) (or at in_size), pl uint8 (B, C, S, S)."""
    theta = random_affine(img.shape[0], img.device, rotate=rotate)
    x = normalize(img)
    work = img.shape[-1] if need_full else in_size
    x = warp(x - BLACK, theta, work) + BLACK
    x = photometric_v3(x, strong=strong)
    return {"x_full": x if need_full else None, "x_in": to_inputs(x, in_size),
            "tgt": warp(pl.float() / 255.0, theta, out_size)}
