"""Batched augmentation on the GPU (affine warp + photometric jitter).

The whole training set lives on the GPU as uint8 tensors, so a batch is built
by indexing, warping with ``grid_sample`` and jittering intensities. This keeps
the MI300X busy without a CPU data-loading pipeline.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# GONG H-alpha JPEGs are already normalised: solar-disk mean ~125, std ~17.
IMG_MEAN = 125.0 / 255.0
IMG_STD = 18.0 / 255.0


def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    return (x_uint8.float() / 255.0 - IMG_MEAN) / IMG_STD


def random_affine(n: int, device, crop_frac: float = 1.0, rotate: bool = True,
                  scale: tuple[float, float] = (0.87, 1.15), shift: float = 0.03) -> torch.Tensor:
    """Affine matrices mapping output coords -> input coords (normalised [-1, 1]).

    crop_frac < 1 samples a random window covering ``crop_frac`` of the image side
    (used when training on crops of a higher-resolution image).
    """
    ang = torch.rand(n, device=device) * 2 * math.pi if rotate else torch.zeros(n, device=device)
    s = torch.empty(n, device=device).uniform_(*scale)
    flip = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(ang), torch.sin(ang)
    k = crop_frac / s
    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0] = k * cos * flip
    theta[:, 0, 1] = -k * sin
    theta[:, 1, 0] = k * sin * flip
    theta[:, 1, 1] = k * cos
    span = max(1.0 - crop_frac, 0.0)
    theta[:, :, 2] = (torch.rand(n, 2, device=device) * 2 - 1) * (span + shift)
    return theta


def warp(x: torch.Tensor, theta: torch.Tensor, out_size: int, mode: str = "bilinear") -> torch.Tensor:
    grid = F.affine_grid(theta, (x.shape[0], x.shape[1], out_size, out_size), align_corners=False)
    return F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=False)


def photometric(x: torch.Tensor) -> torch.Tensor:
    """x: normalised images (B, 1, H, W). Random gamma/contrast/brightness, blur, noise."""
    b = x.shape[0]
    dev = x.device
    # gamma on the [0, 1] scale
    u = (x * IMG_STD + IMG_MEAN).clamp(0, 1)
    gamma = torch.exp(torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.35, 0.35))
    u = u.pow(gamma)
    x = (u - IMG_MEAN) / IMG_STD
    contrast = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.75, 1.3)
    bright = torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.4, 0.4)
    x = x * contrast + bright
    # occasional mild blur (seeing variations)
    blur = torch.rand(b, device=dev) < 0.25
    if blur.any():
        k = torch.tensor([1., 4., 6., 4., 1.], device=dev)
        k = (k[:, None] * k[None, :]) / 256.0
        xb = F.conv2d(x[blur], k[None, None], padding=2)
        x = x.clone()
        x[blur] = xb
    noise = torch.rand(b, 1, 1, 1, device=dev) * 0.15
    x = x + noise * torch.randn_like(x)
    return x


def make_batch(images: torch.Tensor, targets: torch.Tensor, idx: torch.Tensor, out_size: int,
               crop_frac: float = 1.0, rotate: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """images/targets: uint8 (N, 1, S, S) on GPU. Returns normalised image batch and soft target in [0,1]."""
    x = normalize(images[idx])
    y = targets[idx].float() / 255.0
    theta = random_affine(len(idx), images.device, crop_frac=crop_frac, rotate=rotate)
    # pad value for the image is "black sky", i.e. normalised 0 intensity
    black = (0.0 - IMG_MEAN) / IMG_STD
    x = warp(x - black, theta, out_size) + black
    y = warp(y, theta, out_size)
    x = photometric(x)
    return x, y
