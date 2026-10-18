"""v3 losses for soft multi-annotator targets.

All functions take probabilities or logits of shape (B, 1, H, W) in float32 and return a scalar (mean over
the images of the batch). Sums run over the pixels of one image; eps = 1.

  jml1   Jaccard metric loss (Wang, Ning, Blaschko, NeurIPS 2023):
             2 |P - y|_1 / (|P|_1 + |y|_1 + |P - y|_1 + eps)
         equals the soft Jaccard loss on hard labels and, unlike it, is minimised at P = y for soft labels.
  dml1   Dice semimetric loss (Wang et al., MICCAI 2024):  |P - y|_1 / (|P|_1 + |y|_1 + eps)
         equals the soft Dice loss on hard labels.
  blob   instance-balanced JML1 (Kofler et al., IPMI 2023): one term per GT instance i of one annotation set,
             JML1(P * M_i, G_i),  M_i = 1 - union of the other instances,
         averaged over instances, so every filament counts equally (as in PQ). Computed in closed form from
         per-instance sums (see blob_loss), exact when the instances of one set do not overlap.
  skeleton_recall  1 - sum(P T) / (sum(T) + eps)  (Kirchhoff et al., ECCV 2024); ablation only.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

EPS = 1.0


def _sum(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).sum(1)


def jml1(p: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    diff = _sum((p - y).abs())
    return (2 * diff / (_sum(p) + _sum(y) + diff + eps)).mean()


def dml1(p: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return (_sum((p - y).abs()) / (_sum(p) + _sum(y) + eps)).mean()


def soft_dice(p: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """v1's soft Dice loss (mis-specified for soft labels; kept for the ablation baseline)."""
    inter = _sum(p * y)
    return (1 - (2 * inter + eps) / (_sum(p) + _sum(y) + eps)).mean()


REGION = {"jml": jml1, "dml": dml1, "dice": soft_dice}


def area_down(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.shape[-1] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="area")


def blob_loss(p: torch.Tensor, inst: torch.Tensor, n_max: int, eps: float = EPS) -> torch.Tensor:
    """p: (B, 1, h, w) probabilities; inst: (B, h, w) int64 instance labels 0..n_max of one sampled set.

    With disjoint instances, for instance i with pixel set G_i, A = sum of P over the background
    (pixels of no instance), s_i = sum of P over G_i and a_i = |G_i|:
        |P M_i|_1 = A + s_i,   |G_i|_1 = a_i,   |P M_i - G_i|_1 = A + a_i - s_i
    so JML1(P M_i, G_i) = 2 (A + a_i - s_i) / (2 A + 2 a_i + eps). Images without instances are skipped.
    """
    b = p.shape[0]
    k = n_max + 1
    lab = inst.clamp(0, n_max).flatten(1) + k * torch.arange(b, device=p.device)[:, None]
    s = torch.zeros(b * k, device=p.device, dtype=torch.float32).scatter_add_(0, lab.flatten(), p.flatten().float())
    a = torch.bincount(lab.flatten(), minlength=b * k).float()
    s, a = s.view(b, k), a.view(b, k)
    bg = s[:, :1]
    s, a = s[:, 1:], a[:, 1:]
    valid = a > 0
    term = 2 * (bg + a - s) / (2 * bg + 2 * a + eps)
    n = valid.sum(1)
    per_image = (term * valid).sum(1) / n.clamp(min=1)
    has = n > 0
    if not has.any():
        return p.sum() * 0.0
    return per_image[has].mean()


def skeleton_recall(p: torch.Tensor, t: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return (1 - _sum(p * t) / (_sum(t) + eps)).mean()


def core_loss(d_hat: torch.Tensor, d: torch.Tensor, y: torch.Tensor, margin: int = 4) -> torch.Tensor:
    """SmoothL1 on pixels with y > 0 or within ``margin`` px of y > 0."""
    region = F.max_pool2d((y > 0).float(), 2 * margin + 1, stride=1, padding=margin)
    diff = F.smooth_l1_loss(d_hat, d, reduction="none", beta=0.1)
    return ((diff * region).flatten(1).sum(1) / region.flatten(1).sum(1).clamp(min=1)).mean()


@dataclass
class LossWeights:
    region: str = "jml"       # jml | dml | dice (v1)
    w_region: float = 1.0
    w_blob: float = 0.5
    w_s: float = 0.5
    w_d: float = 0.25
    w_ds: float = 1.0         # multiplies 0.25 (stride 4) and 0.125 (stride 8)
    w_sr: float = 0.0         # skeleton recall: ablation only (default off)
    blob_size: int = 1024


def supervised_loss(out: dict, y: torch.Tensor, t: torch.Tensor | None, d: torch.Tensor | None,
                    inst: torch.Tensor | None, n_max: int, w: LossWeights) -> tuple[torch.Tensor, dict]:
    """Total v3 loss for one labelled batch. ``out`` holds float32 logits: p, optional s, d, ds (list)."""
    region = REGION[w.region]
    lp = out["p"]
    p = torch.sigmoid(lp)
    parts = {"bce": F.binary_cross_entropy_with_logits(lp, y)}
    if w.w_region:
        parts["region"] = w.w_region * region(p, y)
    if w.w_blob and inst is not None:
        parts["blob"] = w.w_blob * blob_loss(area_down(p, w.blob_size), inst, n_max)
    if w.w_sr and t is not None:
        parts["sr"] = w.w_sr * skeleton_recall(p, t)
    if "s" in out and t is not None and w.w_s:
        ls = out["s"]
        parts["skel"] = w.w_s * (F.binary_cross_entropy_with_logits(ls, t) + dml1(torch.sigmoid(ls), t))
    if "d" in out and d is not None and w.w_d:
        parts["core"] = w.w_d * core_loss(torch.sigmoid(out["d"]), d, y)
    if out.get("ds") and w.w_ds:
        total_ds = 0.0
        for coef, logit in zip((0.25, 0.125), out["ds"]):
            yd = area_down(y, logit.shape[-1])
            total_ds = total_ds + coef * (F.binary_cross_entropy_with_logits(logit, yd)
                                          + region(torch.sigmoid(logit), yd))
        parts["ds"] = w.w_ds * total_ds
    total = sum(parts.values())
    return total, {k: float(v.detach()) for k, v in parts.items()}


def unlabeled_loss(out: dict, y_pl: torch.Tensor, t_pl: torch.Tensor | None, weight: torch.Tensor | None = None
                   ) -> torch.Tensor:
    """Noisy-Student term on soft pseudo-labels: BCE + JML1 on P, plus 0.5 BCE on S. weight: (B,) per frame."""
    lp = out["p"]
    p = torch.sigmoid(lp)
    bce = F.binary_cross_entropy_with_logits(lp, y_pl, reduction="none").flatten(1).mean(1)
    diff = _sum((p - y_pl).abs())
    jml = 2 * diff / (_sum(p) + _sum(y_pl) + diff + EPS)
    per = bce + jml
    if "s" in out and t_pl is not None:
        per = per + 0.5 * F.binary_cross_entropy_with_logits(out["s"], t_pl, reduction="none").flatten(1).mean(1)
    if weight is not None:
        return (per * weight).sum() / weight.sum().clamp(min=1e-6)
    return per.mean()
