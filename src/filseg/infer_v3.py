"""v3 full-disk inference with dihedral TTA for FilaNet models (P, S, D heads)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .augment import normalize
from .augment_v3 import to_inputs
from .infer import _dihedral, _dihedral_inv


@torch.no_grad()
def predict_maps(models: list, images, tta: int = 8, batch_size: int = 2, out_size: int | None = None,
                 heads: tuple[str, ...] = ("p",), views: tuple[int, ...] | None = None) -> dict[str, np.ndarray]:
    """images: uint8 (N, 2048, 2048) (array or memmap). All models must share size/out_size (fold models of one
    member). Returns {head: float16 (N, out, out)} averaged over models and the dihedral views ``views``
    (default: the first ``tta``; 0-3 = rotations by k*90 deg, 4-7 = horizontal flip then rotation)."""
    views = tuple(range(tta)) if views is None else tuple(views)
    m0 = models[0]
    size, native_out = m0.size, m0.out_size
    out_size = out_size or native_out
    channels_last = m0.cfg.model != "vit_dpt"
    res = {h: np.zeros((len(images), out_size, out_size), np.float16) for h in heads}
    for i0 in range(0, len(images), batch_size):
        chunk = np.stack([np.asarray(im) for im in images[i0:i0 + batch_size]])
        x_full = normalize(torch.from_numpy(chunk).cuda()[:, None])
        x_in = to_inputs(x_full, size)
        acc = {h: torch.zeros(len(chunk), 1, native_out, native_out, device="cuda") for h in heads}
        for model in models:
            for k in views:
                xi = _dihedral(x_in, k)
                xf = _dihedral(x_full, k) if model.hr else None
                if channels_last:
                    xi = xi.contiguous(memory_format=torch.channels_last)
                    xf = xf.contiguous(memory_format=torch.channels_last) if xf is not None else None
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(xi, xf)
                for h in heads:
                    acc[h] += _dihedral_inv(torch.sigmoid(out[h].float()), k)
        for h in heads:
            a = acc[h] / (len(models) * len(views))
            if out_size != native_out:
                a = F.interpolate(a, size=(out_size, out_size), mode="bilinear", align_corners=False)
            res[h][i0:i0 + len(chunk)] = a[:, 0].cpu().numpy().astype(np.float16)
    return res
