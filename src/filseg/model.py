"""Network construction (segmentation_models_pytorch U-Net family, 1-channel input)."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch


def build_model(arch: str = "unet", encoder: str = "resnet34", pretrained: bool = True,
                classes: int = 1) -> torch.nn.Module:
    weights = "imagenet" if pretrained else None
    kw = dict(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=classes)
    if arch == "unet":
        return smp.Unet(decoder_attention_type="scse", **kw)
    if arch == "unetpp":
        return smp.UnetPlusPlus(decoder_attention_type="scse", **kw)
    if arch == "fpn":
        return smp.FPN(**kw)
    raise ValueError(arch)


def load_checkpoint(path: str, device="cuda") -> tuple[torch.nn.Module, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["arch"], cfg["encoder"], pretrained=False, classes=cfg.get("classes", 1))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg
