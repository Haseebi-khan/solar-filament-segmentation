"""v3 networks.

Three families share one decoder vocabulary (Up blocks with GroupNorm, GELU and scSE attention) and the same
heads: P (filament probability), S (soft skeleton), D (soft core), plus deep-supervision logits for P.

  cnx_unet  U-Net on a timm ConvNeXt encoder (M1: convnext_large.dinov3_lvd1689m at native 2048 px)
  vit_dpt   DPT decoder on a timm DINOv3 ViT (M2: vit_large_patch16_dinov3.lvd1689m at 1536 px)
  smp_unet  U-Net on a segmentation_models_pytorch encoder (M3: resnet34 at 1536 px, v1 encoder weights)

With ``hr_stem`` a full-resolution detail branch runs on the native 2048 px image (stride 1 and 2 features)
and the decoder ends at 2048 px whatever the encoder resolution; otherwise the output has the encoder input
resolution. Inputs are 1-channel images normalised as in v1. Every forward returns float logits in a dict.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

NATIVE = 2048


@dataclass
class ModelConfig:
    model: str = "smp_unet"                  # smp_unet | cnx_unet | vit_dpt
    backbone: str = "resnet34"
    size: int = 1536                         # encoder input resolution (full disk, area-resized)
    hr_stem: bool = False                    # detail branch at stride 1 and 2 (skip connections for thin filaments)
    stem_native: bool = True                 # True: stem on the native 2048 image, output 2048; False: stem at
                                             # ``size``, output ``size`` (3x cheaper, for the budget members)
    stem_ch: list = field(default_factory=lambda: [32, 64])   # channels of the stride-1 / stride-2 stem features
    edge_inputs: bool = False                # Sobel magnitude + LoG channels into the HR stem
    aux_heads: bool = True                   # S and D heads
    deep_sup: bool = True                    # deep-supervision logits (training only)
    drop_path: float = 0.0
    pretrained: bool = True
    vit_taps: list = field(default_factory=lambda: [5, 11, 17, 23])
    ckpt_stages: list = field(default_factory=list)   # ConvNeXt stages with gradient checkpointing
    ckpt_vit: bool = True                    # gradient checkpointing of all ViT blocks
    ckpt_hr: bool = False                    # gradient checkpointing of the full-resolution blocks

    @property
    def out_size(self) -> int:
        return NATIVE if (self.hr_stem and self.stem_native) else self.size


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if c % 8 == 0 else 1, c)


class ConvGNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), _gn(cout), nn.GELU())


class SCSE(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        h = max(c // r, 4)
        self.cse = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, h, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(h, c, 1), nn.Sigmoid())
        self.sse = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)


class Up(nn.Module):
    """Bilinear resize to the skip's size (or ``size``), concatenate the skip, two conv blocks, scSE."""

    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(ConvGNAct(c_in + c_skip, c_out), ConvGNAct(c_out, c_out))
        self.att = SCSE(c_out)

    def forward(self, x, skip=None, size=None):
        size = tuple(skip.shape[-2:]) if skip is not None else tuple(size)
        if tuple(x.shape[-2:]) != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip.to(x.dtype)], 1)
        return self.att(self.block(x))


class EdgeFilters(nn.Module):
    """Fixed Sobel-magnitude and Laplacian-of-Gaussian (sigma 1.5) channels appended to the image."""

    def __init__(self, sigma: float = 1.5):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        r = int(math.ceil(3 * sigma))
        ax = torch.arange(-r, r + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        q = (xx ** 2 + yy ** 2) / (2 * sigma ** 2)
        log = (q - 1) * torch.exp(-q)
        log = log - log.mean()
        log = log / log.abs().sum()
        self.register_buffer("kx", kx[None, None])
        self.register_buffer("ky", kx.t().contiguous()[None, None])
        self.register_buffer("klog", log[None, None])
        self.r = r

    def forward(self, x):
        gx = F.conv2d(x, self.kx.to(x.dtype), padding=1)
        gy = F.conv2d(x, self.ky.to(x.dtype), padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        lg = F.conv2d(x, self.klog.to(x.dtype), padding=self.r)
        return torch.cat([x, mag, lg], 1)


class HRStem(nn.Module):
    """Full-resolution detail branch: S1 (stride 1, 32 ch) and S2 (stride 2, 64 ch) of the native image."""

    def __init__(self, c1: int = 32, c2: int = 64, edges: bool = False):
        super().__init__()
        self.edges = EdgeFilters() if edges else None
        self.s1 = nn.Sequential(ConvGNAct(3 if edges else 1, c1), ConvGNAct(c1, c1))
        self.s2 = nn.Sequential(ConvGNAct(c1, c2, stride=2), ConvGNAct(c2, c2))

    def forward(self, x):
        if self.edges is not None:
            x = self.edges(x)
        s1 = self.s1(x)
        return s1, self.s2(s1)


class Heads(nn.Module):
    def __init__(self, c: int, n_out: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, n_out, 1))
        with torch.no_grad():  # start near the filament prior (~1% of pixels) for P and S, ~0.1 for D
            self.conv[-1].bias.copy_(torch.tensor([-4.0, -4.0, -2.0][:n_out]))

    def forward(self, x):
        return self.conv(x)


# ----------------------------------------------------------------------------------------------- encoders
class TimmConvEncoder(nn.Module):
    """timm CNN (ConvNeXt) returning its 4 stage outputs (strides 4, 8, 16, 32)."""

    def __init__(self, name: str, pretrained: bool, drop_path: float, ckpt_stages=()):
        super().__init__()
        import timm
        self.m = timm.create_model(name, pretrained=pretrained, in_chans=1, drop_path_rate=drop_path, num_classes=0)
        self.channels = [f["num_chs"] for f in self.m.feature_info][-4:]
        for i in ckpt_stages:
            self.m.stages[i].grad_checkpointing = True

    def forward(self, x):
        return self.m.forward_intermediates(x, indices=4, intermediates_only=True)


class SmpEncoder(nn.Module):
    """segmentation_models_pytorch encoder (v1's ImageNet ResNet-34): strides 2, 4, 8, 16, 32."""

    def __init__(self, name: str, pretrained: bool):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.m = smp.encoders.get_encoder(name, in_channels=1, depth=5, weights="imagenet" if pretrained else None)
        self.channels = list(self.m.out_channels[1:])

    def forward(self, x):
        return self.m(x)[1:]


class ViTEncoder(nn.Module):
    """DINOv3 ViT (timm Eva) returning the patch tokens of the tapped blocks as (B, C, H/16, W/16) maps."""

    def __init__(self, name: str, img_size: int, taps, pretrained: bool, drop_path: float, ckpt: bool):
        super().__init__()
        import timm
        self.m = timm.create_model(name, pretrained=pretrained, in_chans=1, drop_path_rate=drop_path,
                                   num_classes=0, img_size=img_size)
        if ckpt:
            self.m.set_grad_checkpointing(True)
        self.taps = list(taps)
        self.dim = self.m.embed_dim
        self.depth = len(self.m.blocks)

    def forward(self, x):
        return self.m.forward_intermediates(x, indices=self.taps, norm=True, output_fmt="NCHW",
                                            intermediates_only=True, stop_early=True)


class RCU(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.gelu(self.c1(F.gelu(x))))


class Fusion(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.rcu1, self.rcu2, self.out = RCU(c), RCU(c), nn.Conv2d(c, c, 1)

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.rcu1(skip)
        return self.out(self.rcu2(x))


class DPTNeck(nn.Module):
    """Reassemble 4 stride-16 token maps to strides 4/8/16/32 and fuse them top-down (Ranftl et al. 2021)."""

    def __init__(self, dim: int, c: int = 256):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(dim, c, 1) for _ in range(4)])
        self.resample = nn.ModuleList([nn.ConvTranspose2d(c, c, 4, 4), nn.ConvTranspose2d(c, c, 2, 2),
                                       nn.Identity(), nn.Conv2d(c, c, 3, 2, 1)])
        self.fuse = nn.ModuleList([Fusion(c) for _ in range(4)])

    def forward(self, feats):
        r = [res(p(f)) for f, p, res in zip(feats, self.proj, self.resample)]
        x = self.fuse[3](r[3])
        x = self.fuse[2](x, r[2])
        x8 = self.fuse[1](x, r[1])
        x4 = self.fuse[0](x8, r[0])
        return x4, x8


# ----------------------------------------------------------------------------------------------- full model
class FilaNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.size = cfg.size
        self.out_size = cfg.out_size
        self.hr = cfg.hr_stem
        self.full_input = cfg.hr_stem and cfg.stem_native   # the stem (and the output) are at 2048 px
        n_out = 3 if cfg.aux_heads else 1
        c1, c2 = cfg.stem_ch
        if self.hr:
            self.stem = HRStem(c1, c2, edges=cfg.edge_inputs)
        if cfg.model == "cnx_unet":
            self.encoder = TimmConvEncoder(cfg.backbone, cfg.pretrained, cfg.drop_path, cfg.ckpt_stages)
            c = self.encoder.channels
            self.ups = nn.ModuleList([Up(c[3], c[2], 512), Up(512, c[1], 256), Up(256, c[0], 128)])
            tap_ch = (128, 256)
            self.up2 = Up(128, c2 if self.hr else 0, c2 if self.hr else 64)
            self.up1 = Up(c2 if self.hr else 64, c1 if self.hr else 0, c1 if self.hr else 32)
            head_c = c1 if self.hr else 32
        elif cfg.model == "smp_unet":
            self.encoder = SmpEncoder(cfg.backbone, cfg.pretrained)
            c = self.encoder.channels  # strides 2, 4, 8, 16, 32
            self.ups = nn.ModuleList([Up(c[4], c[3], 256), Up(256, c[2], 128), Up(128, c[1], 64)])
            tap_ch = (64, 128)
            self.up_enc2 = Up(64, c[0], c2 if self.hr else 32)
            self.up2 = Up(c2, c2, c2) if self.hr else None
            self.up1 = Up(c2, c1, c1) if self.hr else Up(32, 0, 16)
            head_c = c1 if self.hr else 16
        elif cfg.model == "vit_dpt":
            self.encoder = ViTEncoder(cfg.backbone, cfg.size, cfg.vit_taps, cfg.pretrained, cfg.drop_path,
                                      cfg.ckpt_vit)
            self.neck = DPTNeck(self.encoder.dim, 256)
            tap_ch = (256, 256)
            self.up2 = Up(256, c2 if self.hr else 0, c2 if self.hr else 64)
            self.up1 = Up(c2 if self.hr else 64, c1 if self.hr else 0, c1 if self.hr else 32)
            head_c = c1 if self.hr else 32
        else:
            raise ValueError(cfg.model)
        self.heads = Heads(head_c, n_out)
        self.ds_heads = nn.ModuleList([nn.Conv2d(ch, 1, 1) for ch in tap_ch]) if cfg.deep_sup else None

    def _run(self, module, *args):
        if self.cfg.ckpt_hr and self.training:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, x_in: torch.Tensor, x_full: torch.Tensor | None = None) -> dict:
        h_in = x_in.shape[-1]
        s1 = s2 = None
        if self.hr:
            s1, s2 = self._run(self.stem, x_full if (self.full_input and x_full is not None) else x_in)
        if self.cfg.model == "vit_dpt":
            x4, x8 = self.neck(self.encoder(x_in))
            x = x4
            taps = (x4, x8)
        else:
            f = self.encoder(x_in)
            if self.cfg.model == "cnx_unet":
                x = self.ups[0](f[3], f[2])
                x8 = self.ups[1](x, f[1])
                x = self.ups[2](x8, f[0])
            else:
                x = self.ups[0](f[4], f[3])
                x8 = self.ups[1](x, f[2])
                x = self.ups[2](x8, f[1])
            taps = (x, x8)
        if self.cfg.model == "smp_unet":
            x = self.up_enc2(x, f[0])
            if self.hr:
                x = self.up2(x, s2)
                x = self._run(self.up1, x, s1)
            else:
                x = self.up1(x, size=(h_in, h_in))
        elif self.hr:
            x = self.up2(x, s2)
            x = self._run(self.up1, x, s1)
        else:
            x = self.up2(x, size=(h_in // 2, h_in // 2))
            x = self.up1(x, size=(h_in, h_in))
        out = self._run(self.heads, x)
        res = {"p": out[:, :1]}
        if out.shape[1] == 3:
            res["s"], res["d"] = out[:, 1:2], out[:, 2:3]
        if self.training and self.ds_heads is not None:
            res["ds"] = [h(t) for h, t in zip(self.ds_heads, taps)]
        return res

    # ------------------------------------------------------------------ optimiser parameter groups
    def param_groups(self, lr: float, enc_lr: float, llrd: float, wd: float) -> list[dict]:
        """Decoder/stem/heads at ``lr``; encoder at ``enc_lr`` (ViT: layer-wise decay ``llrd`` from the top
        block down to the patch embedding). No weight decay on biases, norms, layer scales and tokens."""
        groups: dict[tuple, dict] = {}

        def add(p, name, base_lr):
            if not p.requires_grad:
                return
            decay = 0.0 if (p.ndim <= 1 or name.endswith(("cls_token", "reg_token", "pos_embed"))) else wd
            key = (base_lr, decay)
            g = groups.setdefault(key, {"params": [], "lr": base_lr, "lr0": base_lr, "weight_decay": decay})
            g["params"].append(p)

        enc = self.encoder.m
        if self.cfg.model == "vit_dpt":
            depth = len(enc.blocks)
            for name, p in enc.named_parameters():
                if name.startswith("blocks."):
                    layer = int(name.split(".")[1]) + 1
                elif name.startswith(("norm", "fc_norm", "head")):
                    layer = depth
                else:   # patch_embed, cls/reg tokens, rope
                    layer = 0
                add(p, name, enc_lr * llrd ** (depth - layer))
        else:
            for name, p in enc.named_parameters():
                add(p, name, enc_lr)
        enc_ids = {id(p) for p in enc.parameters()}
        for name, p in self.named_parameters():
            if id(p) not in enc_ids:
                add(p, name, lr)
        return list(groups.values())


def build_model_v3(cfg: ModelConfig | dict) -> FilaNet:
    if isinstance(cfg, dict):
        cfg = ModelConfig(**{k: v for k, v in cfg.items() if k in ModelConfig.__dataclass_fields__})
    return FilaNet(cfg)


def load_checkpoint_v3(path: str, device="cuda") -> tuple[FilaNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = dict(ck["model_cfg"], pretrained=False)
    model = build_model_v3(mcfg)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    if mcfg["model"] != "vit_dpt":
        model = model.to(memory_format=torch.channels_last)
    return model, ck


def model_config_dict(cfg: ModelConfig) -> dict:
    return asdict(cfg)
