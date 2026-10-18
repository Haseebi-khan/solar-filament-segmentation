"""v3: train one FilaNet member on one CV fold (or on all images with --fold -1).

  # 50-step benchmark first (time per step, peak memory, projected run time; nothing is saved)
  python src/train_v3.py --config configs/m1_cnxl_2048_v3.yaml --fold 0 --bench-steps 50
  # full run
  python src/train_v3.py --config configs/m1_cnxl_2048_v3.yaml --fold 0
  # any config field can be overridden on the command line, e.g. a loss ablation on the fast proxy
  python src/train_v3.py --config configs/proxy_r34_1024_v3.yaml --name abl_dml --region dml --fold 0

Targets (see filseg/targets_v3.py, built by prepare_targets_v3.py) live on the GPU as uint8, like v1:
y (soft mask), T (soft skeleton), D (soft core) at the output resolution and the per-set instance maps at
1024 px for the blob loss. Losses: filseg/losses_v3.py. Images are sampled in proportion to their number of
annotation sets (v1 rule). Weights are averaged with an EMA and the EMA weights are saved.

Noisy-Student (Stage B): with --unlabeled <dir> --pseudo <file pattern with {fold}> half of every batch
(--u-frac) comes from the unlabelled GONG frames with their soft pseudo-labels (no thresholding), strong
photometric noise on both halves with --student-aug. Frames within --u-exclude-hours of a validation image of
this fold are dropped, so the out-of-fold numbers stay honest even with a 2010-2022 pool.

Writes $FILSEG_WORK/models/<name>/fold<k>.pt (EMA weights + configs), fold<k>.log, and fold<k>_last.pt
(model, EMA, optimiser; for --resume, deleted at the end).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import queue
import sys
import threading
import time
from dataclasses import MISSING, asdict, dataclass, field, fields
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from filseg.augment_v3 import make_batch_v3, warp_unlabeled
from filseg.data import CACHE_DIR, WORK_ROOT
from filseg.infer import resize_uint8
from filseg.infer_v3 import predict_maps
from filseg.losses_v3 import LossWeights, supervised_loss, unlabeled_loss
from filseg.metric import evaluate
from filseg.models_v3 import ModelConfig, build_model_v3
from filseg.postprocess import PPConfig, extract


@dataclass
class TrainConfig:
    name: str = ""
    fold: int = 0
    seed: int = 0
    # ---- model (filseg.models_v3.ModelConfig)
    model: str = "smp_unet"
    backbone: str = "resnet34"
    size: int = 1536
    hr_stem: bool = False
    stem_native: bool = True
    stem_ch: list = field(default_factory=lambda: [32, 64])
    edge_inputs: bool = False
    aux_heads: bool = True
    deep_sup: bool = True
    drop_path: float = 0.0
    pretrained: bool = True
    vit_taps: list = field(default_factory=lambda: [5, 11, 17, 23])
    ckpt_stages: list = field(default_factory=list)
    ckpt_vit: bool = True
    ckpt_hr: bool = False
    # ---- optimisation
    epochs: int = 80
    bs: int = 8                      # images per micro-batch (labelled + unlabelled)
    accum: int = 1                   # gradient accumulation (effective batch = bs * accum)
    lr: float = 5e-4                 # decoder, stem and heads
    enc_lr: float = 0.0              # encoder (0 = same as lr); for ViTs the top block's LR
    llrd: float = 1.0                # ViT layer-wise LR decay
    wd: float = 1e-4
    warmup: float = 2.0              # warm-up epochs, then cosine to 1 %
    ema: float = 0.998
    clip: float = 2.0
    # ---- losses (filseg.losses_v3.LossWeights)
    region: str = "jml"
    w_region: float = 1.0
    w_blob: float = 0.5
    w_s: float = 0.5
    w_d: float = 0.25
    w_ds: float = 1.0
    w_sr: float = 0.0
    # ---- data
    uniform_sampling: bool = False
    no_rotate: bool = False
    # ---- Stage B (Noisy Student)
    unlabeled: str = ""              # directory with U_images.npy and U_meta.csv (gong_qc_v3.py); $VARS expanded
    pseudo: str = ""                 # pseudo-label file pattern, e.g. $FILSEG_UNLABELED/pl/m1+m2/fold{fold}.npy
    lambda_u: float = 1.0
    u_frac: float = 0.5
    u_exclude_hours: float = 72.0
    student_aug: bool = False
    init: str = ""                   # warm start: checkpoint whose EMA weights initialise the model
    # ---- run control
    eval_every: int = 10
    bench_steps: int = 0             # > 0: benchmark this many micro-steps, print timings, exit
    save_every: int = 5              # epochs between resume checkpoints
    resume: bool = False

    def model_config(self) -> ModelConfig:
        return ModelConfig(**{f.name: getattr(self, f.name) for f in fields(ModelConfig)})

    def loss_weights(self) -> LossWeights:
        return LossWeights(**{f.name: getattr(self, f.name) for f in fields(LossWeights) if hasattr(self, f.name)})


def parse_config(argv=None) -> TrainConfig:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    defaults = {}
    if known.config:
        with open(known.config) as f:
            defaults = yaml.safe_load(f) or {}
        unknown = set(defaults) - {f.name for f in fields(TrainConfig)}
        if unknown:
            raise SystemExit(f"unknown keys in {known.config}: {sorted(unknown)}")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML file with TrainConfig fields (CLI overrides it)")
    for f in fields(TrainConfig):
        fallback = f.default_factory() if f.default_factory is not MISSING else f.default
        default = defaults.get(f.name, fallback)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(default, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(default, list):
            ap.add_argument(flag, type=int, nargs="*", default=default)
        else:
            ap.add_argument(flag, type=type(default), default=default)
    args = vars(ap.parse_args(argv))
    args.pop("config")
    cfg = TrainConfig(**args)
    if not cfg.name:
        ap.error("--name is required (or 'name:' in the config)")
    return cfg


# ----------------------------------------------------------------------------------------------- data
def _stamp(stem: str) -> datetime:
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S")


class UnlabeledStream:
    """Background thread that reads random unlabelled frames and their pseudo-labels from disk."""

    def __init__(self, images: np.ndarray, pseudo: np.ndarray, weights: np.ndarray, keep: np.ndarray,
                 batch: int, seed: int, depth: int = 4):
        self.images, self.pseudo, self.weights = images, pseudo, weights
        self.keep = keep
        self.batch = batch
        self.rng = np.random.default_rng(seed + 1000)
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self.stop:
            idx = np.sort(self.rng.choice(self.keep, self.batch, replace=False))
            img = torch.from_numpy(np.stack([np.asarray(self.images[i]) for i in idx]))[:, None].pin_memory()
            pl = torch.from_numpy(np.stack([np.asarray(self.pseudo[i]) for i in idx])).pin_memory()
            w = torch.from_numpy(self.weights[idx].astype(np.float32))
            self.q.put((img, pl, w))

    def next(self, dev):
        img, pl, w = self.q.get()
        return img.to(dev, non_blocking=True), pl.to(dev, non_blocking=True), w.to(dev)


def expand(path: str) -> str:
    """Expands $VARS in config paths; $FILSEG_UNLABELED defaults to $FILSEG_WORK/unlabeled."""
    os.environ.setdefault("FILSEG_UNLABELED", os.path.join(WORK_ROOT, "unlabeled"))
    return os.path.expandvars(os.path.expanduser(path))


def load_unlabeled(cfg: TrainConfig, val_stems: list[str], say):
    import pandas as pd
    u_dir = expand(cfg.unlabeled)
    images = np.load(os.path.join(u_dir, "U_images.npy"), mmap_mode="r")
    meta = pd.read_csv(os.path.join(u_dir, "U_meta.csv"))
    path = expand(cfg.pseudo).format(fold=cfg.fold)
    pseudo = np.load(path, mmap_mode="r")
    wpath = os.path.splitext(path)[0] + "_w.npy"
    weights = np.load(wpath) if os.path.exists(wpath) else np.ones(len(images), np.float32)
    assert len(images) == len(meta) == len(pseudo) == len(weights), "U images, meta, pseudo-labels disagree"
    keep = np.ones(len(images), bool)
    if val_stems and cfg.u_exclude_hours > 0:
        val_t = np.array(sorted(_stamp(s).timestamp() for s in val_stems))
        u_t = np.array([_stamp(s).timestamp() for s in meta["stem"]])
        pos = np.searchsorted(val_t, u_t)
        gap = np.full(len(u_t), np.inf)
        for off in (-1, 0):
            j = np.clip(pos + off, 0, len(val_t) - 1)
            gap = np.minimum(gap, np.abs(u_t - val_t[j]))
        keep &= gap > cfg.u_exclude_hours * 3600
    say(f"unlabelled frames {len(images)}, kept {int(keep.sum())} (> {cfg.u_exclude_hours:g} h from every "
        f"validation image of fold {cfg.fold}); pseudo-labels {path} {tuple(pseudo.shape)}")
    return images, pseudo, weights, np.where(keep)[0]


# ----------------------------------------------------------------------------------------------- validation
@torch.no_grad()
def validate(model, images, stems, gt_rles) -> dict:
    probs = predict_maps([model], images, tta=1, batch_size=2, heads=("p",))["p"]
    out = {}
    for name, cfg in [("pq@0.5", PPConfig(t_seed=0.5, t_mask=0.5, min_area=64)),
                      ("pq@0.6l8", PPConfig(t_seed=0.6, t_mask=0.6, min_area=100, link=8, close=-1))]:
        preds = {s: extract(p, cfg)[0] for s, p in zip(stems, probs)}
        out[name] = evaluate({s: gt_rles[s] for s in stems}, preds).pq
    return out


def main(argv=None):
    cfg = parse_config(argv)
    mcfg = cfg.model_config()
    lw = cfg.loss_weights()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = False  # MIOpen exhaustive search costs minutes per new shape
    dev = torch.device("cuda")
    out_dir = os.path.join(WORK_ROOT, "models", cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    tag = "full" if cfg.fold < 0 else f"fold{cfg.fold}"
    bench = cfg.bench_steps > 0
    if bench:
        cfg.bench_steps = max(cfg.bench_steps, 10)  # the first 5 micro-steps (kernel compilation) are not timed
    log = open(os.devnull if bench else os.path.join(out_dir, f"{tag}.log"), "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(json.dumps(asdict(cfg)))
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stems, folds, n_sets = meta["stems"], meta["folds"], meta["n_sets"]
    tr_idx = np.where(folds != cfg.fold)[0]
    va_idx = np.where(folds == cfg.fold)[0] if cfg.fold >= 0 else np.array([], int)
    val_stems = [stems[i] for i in va_idx]

    # ---- labelled data on the GPU (uint8 / int16)
    need_full = (mcfg.hr_stem and mcfg.stem_native) or mcfg.size == 2048
    out_size = mcfg.out_size
    full = np.load(os.path.join(CACHE_DIR, "train_images.npy"), mmap_mode="r")
    t0 = time.time()
    if need_full:
        g_img = torch.from_numpy(np.asarray(full[tr_idx])).to(dev)[:, None]
    else:
        g_img = torch.from_numpy(np.stack([resize_uint8(np.asarray(full[i]), mcfg.size) for i in tr_idx])
                                 ).to(dev)[:, None]
    kinds = ["soft"] + (["skel"] if (mcfg.aux_heads or lw.w_sr) else []) + (["core"] if mcfg.aux_heads else [])
    g_tgt = torch.empty((len(tr_idx), len(kinds), out_size, out_size), dtype=torch.uint8, device=dev)
    for c, kind in enumerate(kinds):
        path = os.path.join(CACHE_DIR, f"train_{kind}_{out_size}.npy")
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing: run src/prepare_targets_v3.py --sizes {out_size}")
        g_tgt[:, c] = torch.from_numpy(np.ascontiguousarray(np.load(path, mmap_mode="r")[tr_idx])).to(dev)
    g_inst = g_nsets = None
    n_max = 1
    if lw.w_blob > 0:
        with open(os.path.join(CACHE_DIR, "targets_v3.pkl"), "rb") as f:
            tinfo = pickle.load(f)
        n_max = int(tinfo["max_instances_per_set"])
        inst = np.load(os.path.join(CACHE_DIR, f"train_inst_{tinfo['inst_size']}.npy"), mmap_mode="r")
        g_inst = torch.from_numpy(np.ascontiguousarray(inst[tr_idx])).to(dev)
        g_nsets = torch.from_numpy(n_sets[tr_idx]).to(dev)
        lw.blob_size = int(tinfo["inst_size"])
    weights = torch.ones(len(tr_idx)) if cfg.uniform_sampling else torch.from_numpy(n_sets[tr_idx]).float()
    val_imgs = np.asarray(full[va_idx]) if len(va_idx) else None
    say(f"train images {len(tr_idx)}  val images {len(va_idx)}  input {mcfg.size}  output {out_size}  "
        f"targets {kinds}  blob {'on' if g_inst is not None else 'off'}  data on GPU in {time.time() - t0:.0f}s")

    # ---- unlabelled stream
    stream = None
    bs_u = 0
    if cfg.unlabeled:
        bs_u = max(1, int(round(cfg.bs * cfg.u_frac)))
        u_img, u_pl, u_w, u_keep = load_unlabeled(cfg, val_stems, say)
        stream = UnlabeledStream(u_img, u_pl, u_w, u_keep, bs_u, cfg.seed)
    bs_l = cfg.bs - bs_u
    assert bs_l >= 1

    # ---- model, EMA, optimiser
    model = build_model_v3(mcfg).to(dev)
    channels_last = mcfg.model != "vit_dpt"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if cfg.init:
        ck = torch.load(cfg.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        say(f"initialised from {cfg.init}")
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    groups = model.param_groups(cfg.lr, cfg.enc_lr or cfg.lr, cfg.llrd, cfg.wd)
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.wd)
    n_params = sum(p.numel() for p in model.parameters())
    say(f"model {mcfg.model}/{mcfg.backbone}: {n_params / 1e6:.1f} M parameters, {len(groups)} parameter groups")

    # an epoch = len(tr_idx) images through the network; a student (half unlabelled) therefore sees half as
    # many labelled images per epoch as its teacher, at the same compute
    micro_per_epoch = max(1, math.ceil(len(tr_idx) / cfg.bs))
    opt_per_epoch = max(1, micro_per_epoch // cfg.accum)
    total = opt_per_epoch * cfg.epochs
    warm = max(1, int(cfg.warmup * opt_per_epoch))

    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        q = (step - warm) / max(1, total - warm)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, q)))

    start_epoch, opt_step = 0, 0
    last_path = os.path.join(out_dir, f"{tag}_last.pt")
    if cfg.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start_epoch, opt_step = ck["epoch"], ck["opt_step"]
        say(f"resumed from {last_path} at epoch {start_epoch}")
    say(f"micro-batches/epoch {micro_per_epoch} ({bs_l} labelled + {bs_u} unlabelled)  accum {cfg.accum}  "
        f"optimiser steps {total}")

    def cl(x):
        return x.contiguous(memory_format=torch.channels_last) if (x is not None and channels_last) else x

    t0 = time.time()
    run = {}
    micro = 0
    torch.cuda.reset_peak_memory_stats()
    t_bench = None
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        for _ in range(micro_per_epoch):
            if micro % cfg.accum == 0:
                f_lr = lr_factor(opt_step)
                for g in opt.param_groups:
                    g["lr"] = g["lr0"] * f_lr
            idx = torch.multinomial(weights, bs_l, replacement=True).to(dev)
            b = make_batch_v3(g_img, idx, mcfg.size, out_size, need_full, g_tgt, g_inst, g_nsets,
                              inst_size=lw.blob_size, rotate=not cfg.no_rotate, strong=cfg.student_aug)
            x_in, x_full = b["x_in"], b["x_full"]
            if stream is not None:
                u_img, u_pl, u_wt = stream.next(dev)
                if not need_full and u_img.shape[-1] != mcfg.size:
                    u_img = F.interpolate(u_img.float(), size=(mcfg.size, mcfg.size), mode="area")
                if u_pl.shape[-1] != out_size:
                    u_pl = F.interpolate(u_pl.float(), size=(out_size, out_size), mode="bilinear",
                                         align_corners=False)
                bu = warp_unlabeled(u_img, u_pl, mcfg.size, out_size, need_full, rotate=not cfg.no_rotate,
                                    strong=True)
                x_in = torch.cat([x_in, bu["x_in"]])
                x_full = torch.cat([x_full, bu["x_full"]]) if x_full is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(cl(x_in), cl(x_full))
            out = {k: ([t.float() for t in v] if k == "ds" else v.float()) for k, v in out.items()}
            lab = {k: ([t[:bs_l] for t in v] if k == "ds" else v[:bs_l]) for k, v in out.items()}
            tg = b["tgt"]
            y = tg[:, :1]
            t = tg[:, 1:2] if tg.shape[1] > 1 else None
            d = tg[:, 2:3] if tg.shape[1] > 2 else None
            loss, parts = supervised_loss(lab, y, t, d, b.get("inst"), n_max, lw)
            if stream is not None:
                unl = {k: v[bs_l:] for k, v in out.items() if k != "ds"}
                pl = bu["tgt"]
                lu = unlabeled_loss(unl, pl[:, :1], pl[:, 1:2] if pl.shape[1] > 1 else None, u_wt)
                loss = loss + cfg.lambda_u * lu
                parts["unlab"] = float(lu.detach())
            (loss / cfg.accum).backward()
            micro += 1
            if micro % cfg.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    dk = min(cfg.ema, (1 + opt_step) / (10 + opt_step))
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1 - dk)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
                opt_step += 1
            for k, v in dict(parts, total=float(loss.detach())).items():
                run[k] = 0.98 * run[k] + 0.02 * v if k in run else v
            if bench:
                if micro == 5:
                    torch.cuda.synchronize()
                    t_bench = time.time()
                    say(f"bench: first 5 micro-steps (incl. kernel compilation) {t_bench - t0:.0f}s")
                if micro == cfg.bench_steps:
                    torch.cuda.synchronize()
                    dt = (time.time() - t_bench) / max(1, cfg.bench_steps - 5)
                    peak = torch.cuda.max_memory_allocated() / 2 ** 30
                    ep = dt * micro_per_epoch
                    say(f"bench: {dt:.3f} s/micro-step ({cfg.bs} images), peak memory {peak:.1f} GiB; "
                        f"~{ep / 60:.1f} min/epoch, ~{ep * cfg.epochs / 3600:.2f} h for {cfg.epochs} epochs "
                        f"(+ validation); loss {run['total']:.4f}")
                    if stream is not None:
                        stream.stop = True
                    return
        msg = (f"epoch {epoch + 1}/{cfg.epochs}  " + "  ".join(f"{k} {v:.4f}" for k, v in run.items())
               + f"  lr {opt.param_groups[0]['lr']:.2e}  {time.time() - t0:.0f}s")
        if val_imgs is not None and ((epoch + 1) % cfg.eval_every == 0 or epoch + 1 == cfg.epochs):
            res = validate(ema, val_imgs, val_stems, meta["gt_rles"])
            msg += "  " + "  ".join(f"{k} {v:.4f}" for k, v in res.items())
        say(msg)
        if cfg.save_every and (epoch + 1) % cfg.save_every == 0 and epoch + 1 < cfg.epochs:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "epoch": epoch + 1, "opt_step": opt_step, "config": asdict(cfg)}, last_path)

    if stream is not None:
        stream.stop = True
    torch.save({"model": ema.state_dict(), "model_cfg": asdict(mcfg), "config": asdict(cfg)},
               os.path.join(out_dir, f"{tag}.pt"))
    if os.path.exists(last_path):
        os.remove(last_path)
    say(f"saved {tag}.pt  total time {time.time() - t0:.0f}s  peak memory "
        f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")


if __name__ == "__main__":
    main(sys.argv[1:])
