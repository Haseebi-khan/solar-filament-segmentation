"""v3 M6: rater-aware FilaNet. One member on one CV fold, same data, targets, schedule and outputs as train_v3.py,
plus an annotator-conditioned residual head.

  # 30-step benchmark first (time per step, peak memory, projected run time; nothing is saved)
  python src/train_rater_v3.py --config configs/m6_rater_vitb_1536_v3.yaml --fold 0 --bench-steps 30
  # full run (all train_v3.py flags work, e.g. --resume, --epochs, --region)
  python src/train_rater_v3.py --config configs/m6_rater_vitb_1536_v3.yaml --fold 0

Why. The training labels are 37 annotator codes (group + slot, e.g. 0104 + 01) with clearly different styles:
3.4 to 10.5 polygons per set, median filament area 640 to 2,100 px, out-of-fold PQ 0.40 to 0.55 per group. The v3
members fit one map P to the soft consensus, so every annotator's style is noise that P must average out. Here:

  P (consensus)  unchanged v3 head and loss (BCE + JML1 + blob + S/D heads + deep supervision on the soft target)
  R_a (rater a)  logit_P + delta(FiLM_a(F)): F are the decoder features feeding the heads, FiLM_a scales and shifts
                 them with an embedding e_a = e_code + e_group. Trained on the hard mask and the instances of the
                 annotation set sampled for the image (BCE + JML1 + blob loss at 1024 px), i.e. one coherent
                 labelling per sample instead of a pixel average
  unknown rater  with probability --rater-drop a sample uses the "unknown" code and group; codes or groups with
                 fewer than --min-rater-sets training sets of the fold are always unknown. R_unknown learns the
                 population-average annotator, which is what the test set (annotators not given) asks for.

The residual form (delta starts at exactly 0) makes R_a = P at initialisation, so the new head cannot disturb the
proven P recipe early on; the shared features then also have to explain annotator-specific choices (faint
filaments, extent, fragment grouping) instead of only their average. Validation logs PQ of P at 0.4 and 0.6
(v3-loss maps are calibrated lower than v1/v2 maps: their best seed/mask threshold is 0.4) and, per annotation set,
PQ of R_a with the true code versus R_unknown: their difference measures how much of the annotator disagreement is
predictable style rather than noise.

Inputs used: the image only at inference. During training the annotator code comes from the COCO image id
(<annotator>-<stem>), the same field the organisers use to group the metric; spine / bbox / area / category_id
are never read (DECISIONS.md C2).

Outputs (work/models/<name>/): fold<k>.pt (EMA FilaNet weights in the train_v3.py format, so predict_v3.py,
final_ensemble_v2/v3.sh and Stage C use M6 like any other member), fold<k>_rater.pt (EMA rater head + code
vocabulary), fold<k>.log, fold<k>_last.pt (resume checkpoint, deleted at the end).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import sys
import time
from dataclasses import MISSING, asdict, dataclass, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from filseg.augment import normalize, random_affine, warp
from filseg.augment_v3 import BLACK, photometric_v3, to_inputs
from filseg.data import CACHE_DIR, WORK_ROOT, load_train_annotations
from filseg.infer import resize_uint8
from filseg.losses_v3 import area_down, blob_loss, jml1, supervised_loss
from filseg.metric import PQStats, pq_stats
from filseg.models_v3 import build_model_v3
from filseg.postprocess import PPConfig, extract
from train_v3 import TrainConfig


@dataclass
class RaterConfig(TrainConfig):
    # ---- rater head (everything else as train_v3.TrainConfig)
    rater_dim: int = 32              # code / group embedding size
    w_rater: float = 1.0             # BCE + JML1 of R_a against the sampled set's hard mask (1024 px)
    w_rater_blob: float = 0.5        # blob loss of R_a on the same set's instances
    rater_drop: float = 0.25         # probability of training a sample with the unknown code and group
    min_rater_sets: int = 5          # codes / groups with fewer training sets in this fold are unknown
    group_balance: float = 0.0       # 0 = v1 sampling (proportional to n_sets); a > 0 also multiplies an image's
                                     # weight by mean over its sets of share(group)^-a (rare groups seen more)


def parse_config(argv=None) -> RaterConfig:
    """train_v3.parse_config for RaterConfig: YAML defaults, every field overridable on the command line."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    defaults = {}
    if known.config:
        with open(known.config) as f:
            defaults = yaml.safe_load(f) or {}
        unknown = set(defaults) - {f.name for f in fields(RaterConfig)}
        if unknown:
            raise SystemExit(f"unknown keys in {known.config}: {sorted(unknown)}")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML file with RaterConfig fields (CLI overrides it)")
    for f in fields(RaterConfig):
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
    cfg = RaterConfig(**args)
    if not cfg.name:
        ap.error("--name is required (or 'name:' in the config)")
    if cfg.unlabeled:
        ap.error("train_rater_v3.py has no Noisy-Student stream; use train_v3.py for Stage B")
    return cfg


# ----------------------------------------------------------------------------------------------- rater head
class RaterHead(nn.Module):
    """R_a = logit_P + delta(FiLM_a(F)). Index n_codes / n_groups is the unknown code / group."""

    def __init__(self, c: int, n_codes: int, n_groups: int, dim: int = 32):
        super().__init__()
        self.n_codes, self.n_groups = n_codes, n_groups
        self.code = nn.Embedding(n_codes + 1, dim)
        self.group = nn.Embedding(n_groups + 1, dim)
        self.film = nn.Linear(dim, 2 * c)
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, c, 3, padding=1),
                                  nn.GELU(), nn.Conv2d(c, 1, 1))
        with torch.no_grad():   # identity FiLM and delta = 0 at the start: R_a = P
            self.film.weight.zero_()
            self.film.bias.zero_()
            self.body[-1].weight.zero_()
            self.body[-1].bias.zero_()

    def forward(self, feat: torch.Tensor, logit_p: torch.Tensor, code: torch.Tensor, group: torch.Tensor
                ) -> torch.Tensor:
        gamma, beta = self.film(self.code(code) + self.group(group)).chunk(2, 1)
        h = feat * (1 + gamma[:, :, None, None].to(feat.dtype)) + beta[:, :, None, None].to(feat.dtype)
        return logit_p + self.body(h)


def capture_head_input(net: nn.Module) -> dict:
    """Keeps the decoder features that enter ``net.heads`` (the input of the P/S/D heads) in store["x"]."""
    store = {}
    net.heads.register_forward_pre_hook(lambda module, args: store.__setitem__("x", args[0]))
    return store


def rater_loss(r_logit: torch.Tensor, inst: torch.Tensor, n_max: int, cfg: RaterConfig) -> dict:
    """r_logit (B, 1, S, S) float32; inst (B, L, L) int64 instances of the sampled set (0 = background)."""
    p = area_down(torch.sigmoid(r_logit), inst.shape[-1])
    h = (inst > 0).float()[:, None]
    parts = {"r_reg": cfg.w_rater * (F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), h) + jml1(p, h))}
    if cfg.w_rater_blob:
        parts["r_blob"] = cfg.w_rater_blob * blob_loss(p, inst, n_max)
    return parts


# ----------------------------------------------------------------------------------------------- data
def rater_vocab(stems: list[str]) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """All annotator codes and groups (fixed order, so every fold shares one vocabulary) and, per stem, the codes
    of its annotation sets in slot order (the order of train_inst_*.npy and meta['gt_rles'])."""
    _, by_stem = load_train_annotations()
    codes_of = {s: [a.set_id.split("-", 1)[0] for a in by_stem[s]] for s in stems}
    codes = sorted({c for v in codes_of.values() for c in v})
    groups = sorted({c[:4] for c in codes})
    return codes, groups, codes_of


def slot_indices(idx: np.ndarray, stems, codes_of, code_ix, group_ix, n_slots: int, unk_c: int, unk_g: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    code = np.full((len(idx), n_slots), unk_c, np.int64)
    group = np.full((len(idx), n_slots), unk_g, np.int64)
    for r, i in enumerate(idx):
        for k, c in enumerate(codes_of[stems[i]][:n_slots]):
            code[r, k] = code_ix.get(c, unk_c)
            group[r, k] = group_ix.get(c[:4], unk_g)
    return code, group


def make_batch_rater(img, idx, in_size: int, out_size: int, need_full: bool, tgt, inst, n_sets, inst_size: int,
                     rotate: bool = True) -> dict:
    """augment_v3.make_batch_v3, but also returns the sampled annotation-set slot k of every image."""
    b = len(idx)
    theta = random_affine(b, img.device, rotate=rotate)
    x = normalize(img[idx])
    work = img.shape[-1] if need_full else in_size
    x = warp(x - BLACK, theta, work) + BLACK
    x = photometric_v3(x)
    k = (torch.rand(b, device=img.device) * n_sets[idx].float()).long().clamp(max=inst.shape[1] - 1)
    lab = inst[idx, k][:, None].float()
    return {"x_full": x if need_full else None, "x_in": to_inputs(x, in_size),
            "tgt": warp(tgt[idx].float() / 255.0, theta, out_size),
            "inst": warp(lab, theta, inst_size, mode="nearest")[:, 0].round().long(), "k": k}


# ----------------------------------------------------------------------------------------------- validation
PP_P = {"pq@0.4l8": PPConfig(t_seed=0.4, t_mask=0.4, min_area=100, link=8, close=-1),
        "pq@0.6l8": PPConfig(t_seed=0.6, t_mask=0.6, min_area=100, link=8, close=-1)}
PP_R = PPConfig(t_seed=0.5, t_mask=0.5, min_area=100, link=8, close=-1)   # R_a is fit to hard masks


@torch.no_grad()
def validate(net, rater, store, images, stems, codes, groups, gt_rles) -> dict:
    """No TTA. PQ of P, and per annotation set PQ of R_a (true code) and R_unknown against that set alone."""
    stats = {k: PQStats() for k in (*PP_P, "rater_true@0.5l8", "rater_unk@0.5l8")}
    channels_last = net.cfg.model != "vit_dpt"
    unk = torch.tensor([rater.n_codes], device="cuda"), torch.tensor([rater.n_groups], device="cuda")
    for img, stem, code, group in zip(images, stems, codes, groups):
        x_full = normalize(torch.from_numpy(np.asarray(img)).cuda()[None, None])
        x_in = to_inputs(x_full, net.size)
        xf = x_full if net.full_input else None
        if channels_last:
            x_in = x_in.contiguous(memory_format=torch.channels_last)
            xf = xf.contiguous(memory_format=torch.channels_last) if xf is not None else None
        n = len(gt_rles[stem])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x_in, xf)
            feat, lp = store["x"], out["p"]
            r_true = rater(feat.repeat(n, 1, 1, 1), lp.repeat(n, 1, 1, 1),
                           torch.from_numpy(code[:n]).cuda(), torch.from_numpy(group[:n]).cuda())
            r_unk = rater(feat, lp, *unk)
        p = torch.sigmoid(lp.float())[0, 0].cpu().numpy()
        preds = {k: extract(p, c)[0] for k, c in PP_P.items()}
        pred_unk = extract(torch.sigmoid(r_unk.float())[0, 0].cpu().numpy(), PP_R)[0]
        r_true = torch.sigmoid(r_true.float())[:, 0].cpu().numpy()
        for k, g in enumerate(gt_rles[stem]):
            for name, pr in preds.items():
                stats[name].add(pq_stats(g, pr))
            stats["rater_true@0.5l8"].add(pq_stats(g, extract(r_true[k], PP_R)[0]))
            stats["rater_unk@0.5l8"].add(pq_stats(g, pred_unk))
    return {k: v.pq for k, v in stats.items()}


# ----------------------------------------------------------------------------------------------- training
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

    # ---- annotator vocabulary; rare codes / groups of this fold's training images -> unknown
    codes, groups, codes_of = rater_vocab(stems)
    n_code_tr = {c: 0 for c in codes}
    for i in tr_idx:
        for c in codes_of[stems[i]]:
            n_code_tr[c] += 1
    n_group_tr = {g: sum(n for c, n in n_code_tr.items() if c[:4] == g) for g in groups}
    code_ix = {c: j for j, c in enumerate(codes) if n_code_tr[c] >= cfg.min_rater_sets}
    group_ix = {g: j for j, g in enumerate(groups) if n_group_tr[g] >= cfg.min_rater_sets}
    unk_c, unk_g = len(codes), len(groups)
    say(f"annotators: {len(codes)} codes ({len(code_ix)} with >= {cfg.min_rater_sets} training sets), "
        f"{len(groups)} groups ({len(group_ix)}); unknown-rater dropout {cfg.rater_drop}")

    # ---- labelled data on the GPU (uint8 / int16), as train_v3.py
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
    with open(os.path.join(CACHE_DIR, "targets_v3.pkl"), "rb") as f:   # instance maps: always needed (R_a)
        tinfo = pickle.load(f)
    n_max = int(tinfo["max_instances_per_set"])
    inst_size = int(tinfo["inst_size"])
    lw.blob_size = inst_size
    inst = np.load(os.path.join(CACHE_DIR, f"train_inst_{inst_size}.npy"), mmap_mode="r")
    g_inst = torch.from_numpy(np.ascontiguousarray(inst[tr_idx])).to(dev)
    g_nsets = torch.from_numpy(n_sets[tr_idx]).to(dev)
    n_slots = g_inst.shape[1]
    tr_code, tr_group = slot_indices(tr_idx, stems, codes_of, code_ix, group_ix, n_slots, unk_c, unk_g)
    g_code, g_group = torch.from_numpy(tr_code).to(dev), torch.from_numpy(tr_group).to(dev)
    va_code, va_group = slot_indices(va_idx, stems, codes_of, code_ix, group_ix, n_slots, unk_c, unk_g)

    weights = torch.ones(len(tr_idx)) if cfg.uniform_sampling else torch.from_numpy(n_sets[tr_idx]).float()
    if cfg.group_balance > 0:
        total = sum(n_group_tr.values())
        share = {g: max(n, 1) / total for g, n in n_group_tr.items()}
        factor = [np.mean([share[c[:4]] ** -cfg.group_balance for c in codes_of[stems[i]]]) for i in tr_idx]
        weights = weights * torch.tensor(factor, dtype=torch.float32)
    val_imgs = np.asarray(full[va_idx]) if len(va_idx) else None
    say(f"train images {len(tr_idx)}  val images {len(va_idx)}  input {mcfg.size}  output {out_size}  "
        f"targets {kinds}  instance maps {inst_size} px  data on GPU in {time.time() - t0:.0f}s")

    # ---- model, rater head, EMA, optimiser
    net = build_model_v3(mcfg).to(dev)
    channels_last = mcfg.model != "vit_dpt"
    if channels_last:
        net = net.to(memory_format=torch.channels_last)
    if cfg.init:
        ck = torch.load(cfg.init, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        say(f"initialised from {cfg.init}")
    head_c = net.heads.conv[0].in_channels
    rater = RaterHead(head_c, len(codes), len(groups), cfg.rater_dim).to(dev)
    if channels_last:
        rater = rater.to(memory_format=torch.channels_last)
    ema_net, ema_rater = copy.deepcopy(net).eval(), copy.deepcopy(rater).eval()
    for p in list(ema_net.parameters()) + list(ema_rater.parameters()):
        p.requires_grad_(False)
    store, ema_store = capture_head_input(net), capture_head_input(ema_net)   # after deepcopy: one store each
    groups_opt = net.param_groups(cfg.lr, cfg.enc_lr or cfg.lr, cfg.llrd, cfg.wd)
    emb = [rater.code.weight, rater.group.weight]
    groups_opt.append({"params": emb, "lr": cfg.lr, "lr0": cfg.lr, "weight_decay": 0.0})
    groups_opt.append({"params": [p for p in rater.parameters() if all(p is not e for e in emb)],
                       "lr": cfg.lr, "lr0": cfg.lr, "weight_decay": cfg.wd})
    opt = torch.optim.AdamW(groups_opt, lr=cfg.lr, weight_decay=cfg.wd)
    train_params = list(net.parameters()) + list(rater.parameters())
    ema_params = list(ema_net.parameters()) + list(ema_rater.parameters())
    say(f"model {mcfg.model}/{mcfg.backbone}: {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parameters "
        f"+ rater head {sum(p.numel() for p in rater.parameters()) / 1e3:.1f} k (features {head_c} ch)")

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
        net.load_state_dict(ck["model"])
        ema_net.load_state_dict(ck["ema"])
        rater.load_state_dict(ck["rater"])
        ema_rater.load_state_dict(ck["ema_rater"])
        opt.load_state_dict(ck["opt"])
        start_epoch, opt_step = ck["epoch"], ck["opt_step"]
        say(f"resumed from {last_path} at epoch {start_epoch}")
    say(f"micro-batches/epoch {micro_per_epoch} ({cfg.bs} images)  accum {cfg.accum}  optimiser steps {total}")

    def cl(x):
        return x.contiguous(memory_format=torch.channels_last) if (x is not None and channels_last) else x

    t0 = time.time()
    run = {}
    micro = 0
    torch.cuda.reset_peak_memory_stats()
    t_bench = None
    for epoch in range(start_epoch, cfg.epochs):
        net.train()
        rater.train()
        for _ in range(micro_per_epoch):
            if micro % cfg.accum == 0:
                f_lr = lr_factor(opt_step)
                for g in opt.param_groups:
                    g["lr"] = g["lr0"] * f_lr
            idx = torch.multinomial(weights, cfg.bs, replacement=True).to(dev)
            b = make_batch_rater(g_img, idx, mcfg.size, out_size, need_full, g_tgt, g_inst, g_nsets, inst_size,
                                 rotate=not cfg.no_rotate)
            code, group = g_code[idx, b["k"]], g_group[idx, b["k"]]
            drop = torch.rand(len(idx), device=dev) < cfg.rater_drop
            code = torch.where(drop, torch.full_like(code, unk_c), code)
            group = torch.where(drop, torch.full_like(group, unk_g), group)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(cl(b["x_in"]), cl(b["x_full"]))
                r_logit = rater(store["x"], out["p"], code, group)
            out = {k: ([t.float() for t in v] if k == "ds" else v.float()) for k, v in out.items()}
            tg = b["tgt"]
            y = tg[:, :1]
            t = tg[:, 1:2] if tg.shape[1] > 1 else None
            d = tg[:, 2:3] if tg.shape[1] > 2 else None
            loss, parts = supervised_loss(out, y, t, d, b["inst"] if lw.w_blob else None, n_max, lw)
            r_parts = rater_loss(r_logit.float(), b["inst"], n_max, cfg)
            loss = loss + sum(r_parts.values())
            parts.update({k: float(v.detach()) for k, v in r_parts.items()})
            (loss / cfg.accum).backward()
            micro += 1
            if micro % cfg.accum == 0:
                torch.nn.utils.clip_grad_norm_(train_params, cfg.clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    dk = min(cfg.ema, (1 + opt_step) / (10 + opt_step))
                    for pe, pm in zip(ema_params, train_params):
                        pe.lerp_(pm, 1 - dk)
                    for be, bm in zip(ema_net.buffers(), net.buffers()):
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
                    return
        msg = (f"epoch {epoch + 1}/{cfg.epochs}  " + "  ".join(f"{k} {v:.4f}" for k, v in run.items())
               + f"  lr {opt.param_groups[0]['lr']:.2e}  {time.time() - t0:.0f}s")
        if val_imgs is not None and ((epoch + 1) % cfg.eval_every == 0 or epoch + 1 == cfg.epochs):
            res = validate(ema_net, ema_rater, ema_store, val_imgs, val_stems, va_code, va_group, meta["gt_rles"])
            msg += "  " + "  ".join(f"{k} {v:.4f}" for k, v in res.items())
        say(msg)
        if cfg.save_every and (epoch + 1) % cfg.save_every == 0 and epoch + 1 < cfg.epochs:
            torch.save({"model": net.state_dict(), "ema": ema_net.state_dict(), "rater": rater.state_dict(),
                        "ema_rater": ema_rater.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1,
                        "opt_step": opt_step, "config": asdict(cfg)}, last_path)

    torch.save({"model": ema_net.state_dict(), "model_cfg": asdict(mcfg), "config": asdict(cfg)},
               os.path.join(out_dir, f"{tag}.pt"))
    torch.save({"rater": ema_rater.state_dict(), "codes": codes, "groups": groups, "code_ix": code_ix,
                "group_ix": group_ix, "head_c": head_c, "rater_dim": cfg.rater_dim},
               os.path.join(out_dir, f"{tag}_rater.pt"))
    if os.path.exists(last_path):
        os.remove(last_path)
    say(f"saved {tag}.pt and {tag}_rater.pt  total time {time.time() - t0:.0f}s  peak memory "
        f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")


if __name__ == "__main__":
    main(sys.argv[1:])
