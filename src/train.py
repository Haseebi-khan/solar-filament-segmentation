"""Train a filament segmentation network on one CV fold (or on all images with --fold -1).

Targets are soft: the fraction of annotators that included each pixel in a
filament polygon. With binary cross-entropy the network therefore estimates the
probability that an annotator marks a pixel, which is what the multi-annotator
Panoptic Quality evaluation rewards.

Example:
  python src/train.py --fold 0 --encoder resnet34 --size 1024 --epochs 60 --name r34_1024
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

from filseg.augment import make_batch
from filseg.data import CACHE_DIR, WORK_ROOT
from filseg.infer import predict_probs, resize_uint8
from filseg.metric import evaluate
from filseg.model import build_model
from filseg.postprocess import PPConfig, extract


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fold", type=int, default=0, help="-1 trains on all images")
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--size", type=int, default=1024, help="full-disk resolution seen by the network")
    ap.add_argument("--crop", type=int, default=0, help="training crop size (0 = full disk)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=2.0, help="warm-up epochs")
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--uniform-sampling", action="store_true",
                    help="sample images uniformly instead of in proportion to their annotator count")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def soft_dice_loss(p: torch.Tensor, y: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    inter = (p * y).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def load_targets(size: int) -> np.ndarray:
    path = os.path.join(CACHE_DIR, f"train_soft_{size}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run prepare_data.py --sizes {size}")
    return np.load(path)


@torch.no_grad()
def validate(model, images_full, stems, gt_rles, size) -> dict:
    probs = predict_probs([model], images_full, net_size=size, tta=1, batch_size=8)
    out = {}
    for t in (0.3, 0.4, 0.5):
        preds = {s: extract(p, PPConfig(t_seed=t, t_mask=t, min_area=64))[0] for s, p in zip(stems, probs)}
        out[f"pq@{t}"] = evaluate({s: gt_rles[s] for s in stems}, preds).pq
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False  # MIOpen exhaustive search costs minutes per new shape
    out_dir = os.path.join(WORK_ROOT, "models", args.name)
    os.makedirs(out_dir, exist_ok=True)
    tag = "full" if args.fold < 0 else f"fold{args.fold}"
    log_path = os.path.join(out_dir, f"{tag}.log")
    log = open(log_path, "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(json.dumps(vars(args)))
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stems, folds, n_sets = meta["stems"], meta["folds"], meta["n_sets"]
    tr_idx = np.where(folds != args.fold)[0]
    va_idx = np.where(folds == args.fold)[0] if args.fold >= 0 else np.array([], int)

    full = np.load(os.path.join(CACHE_DIR, "train_images.npy"), mmap_mode="r")
    images = np.stack([resize_uint8(np.asarray(full[i]), args.size) for i in tr_idx])
    soft = load_targets(args.size)[tr_idx]
    dev = torch.device("cuda")
    g_img = torch.from_numpy(images).to(dev)[:, None]
    g_tgt = torch.from_numpy(soft).to(dev)[:, None]
    weights = torch.ones(len(tr_idx)) if args.uniform_sampling else torch.from_numpy(n_sets[tr_idx]).float()
    val_full = np.asarray(full[va_idx]) if len(va_idx) else None
    val_stems = [stems[i] for i in va_idx]
    say(f"train images {len(tr_idx)}  val images {len(va_idx)}  size {args.size}")

    crop = args.crop or args.size
    crop_frac = crop / args.size
    model = build_model(args.arch, args.encoder).to(dev).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # an epoch = one pass worth of images, scaled by the share of the disk a crop covers
    steps_per_epoch = max(1, math.ceil(len(tr_idx) / args.bs / (crop_frac ** 2)))
    total = steps_per_epoch * args.epochs
    warm = int(args.warmup * steps_per_epoch)

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        q = (step - warm) / max(1, total - warm)
        return args.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * q)))

    say(f"steps/epoch {steps_per_epoch}  total steps {total}")
    t0 = time.time()
    step = 0
    run_loss = 0.0
    for epoch in range(args.epochs):
        model.train()
        for _ in range(steps_per_epoch):
            for gr in opt.param_groups:
                gr["lr"] = lr_at(step)
            idx = torch.multinomial(weights, args.bs, replacement=True).to(dev)
            x, y = make_batch(g_img, g_tgt, idx, crop, crop_frac=crop_frac, rotate=not args.no_rotate)
            x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            logits = logits.float()
            loss = F.binary_cross_entropy_with_logits(logits, y)
            if args.dice_weight > 0:
                loss = loss + args.dice_weight * soft_dice_loss(torch.sigmoid(logits), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            with torch.no_grad():
                d = min(args.ema, (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            run_loss = 0.98 * run_loss + 0.02 * loss.item() if step else loss.item()
            step += 1
        msg = f"epoch {epoch + 1}/{args.epochs}  loss {run_loss:.4f}  lr {lr_at(step):.2e}  {time.time() - t0:.0f}s"
        if val_full is not None and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs):
            res = validate(ema, val_full, val_stems, meta["gt_rles"], args.size)
            msg += "  " + "  ".join(f"{k} {v:.4f}" for k, v in res.items())
        say(msg)

    ckpt = {"model": ema.state_dict(), "config": vars(args)}
    torch.save(ckpt, os.path.join(out_dir, f"{tag}.pt"))
    say(f"saved {tag}.pt  total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
