"""v3 Stage C1: build the multi-hypothesis candidate pool for the training (out-of-fold) or test images.

  python src/build_pool_v3.py --names r34_2048 effv2s_1536 --weights 1 1 --split oof
  python src/build_pool_v3.py --names r34_2048 effv2s_1536 --weights 1 1 --split test
  # with the learned linker (train_linker_v3.py) as extra families H7@0.4 and H7@0.6:
  python src/build_pool_v3.py ... --linker work/linker_v3/<tag>/linker.pkl --link-taus 0.4 0.6

Probability maps come from $FILSEG_WORK/preds/<member>/ (v1, v2 and v3 members alike; S/D maps are used when
every member that has them is a v3 member with aux heads). For out-of-fold images the linker model of the
image's own fold is used (it never saw that fold); for test images the linker trained on all images.

Writes to <out>/<split>/ (default out: $FILSEG_WORK/pools_v3/<members>):
  features.parquet   one row per candidate: image, cand, fold, features (filseg/pool_v3.py) and, for oof,
                     the targets a (sum of TP IoUs over the image's sets), u = a / K, m_frac, best_iou, K
  masks/<i>.pkl      the candidates' masks (packed crops) of image i
  pool_config.json
For oof it also prints the pool oracle (overlap-aware greedy selection by the true u; blueprint 12.2) next to the
same oracle restricted to H1, i.e. how much the extra hypotheses could add at best.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from filseg.data import CACHE_DIR, WORK_ROOT
from filseg.linker_v3 import Linker
from filseg.masks_v3 import gt_crops
from filseg.metric import PQStats
from filseg.pool_v3 import DEFAULT_FAMILIES, build_pool, family_names
from filseg.postprocess import upsample
from filseg.select_v3 import candidate_targets, greedy_select, image_stats

ORACLE_TAUS = np.round(np.arange(0.12, 0.34, 0.02), 2)


class MemberMaps:
    """Blended P (v1 load_blend semantics), per-member P for disagreement, blended S/D if available."""

    def __init__(self, names: list[str], weights: list[float], split: str):
        sub = "oof" if split == "oof" else "test_folds"
        root = os.path.join(WORK_ROOT, "preds")
        names, weights = zip(*[(n, w) for n, w in zip(names, weights) if w > 0])
        self.p = [np.load(os.path.join(root, n, f"{sub}.npy"), mmap_mode="r") for n in names]
        self.w = np.asarray(weights, np.float32) / np.sum(weights)
        self.s, self.d = [], []
        for n, w in zip(names, self.w):
            ps, pd_ = os.path.join(root, n, f"{sub}_s.npy"), os.path.join(root, n, f"{sub}_d.npy")
            if w > 0 and os.path.exists(ps) and os.path.exists(pd_):
                self.s.append((w, np.load(ps, mmap_mode="r")))
                self.d.append((w, np.load(pd_, mmap_mode="r")))

    def __len__(self):
        return len(self.p[0])

    def get(self, i: int):
        parts = [np.asarray(m[i], np.float32) for m in self.p]
        size = max(q.shape[0] for q in parts)
        p = sum(w * (q if q.shape[0] == size else upsample(q, size)) for w, q in zip(self.w, parts))
        p = upsample(p)
        members = [upsample(q) for q, w in zip(parts, self.w) if w > 0]
        s = d = None
        if self.s:
            ws = sum(w for w, _ in self.s)
            s = upsample(sum(w * np.asarray(a[i], np.float32) for w, a in self.s) / (255.0 * ws))
            d = upsample(sum(w * np.asarray(a[i], np.float32) for w, a in self.d) / (255.0 * ws))
        return p, members, s, d


_G = {}


def _init(names, weights, split, linker_path, conf, out_dir):
    _G["maps"] = MemberMaps(names, weights, split)
    _G["images"] = np.load(os.path.join(CACHE_DIR, f"{'train' if split == 'oof' else 'test'}_images.npy"),
                           mmap_mode="r")
    with open(os.path.join(CACHE_DIR, "train_meta.pkl" if split == "oof" else "test_meta.pkl"), "rb") as f:
        _G["meta"] = pickle.load(f)
    _G["linker"] = Linker.load(linker_path) if linker_path else None
    _G.update(split=split, conf=conf, out_dir=out_dir)


def _image(i):
    meta, conf = _G["meta"], _G["conf"]
    p, members, s, d = _G["maps"].get(i)
    image = np.asarray(_G["images"][i])
    fold = int(meta["folds"][i]) if _G["split"] == "oof" else None
    masks, recs = build_pool(p, image, members, s, d, conf["families"], _G["linker"], conf["link_taus"],
                             fold, conf["min_area"])
    with open(os.path.join(_G["out_dir"], "masks", f"{i}.pkl"), "wb") as f:
        pickle.dump([m.packed() for m in masks], f)
    for k, r in enumerate(recs):
        r.update(image=i, cand=k, fold=-1 if fold is None else fold)
    oracle = None
    if _G["split"] == "oof":
        gts = gt_crops(meta["gt_rles"][meta["stems"][i]])
        k_sets = len(gts)
        a, mf, best = candidate_targets(gts, masks)
        for r, ai, mi, bi in zip(recs, a, mf, best):
            r.update(a=ai, u=ai / k_sets, m_frac=mi, best_iou=bi, K=k_sets)
        u = a / k_sets
        is_h1 = np.array([r.get("in_H1", 0) == 1 and r["family"] == 0 for r in recs], bool)
        oracle = {}
        for name, sel in (("pool", np.ones(len(recs), bool)), ("H1", is_h1)):
            sub = [m for m, ok in zip(masks, sel) if ok]
            us = u[sel]
            oracle[name] = {float(t): image_stats(gts, greedy_select(sub, us, t, 0.1)[0]) for t in ORACLE_TAUS}
    return recs, oracle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--split", choices=["oof", "test"], required=True)
    ap.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    ap.add_argument("--linker", default="", help="linker.pkl from train_linker_v3.py (adds families H7@tau)")
    ap.add_argument("--link-taus", type=float, nargs="*", default=[0.4, 0.6])
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    weights = args.weights or [1.0] * len(args.names)
    tag = "+".join(n for n, w in zip(args.names, weights) if w > 0)
    root = args.out or os.path.join(WORK_ROOT, "pools_v3", tag + ("_linker" if args.linker else ""))
    out_dir = os.path.join(root, args.split)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    link_taus = args.link_taus if args.linker else []
    conf = dict(names=args.names, weights=weights, families=args.families, link_taus=link_taus,
                linker=os.path.abspath(args.linker) if args.linker else "", min_area=args.min_area,
                columns=family_names(args.families, link_taus))
    with open(os.path.join(CACHE_DIR, "train_meta.pkl" if args.split == "oof" else "test_meta.pkl"), "rb") as f:
        n_img = len(pickle.load(f)["stems"])
    t0 = time.time()
    with Pool(args.workers, initializer=_init,
              initargs=(args.names, weights, args.split, args.linker, conf, out_dir)) as pool:
        res = pool.map(_image, range(n_img), chunksize=2)
    rows = [r for recs, _ in res for r in recs]
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(out_dir, "features.parquet"))
    with open(os.path.join(out_dir, "pool_config.json"), "w") as f:
        json.dump(conf, f, indent=1)
    names = conf["columns"]
    fam_counts = df["family"].map(dict(enumerate(names))).value_counts().to_dict() if len(df) else {}
    print(f"{args.split}: {len(df)} candidates over {df['image'].nunique() if len(df) else 0}/{n_img} images "
          f"({time.time() - t0:.0f}s on {args.workers} workers); kept per family: {fam_counts}")
    if args.split == "oof":
        with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        n_gt = sum(len(g) for s in meta["stems"] for g in meta["gt_rles"][s])
        res_or = {}
        for name in ("H1", "pool"):
            best = None
            for t in ORACLE_TAUS:
                tot = PQStats()
                for _, o in res:  # every image contributes, also those without candidates (all FN)
                    tot.add(o[name][float(t)])
                if best is None or tot.pq > best[1].pq:
                    best = (float(t), tot)
            res_or[name] = best
        for name, (t, tot) in res_or.items():
            print(f"oracle (greedy, true u > {t}, overlap <= 10%) {name:>4}: {tot}")
        print(f"pool gain over H1 at best: {res_or['pool'][1].pq - res_or['H1'][1].pq:+.4f} "
              f"(Lever B is worth keeping only if this is clearly positive); N_gt {n_gt}")
        with open(os.path.join(out_dir, "oracle.json"), "w") as f:
            json.dump({k: {"tau": t, "pq": s.pq, "sq": s.sq, "rq": s.rq} for k, (t, s) in res_or.items()}, f, indent=1)


if __name__ == "__main__":
    main()
