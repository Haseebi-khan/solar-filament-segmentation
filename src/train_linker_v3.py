"""v3 Stage C2: train the learned fragment linker (filseg/linker_v3.py) on out-of-fold probability maps.

  python src/train_linker_v3.py --names r34_2048 effv2s_1536 --weights 1 1

Pair labels, per annotation set: a node "belongs" to the GT polygon that holds >= 50 % of its pixels; a pair is
1 if both nodes belong to the same polygon, 0 if they belong to different polygons, undefined otherwise. The
target is the mean over the sets with a defined label, the sample weight their number (soft multi-annotator
target). LightGBM regression (lr 0.03, 31 leaves, 500 rounds, bagging 0.8), nested over the 5 folds: the
model used for fold k never saw fold k; a sixth model trained on all images is used for the test images.

It then checks the grouping itself: for every tau, the candidates "pieces joined by edges with nested
p_link > tau" (holes filled, >= 100 px) are scored with the exact additive PQ (best threshold on the mean
probability, as tune_postprocess.py does) next to v1's fixed-radius grouping H1 on the same maps. Use the taus
that come out on top as --link-taus of build_pool_v3.py; the scorer decides per candidate.

Writes <out>/linker.pkl (default $FILSEG_WORK/linker_v3/<members>/) and <out>/report.txt.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np

from build_pool_v3 import MemberMaps
from filseg.data import CACHE_DIR, WORK_ROOT
from filseg.linker_v3 import FEATURES, Linker, edge_features, edges, group, nodes
from filseg.masks_v3 import gt_crops, masks_from_label
from filseg.postprocess import _close_instances, disk_geometry, instance_table, label_instances
from filseg.select_v3 import candidate_targets

PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=20,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
TAUS = (0.3, 0.4, 0.5, 0.6, 0.7)
_G = {}


def _init(names, weights, conf):
    _G["maps"] = MemberMaps(names, weights, "oof")
    _G["images"] = np.load(os.path.join(CACHE_DIR, "train_images.npy"), mmap_mode="r")
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        _G["meta"] = pickle.load(f)
    _G["conf"] = conf


def _graph(i):
    p, _, s, d = _G["maps"].get(i)
    image = np.asarray(_G["images"][i])
    c = _G["conf"]
    lab, n, slices = nodes(p, c["t"], c["min_area"])
    el = edges(lab, n, slices, c["max_gap"])
    return p, s, d, image, lab, n, slices, el


def _pairs(i):
    p, s, d, image, lab, n, slices, el = _graph(i)
    if not el:
        return None
    x = edge_features(p, s, d, image, lab, n, slices, el, disk_geometry(image))
    meta = _G["meta"]
    sets = gt_crops(meta["gt_rles"][meta["stems"][i]])
    node_px = {k: np.nonzero(lab == k) for k in range(1, n + 1)} if n < 400 else None
    lab_sets = []
    for gset in sets:
        g = np.zeros(lab.shape, np.int32)
        for k, m in enumerate(gset, start=1):
            g[m.y0:m.y1, m.x0:m.x1][m.m] = k
        owner = np.zeros(n + 1, np.int32)
        for k in range(1, n + 1):
            px = g[node_px[k]] if node_px is not None else g[lab == k]
            cnt = np.bincount(px, minlength=len(gset) + 1)
            top = int(np.argmax(cnt[1:])) + 1 if len(cnt) > 1 else 0
            owner[k] = top if top and cnt[top] >= 0.5 * len(px) else 0
        lab_sets.append(owner)
    y = np.zeros(len(el))
    w = np.zeros(len(el))
    for owner in lab_sets:
        for e, (a, b, *_) in enumerate(el):
            if owner[a] and owner[b]:
                y[e] += float(owner[a] == owner[b])
                w[e] += 1
    y = np.where(w > 0, y / np.maximum(w, 1), 0.0)
    return x, y, w, np.full(len(el), i)


def _grouping(args):
    i, p_link = args
    p, s, d, image, lab, n, slices, el = _graph(i)
    meta = _G["meta"]
    sets = gt_crops(meta["gt_rles"][meta["stems"][i]])
    k_sets = len(sets)
    rows = []
    variants = [("H1", None)] + [(f"H7@{t:g}", t) for t in TAUS]
    for name, tau in variants:
        if tau is None:
            g, glab = label_instances(p, 0.6, 0.6, 8, close=-1)
        else:
            glab, g, _ = group(lab, n, el, p_link if len(el) else np.zeros(0), tau)
            glab = _close_instances(glab.astype(np.int32), g, -1) if g else glab
        objs = [o for o in instance_table(p, glab, g) if o["area"] >= 100]
        ms = masks_from_label(glab, g)
        cands = [ms[o["label"]] for o in objs]
        a, _, _ = candidate_targets(sets, cands)
        rows += [(name, o["mean"], ai, k_sets) for o, ai in zip(objs, a)]
    return rows


def fit(x, y, w, seed=0, rounds=500):
    return lgb.train(dict(PARAMS, seed=seed), lgb.Dataset(x, y, weight=w, feature_name=FEATURES),
                     num_boost_round=rounds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--t", type=float, default=0.6, help="node threshold on P")
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--max-gap", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    weights = args.weights or [1.0] * len(args.names)
    tag = "+".join(n for n, w in zip(args.names, weights) if w > 0)
    out = args.out or os.path.join(WORK_ROOT, "linker_v3", tag)
    os.makedirs(out, exist_ok=True)
    conf = dict(t=args.t, min_area=args.min_area, max_gap=args.max_gap)
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    folds_img = meta["folds"]
    report = []

    def say(msg):
        print(msg, flush=True)
        report.append(msg)

    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.names, weights, conf)) as pool:
        parts = [r for r in pool.map(_pairs, range(len(meta["stems"])), chunksize=2) if r is not None]
        x = np.concatenate([q[0] for q in parts])
        y = np.concatenate([q[1] for q in parts])
        w = np.concatenate([q[2] for q in parts])
        img = np.concatenate([q[3] for q in parts]).astype(int)
        fold = folds_img[img]
        lab = w > 0
        say(f"edges {len(y)} over {len(parts)} images, labelled {int(lab.sum())} "
            f"(mean target {np.average(y[lab], weights=w[lab]):.3f}); {time.time() - t0:.0f}s")

        oof = np.zeros(len(y))
        fold_models = {}
        for k in np.unique(folds_img):
            tr = lab & (fold != k)
            fold_models[int(k)] = [fit(x[tr], y[tr], w[tr], seed=s) for s in range(args.seeds)]
            va = fold == k
            oof[va] = np.mean([b.predict(x[va]) for b in fold_models[int(k)]], axis=0)
        oof = oof.clip(0, 1)
        hard = lab & ((y == 0) | (y == 1))
        acc = np.average((oof[hard] > 0.5) == (y[hard] > 0.5), weights=w[hard])
        brier = np.average((oof[lab] - y[lab]) ** 2, weights=w[lab])
        base = np.average((np.average(y[lab], weights=w[lab]) - y[lab]) ** 2, weights=w[lab])
        say(f"nested: Brier {brier:.4f} (constant predictor {base:.4f}), accuracy on unanimous pairs {acc:.3f}")
        full = [fit(x[lab], y[lab], w[lab], seed=s) for s in range(args.seeds)]
        Linker(fold_models, full, args.t, args.min_area, args.max_gap).save(os.path.join(out, "linker.pkl"))
        imp = sorted(zip(FEATURES, full[0].feature_importance("gain")), key=lambda t: -t[1])
        say("feature gain: " + ", ".join(f"{k} {v:.0f}" for k, v in imp[:12]))

        by_img = {i: oof[img == i] for i in np.unique(img)}
        jobs = [(i, by_img.get(i, np.zeros(0))) for i in range(len(meta["stems"]))]
        rows = [r for rr in pool.map(_grouping, jobs, chunksize=2) for r in rr]
    n_gt = sum(len(g) for s in meta["stems"] for g in meta["gt_rles"][s])
    names = sorted({r[0] for r in rows}, key=lambda s: (s != "H1", s))
    say("grouping check (exact additive PQ, best threshold on the mean probability, >= 100 px):")
    for name in names:
        r = np.array([(m, a, k) for nm, m, a, k in rows if nm == name])
        best = (0.0, 0.0)
        for thr in np.round(np.arange(0.5, 0.96, 0.01), 2):
            keep = r[:, 0] >= thr
            pq = r[keep, 1].sum() / (0.5 * n_gt + 0.5 * r[keep, 2].sum())
            best = max(best, (pq, thr))
        say(f"  {name:>7}: PQ {best[0]:.4f} at mean >= {best[1]}  ({len(r)} candidates)")
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print("saved", os.path.join(out, "linker.pkl"))


if __name__ == "__main__":
    main()
