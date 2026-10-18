"""v3 Phase 0: how far can a single submission go on this data? (blueprint section 14, items 2-3)

  python src/ceilings_v3.py [--oof-csv work/oof_sub_v2_<members>.csv]

1. Noise ceiling, leave one annotator out (images with 3 annotation sets): a consensus of the other two sets,
   built with a fixed rule, is scored against the held-out set (organisers' PQ, pooled over all held-out sets):
     single          one other annotator as is (= human-human agreement on these images)
     union / inter   pixel union / intersection of the two others, connected components, holes filled
     union_l8 / inter_l8   the same with v1's fragment grouping (pieces within ~16 px form one filament)
     matched         only filaments both others drew (IoU > 0.5), mask = union of the pair
     matched_plus    matched pairs plus every unmatched filament of either annotator
   The best rule estimates what a perfect consensus predictor could reach.
2. Annotator-aware oracle (images with 2+ sets): candidates are every annotator's polygons plus unions and
   intersections of overlapping polygons of different annotators; the overlap-aware greedy selection keeps
   candidates by their true u (mean per-set TP-IoU) above a threshold (best of a grid). This bounds any single
   disjoint submission from above (greedy, so approximately) on those images.
With --oof-csv the model's PQ on the same image subsets is printed next to them. Writes
$FILSEG_WORK/diag_v3/ceilings.json and ceilings.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from filseg.data import CACHE_DIR, WORK_ROOT
from filseg.masks_v3 import CropMask, gt_crops, iou_matrix_crops, masks_from_label, paste
from filseg.metric import PQStats
from filseg.postprocess import _close_instances, _disk
from filseg.select_v3 import candidate_targets, greedy_select, image_stats, pq_stats_crops

RULES = ["single", "union", "inter", "union_l8", "inter_l8", "matched", "matched_plus"]
TAUS = np.round(np.arange(0.05, 0.55, 0.05), 2)


def components(full: np.ndarray, link: int = 0, min_area: int = 30) -> list[CropMask]:
    seeds = full.astype(np.uint8)
    if link > 0:
        n, groups = cv2.connectedComponents(cv2.dilate(seeds, _disk(link)), connectivity=8)
        lab = (groups * seeds).astype(np.int32)
    else:
        n, lab = cv2.connectedComponents(seeds, connectivity=8)
        lab = lab.astype(np.int32)
    lab = _close_instances(lab, n - 1, -1) if n > 1 else lab
    return [m for m in masks_from_label(lab, n - 1).values() if m.area >= min_area]


def disjoint(masks: list[CropMask]) -> list[CropMask]:
    occ = np.zeros((2048, 2048), bool)
    out = []
    for m in masks:
        t = CropMask(m.y0, m.x0, m.m & ~occ[m.y0:m.y1, m.x0:m.x1]).tight()
        if t.area:
            paste(occ, t)
            out.append(t)
    return out


def union_full(sets: list[list[CropMask]]) -> np.ndarray:
    full = np.zeros((2048, 2048), bool)
    for s in sets:
        for m in s:
            paste(full, m)
    return full


def consensus(a: list[CropMask], b: list[CropMask], rule: str) -> list[CropMask]:
    if rule in ("union", "union_l8"):
        return components(union_full([a, b]), 8 if rule.endswith("l8") else 0)
    if rule in ("inter", "inter_l8"):
        return components(union_full([a]) & union_full([b]), 8 if rule.endswith("l8") else 0)
    ious = iou_matrix_crops(a, b)
    pairs, used_a, used_b = [], set(), set()
    for i, j in sorted(zip(*np.nonzero(ious > 0.5)), key=lambda ij: -ious[ij]):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append(CropMask.from_full(a[i].full() | b[j].full()))
    if rule == "matched":
        return disjoint(pairs)
    rest = [m for i, m in enumerate(a) if i not in used_a] + [m for j, m in enumerate(b) if j not in used_b]
    return disjoint(pairs + sorted(rest, key=lambda m: -m.area))


def _noise(i):
    meta = _G["meta"]
    stem = meta["stems"][i]
    sets = gt_crops(meta["gt_rles"][stem])
    out = {r: PQStats() for r in RULES}
    for h in range(len(sets)):
        a, b = [sets[j] for j in range(len(sets)) if j != h]
        out["single"].add(pq_stats_crops(sets[h], a)).add(pq_stats_crops(sets[h], b))
        for rule in RULES[1:]:
            out[rule].add(pq_stats_crops(sets[h], consensus(a, b, rule)))
    return out


def _oracle(i):
    meta = _G["meta"]
    stem = meta["stems"][i]
    sets = gt_crops(meta["gt_rles"][stem])
    cands = [m for s in sets for m in s]
    for s1, s2 in itertools.combinations(range(len(sets)), 2):
        ious = iou_matrix_crops(sets[s1], sets[s2])
        for a_i, b_j in zip(*np.nonzero(ious > 0.1)):
            fa, fb = sets[s1][a_i].full(), sets[s2][b_j].full()
            cands.append(CropMask.from_full(fa | fb))
            inter = CropMask.from_full(fa & fb)
            if inter.area >= 30:
                cands.append(inter)
    a, _, _ = candidate_targets(sets, cands)
    u = a / len(sets)
    return {float(t): image_stats(sets, greedy_select(cands, u, t, 0.1)[0]) for t in TAUS}


_G = {}


def _init(meta):
    _G["meta"] = meta


def pooled(stats_list):
    tot = PQStats()
    for s in stats_list:
        tot.add(s)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-csv", default=None, help="out-of-fold CSV of a model, scored on the same images")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(WORK_ROOT, "diag_v3"))
    args = ap.parse_args()
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    n_sets = meta["n_sets"]
    three = [i for i in range(len(meta["stems"])) if n_sets[i] == 3]
    multi = [i for i in range(len(meta["stems"])) if n_sets[i] >= 2]
    with Pool(args.workers, initializer=_init, initargs=(meta,)) as pool:
        noise = pool.map(_noise, three, chunksize=2)
        oracle = pool.map(_oracle, multi, chunksize=2)
    res = {"noise_ceiling": {}, "oracle": {}}
    lines = [f"## Ceilings (Phase 0)\n\n### Leave-one-annotator-out consensus ({len(three)} images with 3 sets)\n",
             "| rule | PQ | SQ | RQ |", "|---|---|---|---|"]
    for rule in RULES:
        s = pooled([n[rule] for n in noise])
        res["noise_ceiling"][rule] = {"PQ": s.pq, "SQ": s.sq, "RQ": s.rq}
        lines.append(f"| {rule} | {s.pq:.4f} | {s.sq:.4f} | {s.rq:.4f} |")
    best_t = max(TAUS, key=lambda t: pooled([o[float(t)] for o in oracle]).pq)
    s = pooled([o[float(best_t)] for o in oracle])
    res["oracle"] = {"PQ": s.pq, "SQ": s.sq, "RQ": s.rq, "tau": float(best_t), "images": len(multi)}
    lines += [f"\n### Annotator-aware oracle ({len(multi)} images with 2+ sets)\n",
              f"PQ **{s.pq:.4f}** (SQ {s.sq:.4f}, RQ {s.rq:.4f}) at u > {best_t}"]
    if args.oof_csv:
        from filseg.metric import pq_stats
        df = pd.read_csv(args.oof_csv)
        stems = df["filament_id"].str.split("_", n=1).str[0]
        preds = {k: [{"size": [2048, 2048], "counts": c.encode()} for c in g["segmentation_rle"]]
                 for k, g in df.groupby(stems)}
        for name, idx in [("3-set images", three), ("2+-set images", multi)]:
            tot = PQStats()
            for i in idx:
                st = meta["stems"][i]
                for g in meta["gt_rles"][st]:
                    tot.add(pq_stats(g, preds.get(st, [])))
            res[f"model_{name}"] = {"PQ": tot.pq}
            lines.append(f"\nmodel ({os.path.basename(args.oof_csv)}) on the {name}: PQ {tot.pq:.4f}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ceilings.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(args.out, "ceilings.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
