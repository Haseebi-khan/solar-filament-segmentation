"""v3 multi-hypothesis candidate pool (blueprint 12.2) with the features of the stacked scorer (12.5).

Each family runs v1's extraction (seeds p > t_seed, pieces within ~2*link px grouped, grown to p > t_mask,
holes filled) with other settings; H7@tau groups the pieces with the learned linker instead of a fixed radius:

  H1 0.6/0.6/8 (v1 optimum)  H2 0.5/0.5/8 (looser)  H3 0.7/0.7/8 (tighter)  H4 0.6/0.45/8 (watershed growth)
  H5 0.6/0.6/0 (no grouping)  H6 0.6/0.6/16 (aggressive grouping)  H7@tau (linker groups, p > 0.6)

Candidates of all families are pooled; candidates with mutual IoU > 0.95 are merged into the first one (family
order above, so H1's version is kept) and ``stability`` counts how many families produced it. Every candidate
keeps v1's 26 rescorer features plus: family, stability and family membership bits, ensemble disagreement
(mean per-pixel std of the member maps inside the mask), threshold persistence, skeleton and core statistics
(S/D heads of v3 members, NaN otherwise), linker statistics (H7 only), overlap with the rest of the pool and the
best-matching H1 candidate's statistics.
"""
from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize

from .masks_v3 import CropMask, inter_matrix, iou_from_inter, masks_from_label
from .postprocess import _close_instances, add_features, disk_geometry, instance_table, label_instances
from .rescore import FEATURES as V1_FEATURES

FAMILIES = {
    "H1": (0.6, 0.6, 8),
    "H2": (0.5, 0.5, 8),
    "H3": (0.7, 0.7, 8),
    "H4": (0.6, 0.45, 8),
    "H5": (0.6, 0.6, 0),
    "H6": (0.6, 0.6, 16),
}
DEFAULT_FAMILIES = list(FAMILIES)
DEDUPE_IOU = 0.95


def family_names(families: list[str], link_taus: list[float]) -> list[str]:
    return list(families) + [f"H7@{t:g}" for t in link_taus]


def _records(p, lab, n, image, min_area, extra=None) -> tuple[list[CropMask], list[dict]]:
    objs = [o for o in instance_table(p, lab, n) if o["area"] >= min_area]
    if not objs:
        return [], []
    add_features(objs, p, lab, image)
    masks = masks_from_label(lab, n)
    out_m, out_r = [], []
    for o in objs:
        m = masks.get(o["label"])
        if m is None:
            continue
        r = {k: o[k] for k in o if k not in ("label", "y0", "y1", "x0", "x1")}
        r["log_area"] = float(np.log1p(o["area"]))
        r["bbox_h"] = o["y1"] - o["y0"]
        r["bbox_w"] = o["x1"] - o["x0"]
        if extra is not None:
            r.update(extra.get(o["label"], {}))
        out_m.append(m)
        out_r.append(r)
    return out_m, out_r


def build_pool(p: np.ndarray, image: np.ndarray, members: list[np.ndarray] | None = None,
               s: np.ndarray | None = None, d: np.ndarray | None = None, families=DEFAULT_FAMILIES,
               linker=None, link_taus=(), fold: int | None = None, min_area: int = 100
               ) -> tuple[list[CropMask], list[dict]]:
    """p: blended probability (2048, 2048) float32; members: per-member maps (2048, 2048) for disagreement;
    s, d: blended S/D maps in [0, 1] or None. Returns deduplicated candidates and their feature records."""
    names = family_names(families, link_taus if linker is not None else ())
    fam_id = {f: k for k, f in enumerate(names)}
    all_m, all_r = [], []
    for f in families:
        t_seed, t_mask, link = FAMILIES[f]
        n, lab = label_instances(p, t_seed, t_mask, link, close=-1)
        ms, rs = _records(p, lab, n, image, min_area)
        for r in rs:
            r.update(family=fam_id[f], t_seed=t_seed, t_mask=t_mask, link=link,
                     lk_parts=np.nan, lk_pmin=np.nan, lk_pmean=np.nan)
        all_m += ms
        all_r += rs
    if linker is not None and link_taus:
        groups, *_ = linker.groups(p, s, d, image, disk_geometry(image), list(link_taus), fold)
        for tau in link_taus:
            glab, g, stats = groups[tau]
            glab = _close_instances(glab.astype(np.int32), g, -1) if g else glab
            extra = {k: dict(lk_parts=v["n_parts"], lk_pmin=v["p_min"], lk_pmean=v["p_mean"]) for k, v in stats.items()}
            ms, rs = _records(p, glab, g, image, min_area, extra)
            for r in rs:
                r.update(family=fam_id[f"H7@{tau:g}"], t_seed=linker.t, t_mask=linker.t, link=-1)
                r.setdefault("lk_parts", np.nan)
                r.setdefault("lk_pmin", np.nan)
                r.setdefault("lk_pmean", np.nan)
            all_m += ms
            all_r += rs
    if not all_m:
        return [], []

    # ---- deduplicate (first occurrence wins, families in order)
    ious = iou_from_inter(inter_matrix(all_m, all_m), all_m, all_m)
    keep, rep = [], np.full(len(all_m), -1)
    for i in range(len(all_m)):
        prev = [j for j in keep if ious[i, j] > DEDUPE_IOU]
        if prev:
            rep[i] = prev[0]
        else:
            keep.append(i)
            rep[i] = i
    masks = [all_m[i] for i in keep]
    recs = [dict(all_r[i]) for i in keep]
    pos = {i: k for k, i in enumerate(keep)}
    for k in range(len(recs)):
        recs[k]["stability"] = 0
        for f in names:
            recs[k][f"in_{f}"] = 0
    for i in range(len(all_m)):
        r = recs[pos[rep[i]]]
        r["stability"] += 1
        r[f"in_{names[all_r[i]['family']]}"] = 1

    # ---- overlap with the rest of the pool and the best-matching H1 candidate
    sub = ious[np.ix_(keep, keep)].copy()
    np.fill_diagonal(sub, 0.0)
    is_h1 = np.array([r["family"] == fam_id.get("H1", -1) for r in recs])
    for k, r in enumerate(recs):
        r["pool_max_iou"] = float(sub[k].max()) if len(recs) > 1 else 0.0
        r["pool_n_overlap"] = int((sub[k] > 0.1).sum())
        if is_h1.any():
            cand = np.where(is_h1)[0]
            v = np.array([1.0 if c == k else sub[k, c] for c in cand])
            b = int(cand[int(np.argmax(v))])
            r["base_iou"] = float(v.max())
            rb = recs[b]
            r["base_mean"], r["base_p90"], r["base_core80"] = rb["mean"], rb["p90"], rb["core80"]
            r["base_log_area"] = rb["log_area"]
            r["area_ratio_base"] = r["area"] / max(rb["area"], 1)
        else:
            for key in ("base_iou", "base_mean", "base_p90", "base_core80", "base_log_area", "area_ratio_base"):
                r[key] = np.nan

    # ---- pixel statistics inside each mask
    std = np.std(np.stack(members), axis=0) if members is not None and len(members) > 1 else None
    for m, r in zip(masks, recs):
        sl = (slice(m.y0, m.y1), slice(m.x0, m.x1))
        pin = p[sl][m.m]
        a5 = max(int((pin > 0.5).sum()), 1)
        r["persist"] = float((pin > 0.7).sum()) / a5
        r["disagree"] = float(std[sl][m.m].mean()) if std is not None else np.nan
        if s is not None:
            sk = skeletonize(m.m)
            sv = s[sl][sk] if sk.any() else s[sl][m.m]
            r["s_skel_mean"] = float(sv.mean())
            r["s_skel_frac"] = float((sv > 0.5).mean())
        else:
            r["s_skel_mean"] = r["s_skel_frac"] = np.nan
        if d is not None:
            dv = d[sl][m.m]
            r["d_mean"], r["d_max"] = float(dv.mean()), float(dv.max())
        else:
            r["d_mean"] = r["d_max"] = np.nan
    return masks, recs


def feature_columns(families: list[str], link_taus) -> list[str]:
    names = family_names(families, link_taus)
    return (list(V1_FEATURES) + ["family", "t_seed", "t_mask", "link", "stability"] + [f"in_{f}" for f in names]
            + ["lk_parts", "lk_pmin", "lk_pmean", "pool_max_iou", "pool_n_overlap", "base_iou", "base_mean",
               "base_p90", "base_core80", "base_log_area", "area_ratio_base", "persist", "disagree",
               "s_skel_mean", "s_skel_frac", "d_mean", "d_max"])
