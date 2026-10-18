"""v3 learned fragment linker (blueprint 12.3): which nearby pieces of P > 0.6 belong to one filament?

Nodes are the connected components of P > t (area >= min_area px); edges join nodes whose boundary gap is at
most ``max_gap`` px. Each edge gets geometric and photometric features (gap, sizes, widths, orientation and
end-tangent alignment, P / S / D / intensity along the straight segment and along the cheapest path through
1 - P, position on the disk, endpoint-to-endpoint versus T-junction). A LightGBM regressor predicts p_link, the
share of annotators that put both pieces in one polygon (train_linker_v3.py). Pieces joined by edges with
p_link > tau form one candidate (union-find): hypothesis family H7@tau of the candidate pool.
"""
from __future__ import annotations

import pickle

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects
from skimage.graph import route_through_array
from skimage.morphology import skeletonize

FEATURES = ["gap", "gap_rel", "area_min", "area_max", "len_min", "len_max", "width_min", "width_max",
            "angle", "align_min", "align_max", "seg_mean", "seg_min", "seg_p10", "path_mean", "path_min",
            "path_ratio", "path_s", "path_d", "rho", "contrast", "end_min", "end_max", "pmean_min", "pmean_max"]


def nodes(p: np.ndarray, t: float = 0.6, min_area: int = 20) -> tuple[np.ndarray, int, list]:
    """Label map of the kept components (1..n) and their slices."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats((p > t).astype(np.uint8), connectivity=8)
    keep = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
    remap = np.zeros(n, np.int32)
    remap[keep] = np.arange(1, len(keep) + 1)
    lab = remap[lab]
    return lab, len(keep), find_objects(lab, max_label=len(keep))


def edges(lab: np.ndarray, n: int, slices: list, max_gap: int = 48) -> list[tuple]:
    """(i, j, gap, (yi, xi), (yj, xj)) for node pairs i < j with gap <= max_gap; the points are the closest
    pixels of the two nodes."""
    h, w = lab.shape
    out = []
    for i in range(1, n + 1):
        sl = slices[i - 1]
        y0, y1 = max(0, sl[0].start - max_gap), min(h, sl[0].stop + max_gap)
        x0, x1 = max(0, sl[1].start - max_gap), min(w, sl[1].stop + max_gap)
        crop = lab[y0:y1, x0:x1]
        others = np.unique(crop)
        others = others[others > i]
        if len(others) == 0:
            continue
        dist, (iy, ix) = distance_transform_edt(crop != i, return_indices=True)
        for j in others:
            ys, xs = np.nonzero(crop == j)
            dj = dist[ys, xs]
            k = int(np.argmin(dj))
            g = float(dj[k])
            if g <= max_gap:
                pj = (int(ys[k]) + y0, int(xs[k]) + x0)
                pi = (int(iy[ys[k], xs[k]]) + y0, int(ix[ys[k], xs[k]]) + x0)
                out.append((i, int(j), g, pi, pj))
    return out


def _node_info(lab, slices, n, p, image):
    info = {}
    for i in range(1, n + 1):
        sl = slices[i - 1]
        m = lab[sl] == i
        sk = skeletonize(m)
        ys, xs = np.nonzero(m)
        cov = np.cov(np.stack([ys, xs])) if len(ys) > 2 else np.eye(2)
        ev, evec = np.linalg.eigh(cov)
        major = evec[:, int(np.argmax(ev))]
        sky, skx = np.nonzero(sk)
        # skeleton endpoints: skeleton pixels with exactly one 8-neighbour on the skeleton
        nb = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT)
        ey, ex = np.nonzero(sk & (nb == 2))
        area = int(m.sum())
        length = max(int(sk.sum()), 1)
        info[i] = dict(area=area, length=length, width=area / length, major=major,
                       sk=np.stack([sky + sl[0].start, skx + sl[1].start], 1),
                       ends=np.stack([ey + sl[0].start, ex + sl[1].start], 1),
                       pmean=float(p[sl][m].mean()), imean=float(image[sl][m].mean()))
    return info


def _tangent(sk: np.ndarray, pt, r: float = 15.0) -> np.ndarray | None:
    if len(sk) < 3:
        return None
    d = np.hypot(sk[:, 0] - pt[0], sk[:, 1] - pt[1])
    near = sk[d <= r]
    if len(near) < 3:
        near = sk[np.argsort(d)[:min(len(sk), 15)]]
    c = near - near.mean(0)
    _, _, vt = np.linalg.svd(c.astype(np.float64), full_matrices=False)
    return vt[0]


def _sample(a: np.ndarray, pi, pj, n: int) -> np.ndarray:
    ys = np.linspace(pi[0], pj[0], n)
    xs = np.linspace(pi[1], pj[1], n)
    return a[np.round(ys).astype(int).clip(0, a.shape[0] - 1), np.round(xs).astype(int).clip(0, a.shape[1] - 1)]


def edge_features(p, s, d, image, lab, n, slices, edge_list, disk) -> np.ndarray:
    """Feature matrix (n_edges, len(FEATURES)); s / d may be None (NaN features)."""
    if not edge_list:
        return np.zeros((0, len(FEATURES)))
    cy, cx, radius = disk
    info = _node_info(lab, slices, n, p, image)
    rows = []
    for i, j, gap, pi, pj in edge_list:
        a, b = info[i], info[j]
        vec = np.array([pj[0] - pi[0], pj[1] - pi[1]], np.float64)
        nv = np.linalg.norm(vec)
        vec = vec / nv if nv > 0 else np.zeros(2)
        aligns = []
        for nd, pt in ((a, pi), (b, pj)):
            t = _tangent(nd["sk"], pt)
            aligns.append(abs(float(t @ vec)) if (t is not None and nv > 0) else np.nan)
        angle = float(np.degrees(np.arccos(np.clip(abs(a["major"] @ b["major"]), 0, 1))))
        ns = max(2, int(np.ceil(gap)) + 1)
        seg = _sample(p, pi, pj, ns)
        seg_img = _sample(image, pi, pj, ns).astype(np.float64)
        # cheapest path through 1 - P inside a corridor around the two closest points
        m = 16
        y0, y1 = max(0, min(pi[0], pj[0]) - m), min(p.shape[0], max(pi[0], pj[0]) + m + 1)
        x0, x1 = max(0, min(pi[1], pj[1]) - m), min(p.shape[1], max(pi[1], pj[1]) + m + 1)
        cost = 1.0 - p[y0:y1, x0:x1].astype(np.float64) + 0.01
        path, _ = route_through_array(cost, (pi[0] - y0, pi[1] - x0), (pj[0] - y0, pj[1] - x0),
                                      fully_connected=True, geometric=True)
        path = np.asarray(path)
        pp = p[y0:y1, x0:x1][path[:, 0], path[:, 1]]
        ps = s[y0:y1, x0:x1][path[:, 0], path[:, 1]] if s is not None else None
        pd_ = d[y0:y1, x0:x1][path[:, 0], path[:, 1]] if d is not None else None
        ends = []
        for nd, pt in ((a, pi), (b, pj)):
            e = nd["ends"]
            ends.append(float(np.hypot(e[:, 0] - pt[0], e[:, 1] - pt[1]).min()) if len(e) else 99.0)
        mid = ((pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2)
        rows.append([
            gap, gap / max((a["width"] + b["width"]) / 2, 1.0),
            min(a["area"], b["area"]), max(a["area"], b["area"]),
            min(a["length"], b["length"]), max(a["length"], b["length"]),
            min(a["width"], b["width"]), max(a["width"], b["width"]),
            angle, np.nanmin(aligns) if not np.all(np.isnan(aligns)) else np.nan,
            np.nanmax(aligns) if not np.all(np.isnan(aligns)) else np.nan,
            float(seg.mean()), float(seg.min()), float(np.percentile(seg, 10)),
            float(pp.mean()), float(pp.min()), len(path) / max(gap, 1.0),
            float(ps.mean()) if ps is not None else np.nan, float(pd_.max()) if pd_ is not None else np.nan,
            float(np.hypot(mid[0] - cy, mid[1] - cx) / radius),
            float(seg_img.mean() - (a["imean"] + b["imean"]) / 2),
            min(ends), max(ends), min(a["pmean"], b["pmean"]), max(a["pmean"], b["pmean"]),
        ])
    return np.asarray(rows, np.float64)


def group(lab: np.ndarray, n: int, edge_list: list, p_link: np.ndarray, tau: float
          ) -> tuple[np.ndarray, int, dict]:
    """Union-find over edges with p_link > tau. Returns the group label map (1..G), G, and per-group stats
    {label: dict(n_parts, p_min, p_mean)} (p over the edges used)."""
    parent = np.arange(n + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used = []
    for (i, j, *_), pl in zip(edge_list, p_link):
        if pl > tau:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
            used.append((i, pl))
    roots = np.array([find(x) for x in range(n + 1)])
    uniq, newid = np.unique(roots[1:], return_inverse=True)
    remap = np.zeros(n + 1, np.int32)
    remap[1:] = newid + 1
    glab = remap[lab]
    parts = np.bincount(remap[1:], minlength=len(uniq) + 1)
    stats = {g: dict(n_parts=int(parts[g]), p_min=np.nan, p_mean=np.nan) for g in range(1, len(uniq) + 1)}
    acc: dict[int, list] = {}
    for i, pl in used:
        acc.setdefault(int(remap[i]), []).append(pl)
    for g, v in acc.items():
        stats[g].update(p_min=float(np.min(v)), p_mean=float(np.mean(v)))
    return glab, len(uniq), stats


class Linker:
    """LightGBM ensembles: one per CV fold (trained on the other folds) plus one on all images."""

    def __init__(self, folds: dict[int, list], full: list, t: float, min_area: int, max_gap: int):
        self.folds, self.full = folds, full
        self.t, self.min_area, self.max_gap = t, min_area, max_gap

    def predict(self, x: np.ndarray, fold: int | None = None) -> np.ndarray:
        if len(x) == 0:
            return np.zeros(0)
        boosters = self.full if fold is None else self.folds[fold]
        return np.clip(np.mean([b.predict(x, num_threads=1) for b in boosters], axis=0), 0, 1)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"folds": {k: [b.model_to_string() for b in v] for k, v in self.folds.items()},
                         "full": [b.model_to_string() for b in self.full], "t": self.t,
                         "min_area": self.min_area, "max_gap": self.max_gap, "features": FEATURES}, f)

    @classmethod
    def load(cls, path: str) -> "Linker":
        import lightgbm as lgb
        with open(path, "rb") as f:
            d = pickle.load(f)
        mk = lambda ss: [lgb.Booster(model_str=s) for s in ss]  # noqa: E731
        return cls({int(k): mk(v) for k, v in d["folds"].items()}, mk(d["full"]), d["t"], d["min_area"],
                   d["max_gap"])

    def groups(self, p, s, d, image, disk, taus, fold=None) -> tuple[dict, np.ndarray, int, list]:
        """{tau: (group label map, G, stats)} plus the node label map, n and edges (for reuse)."""
        lab, n, slices = nodes(p, self.t, self.min_area)
        el = edges(lab, n, slices, self.max_gap)
        pl = self.predict(edge_features(p, s, d, image, lab, n, slices, el, disk), fold)
        return {tau: group(lab, n, el, pl, tau) for tau in taus}, lab, n, el
