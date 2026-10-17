"""Second-stage instance rescoring.

A permissive post-processing proposes candidate instances. A gradient-boosted
regressor predicts, for each candidate, u = mean over the image's annotation sets
of (IoU if IoU > 0.5 else 0). Because predictions are disjoint, adding an
instance to an image with m annotation sets raises the PQ numerator by m*u and
the denominator by m/2, so it improves PQ iff u > PQ/2. The keep-threshold is
tuned on out-of-fold data.
"""
from __future__ import annotations

import pickle

import numpy as np

FEATURES = ["area", "log_area", "mean", "max", "p90", "i_in", "i_min", "i_ring", "contrast", "core80", "core90",
            "p_ring", "mass", "major", "elong", "fill", "rho", "n_inst", "rank_mean", "bbox_h", "bbox_w",
            "n_parts", "length", "width", "ring_p30", "nn_dist"]


def feature_matrix(objs: list[dict]) -> np.ndarray:
    rows = []
    for o in objs:
        f = dict(o)
        f["log_area"] = np.log1p(o["area"])
        f["bbox_h"] = o["y1"] - o["y0"]
        f["bbox_w"] = o["x1"] - o["x0"]
        rows.append([float(f[k]) for k in FEATURES])
    return np.asarray(rows, np.float64).reshape(len(objs), len(FEATURES))


class Rescorer:
    def __init__(self, boosters: list, threshold: float):
        self.boosters = boosters
        self.threshold = threshold

    def predict(self, objs: list[dict]) -> np.ndarray:
        if not objs:
            return np.zeros(0)
        x = feature_matrix(objs)
        return np.mean([b.predict(x) for b in self.boosters], axis=0)

    def select(self, objs: list[dict], p_full=None, image=None) -> list[bool]:
        return list(self.predict(objs) > self.threshold)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"boosters": [b.model_to_string() for b in self.boosters], "threshold": self.threshold}, f)

    @classmethod
    def load(cls, path: str) -> "Rescorer":
        import lightgbm as lgb
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls([lgb.Booster(model_str=s) for s in d["boosters"]], d["threshold"])
