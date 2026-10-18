"""v3 unit tests (CPU, ~1 min, no data needed except the leakage tests, which skip without the cache).

  .venv/bin/python tests/test_v3.py          # or: .venv/bin/python -m pytest tests/test_v3.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from filseg.losses_v3 import blob_loss, dml1, jml1, soft_dice, supervised_loss, LossWeights  # noqa: E402
from filseg.masks_v3 import CropMask, iou, iou_matrix_crops  # noqa: E402
from filseg.metric import iou_matrix, pq_stats  # noqa: E402
from filseg.select_v3 import greedy_select, pq_stats_crops  # noqa: E402


# ----------------------------------------------------------------------------------------------- losses
def test_jml_dml_zero_with_zero_grad_at_target():
    torch.manual_seed(0)
    y = torch.rand(2, 1, 32, 32)
    p = y.clone().requires_grad_(True)
    for f in (jml1, dml1):
        loss = f(p, y)
        assert abs(loss.item()) < 1e-7
        (g,) = torch.autograd.grad(loss, p)
        assert g.abs().max().item() < 1e-7


def test_jml_dml_equal_soft_jaccard_dice_on_hard_labels():
    torch.manual_seed(1)
    p = torch.rand(3, 1, 16, 16)
    y = (torch.rand(3, 1, 16, 16) > 0.7).float()
    inter = (p * y).flatten(1).sum(1)
    s = p.flatten(1).sum(1) + y.flatten(1).sum(1)
    jac = (1 - inter / (s - inter)).mean()
    assert torch.allclose(jml1(p, y, eps=0.0), jac, atol=1e-6)
    assert torch.allclose(dml1(p, y, eps=0.0), soft_dice(p, y, eps=0.0), atol=1e-6)


def _blob_bruteforce(p, inst):
    losses = []
    for b in range(p.shape[0]):
        n = int(inst[b].max())
        terms = []
        for i in range(1, n + 1):
            g = (inst[b] == i).float()[None, None]
            if g.sum() == 0:
                continue
            others = ((inst[b] > 0) & (inst[b] != i)).float()[None, None]
            terms.append(jml1(p[b:b + 1] * (1 - others), g))
        if terms:
            losses.append(torch.stack(terms).mean())
    return torch.stack(losses).mean()


def test_blob_closed_form_matches_definition():
    torch.manual_seed(2)
    p = torch.rand(2, 1, 24, 24)
    inst = torch.zeros(2, 24, 24, dtype=torch.long)
    inst[0, 2:6, 3:20] = 1
    inst[0, 10:14, 5:9] = 2
    inst[1, 15:20, 2:22] = 3   # label 3 with 1, 2 absent
    got = blob_loss(p, inst, n_max=3)
    assert torch.allclose(got, _blob_bruteforce(p, inst), atol=1e-6)
    single = torch.zeros(1, 24, 24, dtype=torch.long)
    single[0, 4:9, 4:15] = 1
    assert torch.allclose(blob_loss(p[:1], single, 1), jml1(p[:1], (single == 1).float()[:, None]), atol=1e-6)


def test_supervised_loss_runs_and_backprops():
    torch.manual_seed(3)
    out = {"p": torch.randn(2, 1, 64, 64, requires_grad=True), "s": torch.randn(2, 1, 64, 64, requires_grad=True),
           "d": torch.randn(2, 1, 64, 64, requires_grad=True),
           "ds": [torch.randn(2, 1, 16, 16, requires_grad=True), torch.randn(2, 1, 8, 8, requires_grad=True)]}
    y, t, d = torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64)
    inst = torch.randint(0, 3, (2, 32, 32))
    loss, parts = supervised_loss(out, y, t, d, inst, 2, LossWeights(blob_size=32, w_sr=0.5))
    loss.backward()
    assert set(parts) >= {"bce", "region", "blob", "sr", "skel", "core", "ds"}
    assert all(np.isfinite(v) for v in parts.values())


# ----------------------------------------------------------------------------------------------- masks
def _rand_masks(rng, n):
    out = []
    for _ in range(n):
        full = np.zeros((2048, 2048), bool)
        y, x = rng.integers(0, 1900, 2)
        h, w = rng.integers(5, 140, 2)
        full[y:y + h, x:x + w] = rng.random((h, w)) > 0.3
        out.append(full)
    return out


def test_cropmask_iou_and_rle_match_pycocotools():
    rng = np.random.default_rng(0)
    fulls = _rand_masks(rng, 6)
    fulls[1] = fulls[0] | np.roll(fulls[0], 7, axis=1)   # guaranteed overlap
    cms = [CropMask.from_full(f) for f in fulls]
    rles = [c.to_rle() for c in cms]
    for c, r, f in zip(cms, rles, fulls):
        assert (CropMask.from_rle(r).full() == f).all()
        assert (CropMask.unpack(c.packed()).full() == f).all()
    ref = iou_matrix(rles, rles)
    assert np.allclose(iou_matrix_crops(cms, cms), ref)
    assert abs(iou(cms[0], cms[1]) - ref[0, 1]) < 1e-12
    st_a, st_b = pq_stats_crops(cms[:3], cms[2:]), pq_stats(rles[:3], rles[2:])
    assert (st_a.tp, st_a.fp, st_a.fn) == (st_b.tp, st_b.fp, st_b.fn) and abs(st_a.tp_iou_sum - st_b.tp_iou_sum) < 1e-9


def test_greedy_select_is_disjoint_and_respects_tau():
    rng = np.random.default_rng(1)
    cms = [CropMask.from_full(f) for f in _rand_masks(rng, 12)]
    scores = rng.random(len(cms))
    sel, idx = greedy_select(cms, scores, 0.3, rho_ov=0.2, min_area=10)
    occ = np.zeros((2048, 2048), int)
    for m in sel:
        occ[m.y0:m.y1, m.x0:m.x1] += m.m
    assert occ.max() <= 1
    assert all(scores[i] > 0.3 for i in idx)


# ----------------------------------------------------------------------------------------------- taxonomy
def test_taxonomy_fragmented():
    from filseg.taxonomy_v3 import classify_set
    g = np.zeros((2048, 2048), bool)
    g[100:110, 100:400] = True
    a, b = np.zeros_like(g), np.zeros_like(g)
    a[100:110, 100:240] = True
    b[100:110, 260:400] = True
    gts, preds = classify_set([CropMask.from_full(g)], [CropMask.from_full(a), CropMask.from_full(b)])
    assert gts[0]["cls"] == "fragmented"
    assert [p["cls"] for p in preds] == ["fragment", "fragment"]


# ----------------------------------------------------------------------------------------------- pool / linker
def _synthetic_prob():
    p = np.zeros((2048, 2048), np.float32)
    p[500:508, 300:600] = 0.9      # one filament broken by a 14 px gap
    p[500:508, 614:800] = 0.85
    p[1200:1206, 900:1000] = 0.65  # a faint short one
    p[1500:1510, 1500:1530] = 0.95
    return p


def test_pool_and_linker_on_synthetic_map():
    from filseg.linker_v3 import FEATURES, edge_features, edges, group, nodes
    from filseg.pool_v3 import build_pool, feature_columns
    p = _synthetic_prob()
    image = np.full((2048, 2048), 120, np.uint8)
    yy, xx = np.ogrid[:2048, :2048]
    image[(yy - 1024) ** 2 + (xx - 1024) ** 2 > 860 ** 2] = 0
    masks, recs = build_pool(p, image, members=[p, p * 0.9], min_area=50)
    assert masks and len(masks) == len(recs)
    cols = feature_columns(["H1", "H2", "H3", "H4", "H5", "H6"], [])
    assert all(c in recs[0] for c in cols)
    ious = iou_matrix_crops(masks, masks)
    np.fill_diagonal(ious, 0)
    assert ious.max() <= 0.95 + 1e-9, "duplicates survived"
    lab, n, slices = nodes(p, 0.6, 20)
    el = edges(lab, n, slices, 48)
    gaps = {(i, j): g for i, j, g, *_ in el}
    assert len(el) >= 1 and min(gaps.values()) <= 16
    x = edge_features(p, None, None, image, lab, n, slices, el, (1024.0, 1024.0, 860.0))
    assert x.shape == (len(el), len(FEATURES))
    glab, g, stats = group(lab, n, el, np.ones(len(el)), 0.5)
    assert g < n


# ----------------------------------------------------------------------------------------------- models
def test_models_forward_cpu():
    from filseg.models_v3 import ModelConfig, build_model_v3
    cfgs = [ModelConfig(model="smp_unet", backbone="resnet34", size=128, pretrained=False),
            ModelConfig(model="smp_unet", backbone="resnet34", size=128, hr_stem=True, edge_inputs=True,
                        pretrained=False),
            ModelConfig(model="cnx_unet", backbone="convnext_tiny", size=128, hr_stem=True, pretrained=False,
                        ckpt_stages=[1], ckpt_hr=True),
            ModelConfig(model="vit_dpt", backbone="vit_small_patch16_dinov3", size=128, hr_stem=True,
                        vit_taps=[2, 5, 8, 11], pretrained=False)]
    for c in cfgs:
        m = build_model_v3(c).train()
        x_in = torch.randn(1, 1, 128, 128)
        x_full = torch.randn(1, 1, 256, 256) if c.hr_stem else None
        out = m(x_in, x_full)
        exp = 256 if c.hr_stem else 128
        assert out["p"].shape == (1, 1, exp, exp), (c.model, out["p"].shape)
        assert out["s"].shape == out["d"].shape == out["p"].shape
        assert len(out["ds"]) == 2
        sum(v.mean() for k, v in out.items() if k != "ds").backward()
        groups = m.param_groups(5e-4, 1e-4, 0.85, 0.05)
        assert sum(len(g["params"]) for g in groups) == sum(1 for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------------- leakage
def test_leakage_folds_and_unlabeled():
    import pickle
    from datetime import datetime
    work = os.environ.get("FILSEG_WORK", "/root/kaggle/work")
    meta_path = os.path.join(work, "cache", "train_meta.pkl")
    if not os.path.exists(meta_path):
        print("  (skipped: no cache)")
        return
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    assert len(set(meta["stems"])) == len(meta["stems"]), "an image appears twice (its sets would span folds)"
    assert sorted(np.bincount(meta["folds"]).tolist()) == [141, 141, 141, 142, 142]
    u_meta = os.path.join(work, "unlabeled", "U_meta.csv")
    if os.path.exists(u_meta):
        import pandas as pd
        from filseg.data import list_test_stems
        u = pd.read_csv(u_meta)["stem"]
        comp = set(meta["stems"]) | set(list_test_stems())
        assert not set(u) & comp, "an unlabelled frame is a competition image"
        ts = lambda s: datetime.strptime(s[:14], "%Y%m%d%H%M%S").timestamp()  # noqa: E731
        test_t = np.array(sorted(ts(s) for s in list_test_stems()))
        ut = np.array([ts(s) for s in u])
        pos = np.searchsorted(test_t, ut).clip(1, len(test_t) - 1)
        gap = np.minimum(np.abs(ut - test_t[pos - 1]), np.abs(ut - test_t[pos]))
        assert gap.min() > 72 * 3600, "an unlabelled frame is within 72 h of a test image"


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if failed else 0)
