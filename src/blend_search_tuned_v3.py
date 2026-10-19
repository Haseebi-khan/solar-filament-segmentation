"""v3: blend_search_v3.py with the extraction thresholds tuned for every weight vector (CPU).

blend_search_v3.py scores each vector at t_seed = t_mask = 0.6, the optimum of the v1/v2 members (BCE + Dice). The
v3-loss members (soft targets + JML1) are calibrated lower: alone, M3 scores 0.413 at 0.6 but 0.430 at 0.4, M4 0.427
vs 0.441 (best single member). With the fixed grid the after_m4 blend dropped M4 (weights 1 1 1 0). Here each vector
gets the best of STRUCTS; the best (t_seed, t_mask) is printed under each line. Same CLI and output file format as
blend_search_v3.py; pass the printed t_seed to scripts/final_ensemble_v3.sh as T_SEED.

  python src/blend_search_tuned_v3.py --names r34_2048 effv2s_1536 m3_r34_1536_v3 m4_vitb_1536_v3 m5_cnxs_1536_v3 \
      --heavy m4_vitb_1536_v3 --out logs/final_v3_blend_tuned.txt
"""
from __future__ import annotations

import blend_search_v3
from tune_postprocess import collect, search_filters

STRUCTS = [(0.4, 0.4, 8, -1), (0.5, 0.4, 8, -1), (0.5, 0.5, 8, -1), (0.6, 0.6, 8, -1)]


def score(names, weights, workers) -> float:
    use = [(n, w) for n, w in zip(names, weights) if w > 0]
    df, n_gt = collect([n for n, _ in use], [w for _, w in use], STRUCTS, workers=workers)
    best = search_filters(df, n_gt).iloc[0]
    print(f"  best extraction: t_seed {best['t_seed']:g} t_mask {best['t_mask']:g} {best['score']} "
          f">= {best['min_score']:g}, area >= {best['min_area']}", flush=True)
    return float(best["pq"])


if __name__ == "__main__":
    blend_search_v3.score = score   # main() looks the scorer up in its module at call time
    blend_search_v3.main()
