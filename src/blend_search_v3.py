"""v3: search blend weights for any set of members (CPU), as v2's weight comparison but with a generated grid.

  python src/blend_search_v3.py --names r34_2048 effv2s_1536 m3_r34_1536_v3 m1_cnxl_2048_v3 m2_vitl_1536_v3 \
      --heavy m1_cnxl_2048_v3 m2_vitl_1536_v3 --out logs/final_v3_blend_<tag>.txt

Each weight vector is scored like v2's comparison: v1 extraction (t_seed 0.6, t_mask 0.6, link 8, holes filled)
on the blended out-of-fold maps, best area / mean-probability filter, exact additive PQ (tune_postprocess.py).
Grid: all ones; --heavy members at 2; each member alone; leave-one-out; then +-1 steps around the best so far,
up to --max-evals vectors. The last line of the output is "BEST <weights>" (one weight per --names entry).
"""
from __future__ import annotations

import argparse
import os

from tune_postprocess import collect, search_filters

STRUCT = [(0.6, 0.6, 8, -1)]


def score(names, weights, workers) -> float:
    use = [(n, w) for n, w in zip(names, weights) if w > 0]
    df, n_gt = collect([n for n, _ in use], [w for _, w in use], STRUCT, workers=workers)
    return float(search_filters(df, n_gt).iloc[0]["pq"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--heavy", nargs="*", default=[], help="members that start with weight 2")
    ap.add_argument("--start", type=float, nargs="*", default=None, help="extra starting vector (e.g. a previous best)")
    ap.add_argument("--max-evals", type=int, default=24)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = len(args.names)
    grid = [tuple([1.0] * n)]
    if args.heavy:
        grid.append(tuple(2.0 if m in args.heavy else 1.0 for m in args.names))
    if args.start:
        assert len(args.start) == n
        grid.append(tuple(float(w) for w in args.start))
    if n > 1:
        grid += [tuple(1.0 if j == i else 0.0 for j in range(n)) for i in range(n)]
        if n > 2:
            grid += [tuple(0.0 if j == i else 1.0 for j in range(n)) for i in range(n)]
    seen, results = set(), []
    log = open(args.out, "a")

    def run(w):
        if w in seen or sum(w) == 0 or len(results) >= args.max_evals:
            return
        seen.add(w)
        pq = score(args.names, w, args.workers)
        results.append((pq, w))
        line = f"weights {' '.join(f'{x:g}' for x in w)}: PQ {pq:.4f}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    log.write(f"== blend search over {' '.join(args.names)}\n")
    for w in grid:
        run(w)
    improved = True
    while improved and len(results) < args.max_evals:
        improved = False
        best_pq, best_w = max(results)
        for i in range(n):
            for d in (1.0, -1.0):
                w = list(best_w)
                w[i] = max(0.0, w[i] + d)
                run(tuple(w))
        if max(results)[0] > best_pq:
            improved = True
    best_pq, best_w = max(results)
    line = f"BEST {' '.join(f'{x:g}' for x in best_w)}"
    log.write(f"best PQ {best_pq:.4f}\n{line}\n")
    log.close()
    print(f"best PQ {best_pq:.4f}")
    print(line)


if __name__ == "__main__":
    main()
