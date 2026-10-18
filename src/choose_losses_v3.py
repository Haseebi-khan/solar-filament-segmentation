"""v3: turn the proxy ablations (scripts/ablate_v3.sh) into the loss flags of the real members.

  python src/choose_losses_v3.py [--dir work/ablations_v3]

Every decision uses the paired bootstrap gate of compare_v3.py (mean delta >= +0.003 and P(delta > 0) >= 0.90):
  region   start from JML1 (correct for soft labels, equal to soft Jaccard on hard ones); DML1 replaces it if it
           passes the gate against JML1; v1's soft Dice only if it passes the gate against the choice
  blob     on only if abl_jml_blob passes the gate against abl_jml
  heads    S/D heads + deep supervision stay on unless abl_full is significantly worse than abl_jml_blob
           (gate reversed): S and D also feed the Stage C features and the linker, which the proxy score
           does not measure (DECISIONS.md)
  sr       skeleton recall on only if abl_full_sr passes the gate against abl_full
Missing ablations keep the blueprint default. The last output line holds the train_v3.py flags.
"""
from __future__ import annotations

import argparse
import os
import pickle

from compare_v3 import bootstrap, per_image_stats
from filseg.data import CACHE_DIR, WORK_ROOT

_CACHE = {}


def stats(path, meta):
    if path not in _CACHE:
        _CACHE[path] = per_image_stats(path, meta)
    return _CACHE[path]


def gate(a: str, b: str, d: str, meta, log) -> bool | None:
    """True if b passes the gate against a; None if a CSV is missing."""
    pa, pb = os.path.join(d, a + ".csv"), os.path.join(d, b + ".csv")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        log(f"{b} vs {a}: missing, keeping the default")
        return None
    r = bootstrap(stats(pa, meta), stats(pb, meta))
    ok = r["delta_mean"] >= 0.003 and r["p_gt0"] >= 0.90
    log(f"{b} vs {a}: PQ {r['pq_b']:.4f} vs {r['pq_a']:.4f}, delta {r['delta_mean']:+.4f} "
        f"[{r['ci90'][0]:+.4f}, {r['ci90'][1]:+.4f}], P(>0) {r['p_gt0']:.2f} -> {'pass' if ok else 'no'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(WORK_ROOT, "ablations_v3"))
    ap.add_argument("--out", default="logs/final_v3_ablations.txt")
    args = ap.parse_args()
    with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log("== loss choice from the proxy ablations")
    region, region_run = "jml", "abl_jml"
    if gate("abl_jml", "abl_dml", args.dir, meta, log):
        region, region_run = "dml", "abl_dml"
    if gate(region_run, "abl_dice", args.dir, meta, log):
        region = "dice"
    blob = bool(gate("abl_jml", "abl_jml_blob", args.dir, meta, log))
    worse = gate("abl_full", "abl_jml_blob", args.dir, meta, log)   # reversed: is "without heads" better?
    heads = not worse
    sr = bool(gate("abl_full", "abl_full_sr", args.dir, meta, log))
    flags = (f"--region {region} --w-blob {0.5 if blob else 0} --w-sr {0.5 if sr else 0} "
             + ("--aux-heads --deep-sup" if heads else "--no-aux-heads --no-deep-sup"))
    log(f"chosen: region {region}, blob {'on' if blob else 'off'}, S/D heads + deep supervision "
        f"{'on' if heads else 'off'}, skeleton recall {'on' if sr else 'off'}")
    with open(args.out, "a") as f:
        f.write("\n".join(lines) + f"\nFLAGS {flags}\n")
    print(f"FLAGS {flags}")


if __name__ == "__main__":
    main()
