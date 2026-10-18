"""v3 Stage B: download unlabelled GONG H-alpha frames (external data, rule C3) for semi-supervised training.

  # primary pool: 2023-01-01 .. 2026-09-30, outside MAGFiLO's 2011-2022 span (cannot overlap any competition image)
  python src/gong_download_v3.py pool --start 2023-01-01 --end 2026-09-30 --out work/unlabeled
  # the archive frames of 30 training images, for the domain check (gong_domain_check_v3.py)
  python src/gong_download_v3.py train-stamps --n 30 --out work/unlabeled

Source: NSO GONG H-alpha 2048x2048 JPEG product ``hag`` (https://gong2.nso.edu/ftp/HA/hag/YYYYMM/YYYYMMDD/
<YYYYMMDDHHMMSS><site>h.jpg; same file naming, geometry and processing as the competition images; checked on
20110109104734Ch: same disk position, Pearson 0.974 on the disk, different JPEG compression).

Sampling: at most --per-day frames per day, at least --min-gap-hours apart, preferring different sites.
Frames within --exclude-hours of any competition test image are always skipped (relevant only for a 2010-2022
secondary pool; the per-fold validation exclusion happens at training time, see train_v3.py).

Polite and resumable: at most 4 concurrent connections, retries with exponential backoff, files written
atomically and skipped when already present and complete; manifest.csv records url, bytes and sha256.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import pickle
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

from filseg.data import CACHE_DIR, WORK_ROOT, list_test_stems

BASE = "https://gong2.nso.edu/ftp/HA/hag"
NAME = re.compile(r'href="(\d{14})([A-Z])h\.jpg"')
UA = {"User-Agent": "filseg-v3 research downloader (Solar Filament Segmentation Challenge 2026)"}


def fetch(url: str, attempts: int = 6, timeout: int = 60) -> bytes | None:
    """GET with exponential backoff; None on 404."""
    for k in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            err = e
        time.sleep(min(60, 2 ** k) + random.random())
    raise RuntimeError(f"giving up on {url}: {err}")


def jpeg_complete(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(2)
            f.seek(-2, os.SEEK_END)
            tail = f.read(2)
        return head == b"\xff\xd8" and tail == b"\xff\xd9"
    except OSError:
        return False


def stem_url(stem: str) -> str:
    return f"{BASE}/{stem[:6]}/{stem[:8]}/{stem}.jpg"


def stamp(stem: str) -> datetime:
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S")


def day_listing(day: date) -> list[str]:
    html = fetch(f"{BASE}/{day:%Y%m}/{day:%Y%m%d}/")
    if html is None:
        return []
    return sorted({f"{t}{s}h" for t, s in NAME.findall(html.decode("utf-8", "replace"))})


def pick(stems: list[str], per_day: int, gap_h: float, rng: random.Random) -> list[str]:
    """Up to per_day frames, pairwise >= gap_h apart, first pass one per site, second pass any site."""
    order = stems[:]
    rng.shuffle(order)
    chosen: list[str] = []
    for one_per_site in (True, False):
        for s in order:
            if len(chosen) >= per_day:
                break
            if s in chosen or (one_per_site and s[14] in {c[14] for c in chosen}):
                continue
            if all(abs((stamp(s) - stamp(c)).total_seconds()) >= gap_h * 3600 for c in chosen):
                chosen.append(s)
    return sorted(chosen)


def download(stems: list[str], raw_dir: str, manifest: str, workers: int = 4) -> None:
    done = set()
    if os.path.exists(manifest):
        with open(manifest) as f:
            done = {r["stem"] for r in csv.DictReader(f)}
    todo = [s for s in stems if not (s in done and jpeg_complete(os.path.join(raw_dir, s[:8], s + ".jpg")))]
    print(f"{len(stems)} frames requested, {len(stems) - len(todo)} already present, downloading {len(todo)}",
          flush=True)
    new = not os.path.exists(manifest)
    with open(manifest, "a", newline="") as mf:
        w = csv.writer(mf)
        if new:
            w.writerow(["stem", "url", "bytes", "sha256"])

        def one(s):
            url = stem_url(s)
            data = fetch(url)
            if data is None or data[:2] != b"\xff\xd8" or data[-2:] != b"\xff\xd9":
                return s, url, None
            d = os.path.join(raw_dir, s[:8])
            os.makedirs(d, exist_ok=True)
            tmp = os.path.join(d, s + ".jpg.part")
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, os.path.join(d, s + ".jpg"))
            return s, url, data

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=min(workers, 4)) as ex:
            for k, (s, url, data) in enumerate(ex.map(one, todo), start=1):
                if data is not None:
                    w.writerow([s, url, len(data), hashlib.sha256(data).hexdigest()])
                if k % 100 == 0:
                    mf.flush()
                    print(f"  {k}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p1 = sub.add_parser("pool")
    p1.add_argument("--start", default="2023-01-01")
    p1.add_argument("--end", default="2026-09-30")
    p1.add_argument("--per-day", type=int, default=4)
    p1.add_argument("--min-gap-hours", type=float, default=3.0)
    p1.add_argument("--exclude-hours", type=float, default=72.0)
    p1.add_argument("--seed", type=int, default=0)
    p2 = sub.add_parser("train-stamps")
    p2.add_argument("--n", type=int, default=30)
    p2.add_argument("--seed", type=int, default=0)
    for p in (p1, p2):
        p.add_argument("--out", default=os.environ.get("FILSEG_UNLABELED", os.path.join(WORK_ROOT, "unlabeled")))
        p.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.mode == "train-stamps":
        with open(os.path.join(CACHE_DIR, "train_meta.pkl"), "rb") as f:
            stems = pickle.load(f)["stems"]
        rng = random.Random(args.seed)
        chosen = sorted(rng.sample(stems, args.n))
        out = os.path.join(args.out, "domain")
        os.makedirs(out, exist_ok=True)
        download(chosen, os.path.join(out, "raw"), os.path.join(out, "manifest.csv"), args.workers)
        return

    test_t = sorted(stamp(s).timestamp() for s in list_test_stems())
    excl = args.exclude_hours * 3600
    os.makedirs(args.out, exist_ok=True)
    listing_path = os.path.join(args.out, "listing.csv")
    listed: dict[str, list[str]] = {}
    if os.path.exists(listing_path):
        with open(listing_path) as f:
            for r in csv.DictReader(f):
                listed.setdefault(r["day"], []).extend(r["stems"].split() if r["stems"] else [])
    d0, d1 = date.fromisoformat(args.start), date.fromisoformat(args.end)
    days = [d0 + timedelta(days=k) for k in range((d1 - d0).days + 1)]
    todo = [d for d in days if f"{d:%Y%m%d}" not in listed]
    print(f"{len(days)} days, {len(days) - len(todo)} listed before, listing {len(todo)}", flush=True)
    with open(listing_path, "a", newline="") as lf:
        w = csv.writer(lf)
        if os.path.getsize(listing_path) == 0:
            w.writerow(["day", "stems"])
        with ThreadPoolExecutor(max_workers=min(args.workers, 4)) as ex:
            for k, (d, stems) in enumerate(zip(todo, ex.map(day_listing, todo)), start=1):
                listed[f"{d:%Y%m%d}"] = stems
                w.writerow([f"{d:%Y%m%d}", " ".join(stems)])
                if k % 50 == 0:
                    lf.flush()
                    print(f"  listed {k}/{len(todo)} days", flush=True)
    rng = random.Random(args.seed)
    chosen = []
    for d in days:
        key = f"{d:%Y%m%d}"
        cands = []
        for s in listed.get(key, []):
            t = stamp(s).timestamp()
            if not any(abs(t - u) < excl for u in test_t):
                cands.append(s)
        chosen += pick(cands, args.per_day, args.min_gap_hours, rng)
    sites = {}
    for s in chosen:
        sites[s[14]] = sites.get(s[14], 0) + 1
    print(f"selected {len(chosen)} frames over {len(days)} days; per site {dict(sorted(sites.items()))}", flush=True)
    download(chosen, os.path.join(args.out, "raw"), os.path.join(args.out, "manifest.csv"), args.workers)
    print("done; next: python src/gong_domain_check_v3.py, then python src/gong_qc_v3.py")


if __name__ == "__main__":
    main()
