#!/usr/bin/env python3
"""Build the movie sample for the merging evaluation from the semantic movies benchmark (benchmark/*.csv).

One row per movie: id, title, plot (the inputs the operators read) and gt_* label columns, all from IMDB's
movie_info keyed by movie id (benchmark/movie.csv has several conflicting genre/lang rows per id, so only its
title is used):

    gt_comedy    'Comedy' is among the movie's genres (info type 'genres')
    gt_language  first listed language (info type 'languages')
    gt_english   gt_language == 'English'
    gt_rating    first US MPAA certificate (info type 'certificates', USA:...), TV ratings mapped to MPAA
    gt_children  gt_rating in (G, PG)

Pools: 'rated' (default) only samples movies with a US rating, so every row has all labels; 'all' samples from
every movie with a plot and leaves gt_rating / gt_children empty where unknown.

Usage: python3 sem_merge_evaluation/make_dataset.py [--n 500] [--seed 42] [--pool rated|all]
"""

import argparse
import csv
import random
import sys

from common import BENCHMARK_DIR, DATA_DIR, DEFAULT_N, DEFAULT_POOL, DEFAULT_SEED, dataset_name, dataset_path

GENRES, LANGUAGES, CERTIFICATES, PLOT = 3, 4, 5, 98
# USA certificate -> MPAA rating. Unrated / Not Rated / Approved / M are left unknown.
US_RATING = {"G": "G", "PG": "PG", "PG-13": "PG-13", "R": "R", "NC-17": "NC-17", "X": "NC-17",
             "TV-Y": "G", "TV-Y7": "G", "TV-G": "G", "TV-PG": "PG", "TV-14": "PG-13", "TV-MA": "R"}


def build(pool: str) -> list[dict]:
    if not (BENCHMARK_DIR / "movie_info.csv").exists():
        raise SystemExit(f"{BENCHMARK_DIR}/movie_info.csv not found: point SEM_MERGE_BENCHMARK_DIR at the semantic "
                         "movies benchmark directory (movie.csv, movie_info.csv)")
    csv.field_size_limit(sys.maxsize)
    titles = {}
    with open(BENCHMARK_DIR / "movie.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            titles.setdefault(int(r["id"]), r["title"])

    # (movie_info.id, info) per movie and type; the lowest movie_info id is "first listed".
    info: dict[tuple[int, int], list[tuple[int, str]]] = {}
    with open(BENCHMARK_DIR / "movie_info.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = int(r["info_type_id"])
            if t in (GENRES, LANGUAGES, CERTIFICATES, PLOT):
                info.setdefault((int(r["movie_id"]), t), []).append((int(r["id"]), r["info"]))

    def first(mid, t, pred=lambda v: True):
        vals = sorted(x for x in info.get((mid, t), []) if pred(x[1]))
        return vals[0][1] if vals else None

    rows = []
    for mid, title in sorted(titles.items()):
        plot = first(mid, PLOT)
        if not plot or not title:
            continue
        language = first(mid, LANGUAGES)
        cert = first(mid, CERTIFICATES, lambda v: v.startswith("USA:") and v[4:] in US_RATING)
        rating = US_RATING[cert[4:]] if cert else None
        if pool == "rated" and rating is None:
            continue
        rows.append({
            "id": mid, "title": title, "plot": plot.strip(),
            "gt_comedy": any(v == "Comedy" for _, v in info.get((mid, GENRES), [])),
            "gt_language": language or "",
            "gt_english": "" if language is None else language == "English",
            "gt_rating": rating or "",
            "gt_children": "" if rating is None else rating in ("G", "PG"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="sample size (default %(default)s)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--pool", choices=["rated", "all"], default=DEFAULT_POOL)
    args = ap.parse_args()

    rows = build(args.pool)
    if args.n > len(rows):
        raise SystemExit(f"--n {args.n} exceeds the {len(rows)} movies in pool '{args.pool}'"
                         + (" (try --pool all)" if args.pool == "rated" else ""))
    sample = sorted(random.Random(args.seed).sample(rows, args.n), key=lambda r: r["id"])

    DATA_DIR.mkdir(exist_ok=True)
    out = dataset_path(dataset_name(args.n, args.seed, args.pool))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sample[0]))
        w.writeheader()
        w.writerows(sample)

    def rate(col):
        vals = [r[col] for r in sample if r[col] != ""]
        return f"{sum(v is True for v in vals)}/{len(vals)}"
    print(f"{out.relative_to(DATA_DIR.parent.parent)}: {len(sample)} movies (pool '{args.pool}', {len(rows)} available)")
    print(f"  comedy {rate('gt_comedy')}, english {rate('gt_english')}, children {rate('gt_children')}, "
          f"comedy & children {sum(r['gt_comedy'] is True and r['gt_children'] is True for r in sample)}, "
          f"comedy & english {sum(r['gt_comedy'] is True and r['gt_english'] is True for r in sample)}")


if __name__ == "__main__":
    main()
