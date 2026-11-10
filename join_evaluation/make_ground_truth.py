#!/usr/bin/env python3
"""Build ground_truth/<query>.csv (one per LIMIT sweep, shared by its variants) (columns a,b: the two ids of a true pair) for every query.

    same_sentiment / opposite_sentiment   from the review labels (review.scoreSentiment) in the SemBench DB.
        This is complete. SemBench's own Q7 ground truth stops at 5,000 rows because of the query's LIMIT
        (plan Sec. 7 item 10), so it is not used.
    pc_compatible                          converted from evaluation/results/pc_parts/ground_truth/q5.csv
                                           (made by evaluation/make_pc_parts_ground_truth.py)
    same_movie                             the plot and the tagline come from the same movie (sm_tagline)

Pairs are unordered when scored, so the order of a and b doesn't matter.
Usage: python3 join_evaluation/make_ground_truth.py
"""

import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DB_DIR, GT_DIR, IPDB_BIN, REPO, load_queries  # noqa: E402

MOVIE = "ant_man_and_the_wasp_quantumania"
REVIEW_PAIRS = """
SELECT r1.reviewId AS a, r2.reviewId AS b
FROM review r1 JOIN review r2 ON r1.id = r2.id AND r1.reviewId <> r2.reviewId
WHERE r1.id = '{movie}' AND r1.scoreSentiment {op} r2.scoreSentiment
"""
PC_GT = REPO / "evaluation" / "results" / "pc_parts" / "ground_truth" / "q5.csv"


def ipdb_copy(db: Path, select: str, out: Path):
    sql = f"COPY ({select}) TO '{out}' (HEADER, DELIMITER ',');\n"
    proc = subprocess.run([str(IPDB_BIN), "-readonly", str(db)], input=sql, text=True, capture_output=True)
    if proc.returncode != 0 or "Error" in proc.stdout + proc.stderr:
        sys.exit(f"Failed to build {out.name}:\n{proc.stdout}{proc.stderr}")


def same_movie_select(query_sql: str) -> str:
    """sm_tagline itself, with the LLM predicate replaced by plot.movie_id = tagline.movie_id."""
    select = "SELECT p.movie_id AS plot_movie_id, t.movie_id AS tagline_movie_id"
    if select not in query_sql or "\nWHERE LLM" not in query_sql:
        sys.exit("sm_tagline no longer has the expected shape; update same_movie_select()")
    body = query_sql.replace(select, "SELECT p.movie_id AS a, t.movie_id AS b")
    return body.split("\nWHERE LLM")[0] + "\nWHERE p.movie_id = t.movie_id"


def main():
    GT_DIR.mkdir(exist_ok=True)
    done = set()
    for q in load_queries(include_optional=True):
        if q.gt_id in done:  # the variants of a LIMIT sweep share one file
            continue
        done.add(q.gt_id)
        out = GT_DIR / f"{q.gt_id}.csv"
        kind = q.ground_truth
        if kind in ("same_sentiment", "opposite_sentiment"):
            op = "=" if kind == "same_sentiment" else "<>"
            ipdb_copy(DB_DIR / q.db, REVIEW_PAIRS.format(movie=MOVIE, op=op), out)
        elif kind == "same_movie":
            ipdb_copy(DB_DIR / q.db, same_movie_select(q.sql), out)
        elif kind == "pc_compatible":
            if not PC_GT.exists():
                sys.exit(f"Missing {PC_GT}; create it with evaluation/make_pc_parts_ground_truth.py")
            with open(PC_GT) as f, open(out, "w", newline="") as g:
                w = csv.writer(g)
                w.writerow(["a", "b"])
                for row in csv.DictReader(f):
                    w.writerow([row["cpu_id"], row["motherboard_id"]])
        else:
            sys.exit(f"{q.id}: unknown ground_truth '{kind}'")
        with open(out) as f:
            pairs = {frozenset(r) for r in list(csv.reader(f))[1:]}
        print(f"{q.gt_id:<15} {kind:<19} {len(pairs):>6} unordered true pairs -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
