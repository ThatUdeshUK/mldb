"""Shared definitions for the join-decomposition evaluation (stdlib only)."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
QUERY_DIR = HERE / "queries"
GT_DIR = HERE / "ground_truth"
RESULTS_DIR = HERE / "results"
ANALYSIS_DIR = HERE / "analysis"

# The iPDB databases live at the repo root (built by evaluation/setup_ipdb_icde.sh).
DB_DIR = Path(os.environ.get("JOIN_EVAL_DB_DIR", REPO))
IPDB_BIN = Path(os.environ.get("IPDB_BIN", REPO.parent / "iPDb" / "build" / "release_api" / "ipdb"))

# Arms of the comparison (plan E5). Each arm is the list of session settings applied before the query.
#   pairwise   - never decompose: the reference plan (one LLM judgment per pair, with base optimizations)
#   decomposed - always decompose: no cost check (T1), no runtime check (T2)
#   auto       - iPDB's default decision: T1 cost check at plan time + T2 runtime check with fallback
#   guarded    - always passes T1 (min_speedup ~ 0) but keeps the T2 runtime check, so every query reports the
#                T2 signals and decision (Step 0: calibrating the runtime check)
#   manual     - for hand-decomposed queries without an LLM join (settings are irrelevant there)
ARMS = {
    "pairwise": ["SET join_decomposition = 'never';"],
    "decomposed": ["SET join_decomposition = 'always';"],
    "auto": ["SET join_decomposition = 'auto';"],
    "guarded": ["SET join_decomposition = 'auto';", "SET join_decomposition_min_speedup = 1e-9;"],
    "manual": ["SET join_decomposition = 'never';"],
}
DEFAULT_ARMS = ["pairwise", "decomposed", "auto", "guarded"]

STEM_RE = re.compile(r"^(?P<query>[^|]+)\|arm:(?P<arm>[^|]+)\|model:(?P<model>[^|]+)\|run:(?P<run>\d+)$")


@dataclass
class Query:
    id: str
    path: Path
    sql: str                      # body with the <model> placeholder, no trailing semicolon
    db: str
    limit: int | None             # LIMIT k, or None
    pair_columns: tuple[int, int]  # CSV column positions of the two ids of a result pair
    ground_truth: str
    arms: list[str] = field(default_factory=lambda: list(DEFAULT_ARMS))
    settings: list[str] = field(default_factory=list)  # per-query session settings, applied in every arm
    meta: dict = field(default_factory=dict)

    def db_path(self) -> Path:
        return DB_DIR / self.db

    def render(self, model: str) -> str:
        return self.sql.replace("<model>", model)


def load_query(path: Path) -> Query:
    meta, body = {}, []
    for line in path.read_text().splitlines():
        m = re.match(r"^\s*--\s*([a-z_]+):\s*(.*)$", line)
        if m and not body:
            meta[m.group(1)] = m.group(2).strip()
        elif not line.strip().startswith("--"):
            body.append(line)
    sql = "\n".join(body).strip().rstrip(";").strip()
    limit = meta.get("limit", "none")
    a, b = (int(x) for x in meta["pair_columns"].split(","))
    arms = [x.strip() for x in meta["arms"].split(",")] if "arms" in meta else list(DEFAULT_ARMS)
    settings = [s.strip() + ";" for s in meta.get("settings", "").split(";") if s.strip()]
    return Query(id=meta["id"], path=path, sql=sql, db=meta["db"],
                 limit=None if limit == "none" else int(limit), pair_columns=(a, b),
                 ground_truth=meta["ground_truth"], arms=arms, settings=settings, meta=meta)


def load_queries(ids: list[str] | None = None) -> list[Query]:
    queries = [load_query(p) for p in sorted(QUERY_DIR.glob("*.sql"))]
    if ids:
        unknown = set(ids) - {q.id for q in queries}
        if unknown:
            raise SystemExit(f"Unknown queries: {sorted(unknown)}. Known: {[q.id for q in queries]}")
        queries = [q for q in queries if q.id in ids]
    return queries


def stem(query: str, arm: str, model: str, run: int) -> str:
    return f"{query}|arm:{arm}|model:{model}|run:{run}"
