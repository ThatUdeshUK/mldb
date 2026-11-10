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
    family: str = ""              # the query file's id; a LIMIT sweep expands one file into <family>_k<k>
    optional: bool = False        # left out of the default set; run only when named in --queries
    true_cardinality: tuple[float, float, float] | None = None  # (pairs P, left rows d_R, right rows d_S)

    @property
    def gt_id(self) -> str:
        """Ground truth file stem: the variants of a LIMIT sweep share their family's ground truth."""
        return self.family or self.id

    def db_path(self) -> Path:
        return DB_DIR / self.db

    def render(self, model: str) -> str:
        return self.sql.replace("<model>", model)


def load_query(path: Path) -> list[Query]:
    """One query, or one per k if the file declares `-- limits: k1, k2, ...` (the SQL then says `LIMIT <limit>`
    and the variants are named <id>_k<k>)."""
    meta, body = {}, []
    for line in path.read_text().splitlines():
        m = re.match(r"^\s*--\s*([a-z_]+):\s*(.*)$", line)
        if m and not body:
            meta[m.group(1)] = m.group(2).strip()
        elif not line.strip().startswith("--"):
            body.append(line)
    sql = "\n".join(body).strip().rstrip(";").strip()
    a, b = (int(x) for x in meta["pair_columns"].split(","))
    common = dict(path=path, db=meta["db"], pair_columns=(a, b), ground_truth=meta["ground_truth"],
                  arms=[x.strip() for x in meta["arms"].split(",")] if "arms" in meta else list(DEFAULT_ARMS),
                  settings=[s.strip() + ";" for s in meta.get("settings", "").split(";") if s.strip()],
                  meta=meta, optional=(meta.get("optional", "no").split() or ["no"])[0].lower() in ("yes", "true", "1"),
                  true_cardinality=tuple(float(x) for x in meta["true_cardinality"].split(","))
                  if "true_cardinality" in meta else None)
    if "limits" in meta:
        if "<limit>" not in sql:
            raise SystemExit(f"{path.name}: declares limits but its SQL has no LIMIT <limit>")
        return [Query(id=f"{meta['id']}_k{k}", sql=sql.replace("<limit>", str(k)), limit=k, family=meta["id"],
                      **common)
                for k in (int(x) for x in meta["limits"].split(","))]
    limit = meta.get("limit", "none")
    return [Query(id=meta["id"], sql=sql, limit=None if limit == "none" else int(limit), **common)]


def load_queries(ids: list[str] | None = None, include_optional: bool = False) -> list[Query]:
    """All queries (optional ones only with include_optional), or those named in ids. A sweep's family id (e.g.
    sb_q5) selects all of its variants."""
    queries = [q for p in sorted(QUERY_DIR.glob("*.sql")) for q in load_query(p)]
    if ids:
        known = {q.id for q in queries} | {q.family for q in queries if q.family}
        unknown = set(ids) - known
        if unknown:
            raise SystemExit(f"Unknown queries: {sorted(unknown)}. Known: {sorted(known)}")
        return [q for q in queries if q.id in ids or q.family in ids]
    return [q for q in queries if include_optional or not q.optional]


def stem(query: str, arm: str, model: str, run: int) -> str:
    return f"{query}|arm:{arm}|model:{model}|run:{run}"
