"""Shared definitions for the semantic operator merging evaluation (stdlib only)."""

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BENCHMARK_DIR = REPO / "benchmark"
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
ANALYSIS_DIR = HERE / "analysis"
WSIAF_DIR = Path.home() / "Projects" / "WSIAF"

DEFAULT_N = 500
DEFAULT_SEED = 42
DEFAULT_POOL = "rated"


@dataclass(frozen=True)
class Op:
    """One semantic operator: a filter (boolean predicate) or a projection (one extracted value)."""
    name: str
    kind: str          # "filter" | "project"
    column: str        # input column the operator reads
    instruction: str   # natural-language instruction, as given to the WSIAF operator
    output: str        # attribute name of the output (used in merged / single-output projections)
    out_type: str      # "boolean" for filters, a type name for projections
    gt: str            # ground-truth column in the dataset


OPS = {
    "comedy": Op("comedy", "filter", "plot",
                 "Based on the plot, is the genre of this movie Comedy?",
                 "is_comedy", "boolean", "gt_comedy"),
    "children": Op("children", "filter", "plot",
                   "Based on the plot, is this movie good for children?",
                   "good_for_children", "boolean", "gt_children"),
    "english": Op("english", "filter", "title",
                  "Based on the title, is the language of this movie English?",
                  "is_english", "boolean", "gt_english"),
    "rating": Op("rating", "project", "plot",
                 "Based on the plot, estimate the movie's US MPAA rating. "
                 "Answer with exactly one of: G, PG, PG-13, R, NC-17.",
                 "mpaa_rating", "string", "gt_rating"),
    "language": Op("language", "project", "title",
                   "Based on the title, what is the original language of this movie? "
                   "Answer with the language name in English, e.g. English, Spanish, Japanese.",
                   "language", "string", "gt_language"),
}

# Execution defaults, matching the iPDB runs (evaluation/run.sh: ml_batch_size = 16, llm_no_threads = 16):
# 16 rows per LLM call (row marshalling) and 16 concurrent LLM requests.
DEFAULT_MODE = "batch"
DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = 16

# Two-operator combinations. The first operator is always a filter; ops[0] is applied before ops[1] in the
# "separate" strategy.
WORKLOADS = {
    "ff_same": ("filter|filter", "same column (plot)", ["comedy", "children"]),
    "ff_diff": ("filter|filter", "different columns (plot, title)", ["comedy", "english"]),
    "fp_same": ("filter|project", "same column (plot)", ["comedy", "rating"]),
    "fp_diff": ("filter|project", "different columns (plot, title)", ["comedy", "language"]),
}

# Execution strategies for a workload (op1, op2):
#   separate       - op1 on all rows, op2 on op1's survivors (the usual pipeline; for filter|project the
#                    projection only runs on rows that pass the filter)
#   separate_rev   - filter|filter only: op2 first, then op1 on its survivors
#   separate_1out  - control: like "separate", but each operator is a single-output semantic project (the same
#                    operator and prompt format as "merged"), so the merge effect is not confounded with the
#                    filter-vs-project prompt
#   merged         - one semantic project on all rows that extracts both outputs in one call per row
# Every unmerged strategy runs the second operator only on the rows the first filter keeps.
STRATEGIES = ["separate", "separate_rev", "separate_1out", "merged"]
FILTER_ONLY_STRATEGIES = {"separate_rev"}

STEM_RE = re.compile(r"^(?P<workload>[^|]+)\|strategy:(?P<strategy>[^|]+)\|model:(?P<model>[^|]+)"
                     r"\|mode:(?P<mode>[^|]+)\|data:(?P<data>[^|]+)\|run:(?P<run>\d+)$")


def strategies_for(workload: str) -> list[str]:
    kinds = {OPS[o].kind for o in WORKLOADS[workload][2]}
    return [s for s in STRATEGIES if not (s in FILTER_ONLY_STRATEGIES and kinds != {"filter"})]


def dataset_name(n: int, seed: int, pool: str) -> str:
    return f"movies_{pool}_n{n}_s{seed}"


def dataset_path(name: str) -> Path:
    return DATA_DIR / f"{name}.csv"


def model_key(model: str) -> str:
    """Filesystem- and price-friendly model name: 'openai/o4-mini' -> 'o4mini'."""
    return re.sub(r"[^a-z0-9]", "", model.split("/")[-1].lower())


def stem(workload: str, strategy: str, model: str, mode: str, data: str, run: int) -> str:
    return f"{workload}|strategy:{strategy}|model:{model_key(model)}|mode:{mode}|data:{data}|run:{run}"


def load_dataset(name: str) -> list[dict]:
    path = dataset_path(name)
    if not path.exists():
        raise SystemExit(f"{path} not found; build it with sem_merge_evaluation/make_dataset.py")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_prices() -> dict:
    return {k: v for k, v in json.loads((HERE / "prices.json").read_text()).items() if not k.startswith("_")}
