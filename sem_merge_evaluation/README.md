# Semantic operator merging evaluation

What do we gain when two semantic operators over the same rows are merged into **one** operator that extracts
both outputs in a single LLM call per row, instead of running them one after another?

The operators are WSIAF's (`~/Projects/WSIAF/src/operators/`): `semantic_filter_operator` and
`semantic_project_operator`. They are plain operators without iPDB's optimizer, so the measured difference
comes from the merge alone. A merged operator is one `semantic_project_operator` call whose output schema holds
both outputs (filters become boolean attributes, applied afterwards).

## Workload

Data: the semantic movies benchmark (`benchmark/movie.csv`, `benchmark/movie_info.csv`). There is one row per
movie: `id`, `title`, `plot`, plus labels taken from IMDB's `movie_info` (see `make_dataset.py`). Each operator
receives only the columns it reads.

| Operator | Kind | Reads | Instruction (abridged) | Label |
|---|---|---|---|---|
| `comedy` | filter | plot | is the genre Comedy? | `Comedy` among the genres |
| `children` | filter | plot | is this movie good for children? | US rating G or PG |
| `english` | filter | title | is the language English? | first listed language is English |
| `rating` | project | plot | estimate the US MPAA rating (G … NC-17) | first US certificate |
| `language` | project | title | the original language | first listed language |

| Workload | Combination | Columns | Operators |
|---|---|---|---|
| `ff_same` | filter \| filter | same (plot) | `comedy`, `children` |
| `ff_diff` | filter \| filter | different (plot, title) | `comedy`, `english` |
| `fp_same` | filter \| project | same (plot) | `comedy`, `rating` |
| `fp_diff` | filter \| project | different (plot, title) | `comedy`, `language` |

On the same column, merging sends the plot once instead of twice. On different columns, each separate call
already reads only its own column, so merging saves at most the per-call overhead (system prompt, instruction,
JSON framing).

## Strategies

| Strategy | Execution | Rows sent to the LLM (N rows, s = op1 selectivity) |
|---|---|---|
| `separate` | op1 on all rows, op2 on op1's survivors: the usual pipeline | N + sN |
| `separate_rev` | filter \| filter only: op2 first, then op1 on its survivors | N + s₂N |
| `separate_1out` | control: like `separate`, but each operator is a single-output `semantic_project` (the same operator and prompt format as `merged`) | N + sN |
| `merged` | one `semantic_project` on all rows extracting both outputs | N |

Without merging, the second operator never sees a row the first filter eliminated. With 16 rows per call
(below), the call count is ⌈N/16⌉ + ⌈sN/16⌉ for the unmerged strategies and ⌈N/16⌉ for `merged`.

`separate_1out` separates the effect of merging from the effect of the different filter and project prompts:
`merged` vs. `separate_1out` is the merge alone.

The trade-off being measured: merging removes the second operator's calls and repeated input, but it gives up
filter pushdown, because op2's output is computed for every row rather than only for op1's survivors. Merging
pays off when op1 keeps many rows or when the two operators read the same (long) column.

## Data

```bash
python3 sem_merge_evaluation/make_dataset.py                  # 500 movies (default), seed 42
python3 sem_merge_evaluation/make_dataset.py --n 200 --seed 1
python3 sem_merge_evaluation/make_dataset.py --n 2000 --pool all
```

- `--pool rated` (default) samples only movies that have a US rating, so every row has every label; 726 movies
  are available. `--pool all` samples from all 2,854 movies with a plot; `children` / `rating` labels are then
  missing for most rows, and those rows are left out of the quality metrics for those operators.
- Default sample (rated, n=500, seed 42): comedy 106, children 94, comedy ∧ children 34, English 487/493,
  comedy ∧ English 103.
- `run.py` builds a missing dataset automatically. Files go to `data/movies_<pool>_n<N>_s<seed>.csv`
  (gitignored; the sample is deterministic for a given seed).

## Running

`run.py` runs in WSIAF's environment. The endpoint and key come from WSIAF's `.env` (`LLM_BASE_URL`,
`LLM_API_KEY`).

```bash
W="uv run --project $HOME/Projects/WSIAF python"
$W sem_merge_evaluation/run.py --dry-run                     # the plan and the exact instructions, no LLM calls
$W sem_merge_evaluation/run.py                               # all workloads x strategies, n=500, gpt-6-luna, run 1,
                                                             # 16 rows per call, 16 concurrent requests
$W sem_merge_evaluation/run.py --runs 1 2 3 --n 500
$W sem_merge_evaluation/run.py --workloads ff_same fp_same --strategies separate merged
$W sem_merge_evaluation/run.py --mode row                    # one call per row (WSIAF's MCP default)
python3 sem_merge_evaluation/analyze.py                      # -> sem_merge_evaluation/analysis/report.md
```

- `--model` (default `gpt-6-luna`, the paper's `gpt6luna`) takes any litellm model. WSIAF's `.env` still names
  `o4-mini`, which OpenAI has deprecated, so the `.env` model is not used.
- **Execution matches the iPDB runs** (`evaluation/run.sh`: `ml_batch_size = 16`, `llm_no_threads = 16`).
  `--mode batch` (default) marshals `--batch-size` rows (default 16) into each call (WSIAF `row_marshal=True`),
  with `--workers` requests in flight (default 16). `--mode row` makes one call per row; rows are still handed to
  the operator in groups of `--batch-size`, and `--workers` remains the total number of concurrent requests
  (litellm's own pool inside each group is limited to 1). Stages run one after another, so there are never more
  than `--workers` requests in flight. Latency depends on these settings, so keep them the same across the runs
  you compare.
- **Failures:** WSIAF's per-row `semantic_filter` silently keeps a row when its LLM call fails, its batched
  form silently drops rows when the model returns fewer results than rows, and `semantic_project` crashes or
  misaligns rows. `run.py` makes any failed call, and any batched response whose result count differs from the
  number of rows, raise and retries the batch (`--retries`, default
  2). A run that still fails is written to `*.failed.json` and listed at the top of the report. Retried
  attempts' tokens are not counted.
- Finished runs are skipped; `--force` reruns them.

## Outputs

`results/<workload>|strategy:<s>|model:<m>|mode:<row|batch>|data:<dataset>|run:<n>.json` holds the run
metadata, each stage (operator, rows in, calls, tokens, latency, retries) and every value computed per row.

`analysis/` (from `analyze.py`):

| File | Contents |
|---|---|
| `report.md` | Per workload: cost, latency and quality of every strategy, plus the merge-gain table |
| `runs.csv` | One row per run, including per-operator accuracy (`acc_<op>`, `n_<op>`) |
| `summary.csv` | Means per workload, strategy, model, mode and dataset |
| `gain.csv` | `merged` vs. `separate` and vs. the cheapest unmerged strategy: % change in calls, tokens, cost, latency; ΔF1 |

## Metrics and caveats

- **Quality:** P/R/F1 of the filter conjunction against the labels. For filter | project, also the projection
  accuracy on the returned rows (exact match after normalisation, e.g. `PG13` → `PG-13`, `English (US)` →
  `english`).
- **Agreement with `separate`:** the fraction of (row, operator) values computed by both strategies that are
  equal, and the Jaccard overlap of their result sets. Since labels are noisy (see below), use this to tell
  whether merging changes the answers.
- **Cost** is tokens × `prices.json`. Completion tokens include reasoning tokens.
- **Label noise:** the IMDB titles in this data are mostly English even for non-English films, so language
  from the title is hard and the label is often not recoverable from the input. English is also 98% of the
  rated pool, so `english` is not very selective, and `separate_rev` on `ff_diff` prunes almost nothing.
  "Good for children" is approximated by the US rating (G/PG). The rating label has the same caveat.
- Run several runs (`--runs 1 2 3`) for numbers that go into the paper; LLM outputs and latency vary between runs.
