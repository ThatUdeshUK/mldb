# Join decomposition evaluation

Evaluates iPDB's semantic join decomposition and the decision of when to apply it, following
`ipdb_paper/RESTRUCTURE_PLAN.md`:

| Plan item | What this measures |
|---|---|
| **E1**: decomposition across many joins, including ones that don't decompose | Every LLM join in the benchmarks, over a LIMIT sweep, plus a join that isn't attribute-based (`sm_tagline`) |
| **E5**: do T1–T3 choose well against always-on, always-off and an oracle? | Arms `pairwise` / `decomposed` / `auto`, the offline oracle, wrong choices, the cost of wrong choices and wasted fallback cost |
| **Step 0**: calibrate T2's runtime check | Arm `guarded` reports the four signals and the decision on every query, next to the quality decomposition actually loses |
| §7 item 10: SemBench Q7 ground truth truncated by `LIMIT 5000` | Ground truth is rebuilt from the review labels (complete) |

The paper section this feeds is `subsec:eval_decomp` (todos: "extend to other semantic joins", "evaluate the
decision procedure against always-pairwise").

## Workload

Only queries whose LLM predicate reads **both** sides of a join are decomposition candidates. pc_parts Q4 and
semantic_movies Q1/Q3 join relationally and apply the LLM to one side, so every arm would run the same plan;
they are left out. `pc_parts/ipdb/q5_split.sql` is identical to Q5.

| Query | Source | Pairs | LIMIT | Ground truth |
|---|---|---|---|---|
| `sb_q5_k{10,30,100,300,1000,5000}` | D1:Q5 (the paper's is k = 10) | 65,280 (256 × 255 review rows) | sweep | same sentiment (review labels) |
| `sb_q6` | D1:Q6 | 65,280 | 10 | opposite sentiment |
| `sb_q7_k{10,30,100,300,1000,5000}` | D1:Q7 (the paper's is k = 5000) | 65,280 | sweep | opposite sentiment |
| `sb_q5_nolimit` *(optional)* | D1:Q5 without LIMIT (`q5_nolimit_split.sql`) | 65,280 | – | same sentiment |
| `sb_q7_nolimit` *(optional)* | D1:Q7 without LIMIT (`q7_split.sql`) | 65,280 | – | opposite sentiment |
| `sb_q7_manual` | D1:Q7 decomposed by hand (`q7_alt.sql`), arm `manual` only | – | – | opposite sentiment |
| `pc_q5` | D2:Q5, CPU–motherboard compatibility | 41 × 41 | – | `evaluation/results/pc_parts/ground_truth/q5.csv` |
| `sm_tagline` | **New** (D3 data): which tagline belongs to which plot | 32 × 32 | – | same movie |

**LIMIT sweep.** `sb_q5.sql` and `sb_q7.sql` declare `-- limits: 10, 30, 100, 300, 1000, 5000` and expand into
one query per k (`sb_q5_k100`, ...); `--queries sb_q5` selects all of them, and they share `ground_truth/sb_q5.csv`.
The sweep moves the same join across T1's crossover (decompose iff `min(P, k/s) >= 2 x (10 + d_R + d_S)`):
with the true P = 65,280 and d_R = d_S = 256 that is k ≈ 105 at the default s = 0.1, and k ≈ 520 at the
predicate's true selectivity (≈ 0.5). The decomposed arm's cost is roughly flat in k (it extracts every review
once), the pairwise arm's grows with k, so the measured crossover is where they meet; `analyze.py` reports it next
to T1's decision. The no-LIMIT queries are too expensive to run by default: they are `-- optional: yes` and run
only when named (`--queries sb_q7_nolimit`).

`sm_tagline` is the "relationship is not attribute-based" failure mode of
paper §IV-B, with exact ground truth; it is new, so say so if it goes into the paper.

The SemBench movie has 256 review rows but only 128 distinct `reviewId`s (each review appears twice), so the
ground truth has 4,036 same-sentiment and 4,092 opposite-sentiment distinct pairs, matching SemBench's pair
definition (and the "4,092 distinct pairs" of the plan's §7 item 10).

`sm_tagline` is shaped so that iPDB recognises it. iPDB only decomposes an LLM predicate that is a FILTER
directly above the join (or above projections over it), and two DuckDB plan shapes hide that filter:
`JOIN ... ON LLM` over derived tables or CTEs keeps the predicate inside a nested-loop join condition, and a
subquery repeated on both sides becomes a materialized common subplan between the filter and the join. The query
therefore uses a comma join with the predicate in `WHERE`, literal movie ids, and (per query, in every arm)
`SET disabled_optimizers = 'common_subplan'`. These are limitations of the rewrite worth knowing about.

## Arms

| Arm | Settings | Meaning |
|---|---|---|
| `pairwise` | `join_decomposition = 'never'` | No decomposition: iPDB's LLM join (see below) |
| `decomposed` | `'always'` | Always decompose; no T1 cost check, no T2 runtime check |
| `auto` | `'auto'` (defaults) | iPDB's decision: T1 at plan time, then T2 with fallback |
| `guarded` | `'auto'`, `join_decomposition_min_speedup = 1e-9` | Always passes T1, keeps T2: the runtime check's signals for every query |
| `manual` | – | Hand-decomposed queries (no LLM join) |
| oracle | computed by `analyze.py` | Decompose iff decomposed quality ≥ pairwise − δ (default 0.05) and it is cheaper |

Every run is a fresh iPDB session with `threads = 1`, `ml_batch_size = 16`, `llm_no_threads = 16` (as in
`evaluation/run.sh`), JSON profiling on, and the database opened read-only.

**What `pairwise` executes.** In the current iPDB, a non-decomposed LLM join with a BOOLEAN output runs
`LlmApiPredictor::PredictJoin`: one call per input chunk (up to 2,048 pairs) that lists the chunk's unique left
and right items and asks the model for the matching index pairs. It is a set-based join prompt, not one
judgment per pair, so it is **not** the paper's reference result (§II-C, "one dedicated model call per tuple or
pair"); `ref F1` in the report means agreement with this arm. iPDB has no setting for per-pair join evaluation.
A large chunk can also fail: on `pc_q5` (41 × 41 items in one prompt) the call returned nothing after 224 s, iPDB
ignored the error and returned 0 rows. Such runs (LLM calls but no tokens) are marked `llm_failed` and left out.

**T1 and the SemBench joins.** `--explain` shows T1 estimating `pairwise 0 < 2x decomposed 32` for all five
SemBench joins, even without LIMIT: DuckDB estimates the self-join's cardinality as 0, so `auto` keeps them
pairwise at every k of the sweep. This is the plan's known caveat ("cardinality estimates ... make T1 err towards
pairwise"). The evaluation measures its effect, and the sweep table also recomputes T1 offline with the true
cardinalities (`-- true_cardinality: P, d_R, d_S` in the query file) to show what the cost model would decide
with correct estimates.

## Running

From the repo root, with the `*_icde.db` databases built (`evaluation/setup_ipdb_icde.sh`):

```bash
python3 join_evaluation/make_ground_truth.py               # once; no LLM calls
python3 join_evaluation/run.py --explain                   # T1's decision per query; <= 1 planning call each
python3 join_evaluation/run.py --queries pc_q5 sm_tagline  # cheap queries first
python3 join_evaluation/run.py --queries sb_q5 sb_q7 --arms pairwise decomposed auto  # the LIMIT sweep
python3 join_evaluation/run.py                             # everything: all queries x arms, run 1, gpt6luna
python3 join_evaluation/run.py --model sonnet55 --runs 1 2 3
python3 join_evaluation/analyze.py                         # -> join_evaluation/analysis/report.md
```

- **Cost:** the pairwise arm of the 65,280-pair SemBench joins is by far the most expensive part (the paper's
  pairwise D1:Q7 took 2,539 s). Start with `--queries pc_q5 sm_tagline`, and use `--arms` to leave arms out.
- **Resuming:** finished runs are skipped, so an interrupted run can simply be restarted; `--force` reruns.
- **Timeouts:** each run is killed after `--timeout` seconds (default 3600) and reported as such.
- **Step 0 sweeps:** `--set` changes a setting for all runs and `--tag` keeps those results apart, e.g.
  `python3 join_evaluation/run.py --arms guarded --set join_decomposition_min_overlap=0.5 --tag ov0.5`.
- **Other options:** `--dry-run` prints the session SQL; `--ipdb-bin`, `--threads`, `--batch-size`,
  `--llm-threads`. `IPDB_BIN` and `JOIN_EVAL_DB_DIR` override the binary and the database directory.

## Outputs

`results/<query>|arm:<arm>|model:<model>|run:<n>.{csv,json,log,meta.json}`: the result, the DuckDB profile,
iPDB's output (call counts, the planning call's rewrite, the runtime check) and run metadata.

`analysis/` (from `analyze.py`):

| File | Contents |
|---|---|
| `report.md` | All tables below, plus the task prompts and θ each planning call produced |
| `runs.csv` | One row per run: latency, LLM calls, tokens, cost, P/R/F1, reference F1, what was executed, T2 signals |
| `summary.csv` | Means per query, model and arm |
| `decisions.csv` | Oracle vs. `auto` and `guarded`: wrong runs, cost regret, quality regret |
| `wasted.csv` | Fallback runs: extra calls, cost and latency against running pairwise directly |
| `step0.csv` | Runtime-check signals and decision vs. whether decomposition really loses quality |
| `sweep.csv` | LIMIT sweep: pairwise vs. decomposed measured cost per k, the oracle, `auto`'s T1 note, and T1 recomputed with true cardinalities (configured s, and true s) |

## Metrics and caveats

- **Quality** is F1 against ground truth with unordered pairs, or **precision@k** for LIMIT queries, where any
  k true pairs are a correct answer. **Reference F1** compares a run with the pairwise arm's result (paper
  §II-C); it isn't defined for LIMIT queries.
- **What a run executed** is read from the log, not assumed from the arm: `decomposed`; `pairwise` (T1 declined,
  or the planning call gave no usable rewrite); `fallback` (decomposed, then T2 switched back).
- **Run status:** `ok`, `failed` (iPDB error), `timeout`, or `llm_failed` (LLM calls returned no tokens; see
  "What `pairwise` executes"). Only `ok` runs enter the means; the others are listed at the top of the report.
- **Cost** is tokens × `prices.json` (copied from `evaluation/analysis/compare_systems.py`). Add a model there
  to price it; otherwise the oracle compares LLM calls. iPDB does not report the planning call's tokens: the call
  is counted in `calls`, but its tokens are missing from tokens and cost.
- **The rewrite varies between runs**, because the planning call is an LLM call. Use several runs (`--runs 1 2 3`)
  for numbers that go into the paper; `report.md` lists every rewrite so the variation is visible.
- **The `guarded` arm** uses `min_speedup = 1e-9` because the setting must be positive; any query passes T1.
- **T2 only exists in `auto`**: `decomposed` (`'always'`) never falls back, which is the failure case T2 is meant
  to prevent.
