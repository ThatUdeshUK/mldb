#!/usr/bin/env bash
# Run the evaluation benchmarks on iPDB (../../iPDb/build/release_api/ipdb).
#
# Reads   evaluation/queries/<benchmark>/ipdb/<query>.sql        (placeholder <model>)
#         evaluation/queries/<benchmark>/<benchmark>.dbname      (iPDB database with the LLM models registered)
# Writes  evaluation/<RESULTS_ROOT>/<benchmark>/ipdb/<query>|model:<MODEL>|run:<RUN>.csv    query output
#                                                       ...|run:<RUN>.json   DuckDB JSON profile (latency, llm_* counters)
#                                                       ...|run:<RUN>.log    stdout/stderr + wall time
#
# Models are catalog objects inside each iPDB database (CREATE LLM MODEL ...), so the API keys live in
# iPDB's persistent secrets, not here. Register models / build the databases with setup_ipdb_icde.sh.
#
# Every setting can be overridden from the environment:
#     bash evaluation/run.sh                                            # sembench_movies, q1..q10, runs 1 2 3
#     MODEL=gpt6luna RUNS="1" bash evaluation/run.sh
#     MODEL=sonnet55 BENCHMARK=all RUNS="1" bash evaluation/run.sh      # Claude Sonnet (register via setup_ipdb_icde.sh)
#     BENCHMARK=pc_parts QUERIES="q1 q4" bash evaluation/run.sh
#     QUERIES="q2_knn16 q2_knn32" bash evaluation/run.sh                # query variants by file name
#     BENCHMARK=all RUNS="1" bash evaluation/run.sh                     # every benchmark with an ipdb/ dir
#     DISABLED_OPTIMIZERS="limit_to_filter" bash evaluation/run.sh      # ablations
#     JOIN_DECOMPOSITION=never bash evaluation/run.sh                   # ablation: pairwise LLM joins
#
# LLM join decomposition is ENFORCED by default (SET join_decomposition = 'always'): every LLM join predicate
# that qualifies is rewritten into per-side extractions, with no cost-based skip and no runtime fallback to
# pairwise inference (both exist only in 'auto'). After each query the "[decomp: ...]" note from the profile
# is printed; a join that still ran pairwise (e.g. "no usable rewrite" when the planning call fails) is flagged.
set -uo pipefail

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
IPDB_DIR="${IPDB_DIR:-$EVAL_DIR/../../iPDb}"
IPDB_BIN="${IPDB_BIN:-$IPDB_DIR/build/release_api/ipdb}"

BENCHMARK="${BENCHMARK:-sembench_movies}"
MODEL="${MODEL:-gemini38flash}"   # a model name registered in the benchmark DB (see ipdb_models())
RUNS=(${RUNS:-1 2 3})
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-16}"   # SET ml_batch_size
LLM_NO_THREADS="${LLM_NO_THREADS:-16}"   # SET llm_no_threads (concurrent LLM requests)
THREADS="${THREADS:-1}"                  # SET threads (DuckDB execution threads)
DISABLED_OPTIMIZERS="${DISABLED_OPTIMIZERS:-}"   # e.g. limit_to_filter
JOIN_DECOMPOSITION="${JOIN_DECOMPOSITION:-always}"   # SET join_decomposition: always (enforced) | auto | never
RESULTS_ROOT="${RESULTS_ROOT:-results_icde}"
case "$JOIN_DECOMPOSITION" in
  always|auto|never) ;;
  *) echo "JOIN_DECOMPOSITION must be always, auto or never (got '$JOIN_DECOMPOSITION')" >&2; exit 1 ;;
esac
# Disabling the optimizer would silently override the enforced mode; use JOIN_DECOMPOSITION=never instead.
if [ "$JOIN_DECOMPOSITION" != never ] && [[ ",${DISABLED_OPTIMIZERS// /}," == *",join_decomposition,"* ]]; then
  echo "DISABLED_OPTIMIZERS contains join_decomposition, which overrides JOIN_DECOMPOSITION=$JOIN_DECOMPOSITION." >&2
  echo "For the pairwise ablation use JOIN_DECOMPOSITION=never instead." >&2; exit 1
fi
case "$RESULTS_ROOT" in /*) ;; *) RESULTS_ROOT="$EVAL_DIR/$RESULTS_ROOT" ;; esac

if [ "$BENCHMARK" = all ]; then
  for d in "$EVAL_DIR"/queries/*/ipdb; do
    BENCHMARK="$(basename "$(dirname "$d")")" bash "$0" || exit $?
  done
  exit 0
fi

QUERY_DIR="$EVAL_DIR/queries/$BENCHMARK/ipdb"
[ -d "$QUERY_DIR" ] || { echo "No iPDB queries for benchmark '$BENCHMARK' ($QUERY_DIR)" >&2; exit 1; }
if [ -n "${QUERIES:-}" ]; then
  QUERIES=($QUERIES)
else  # base queries only (q1.sql ... q10.sql); variants like q2_knn16 / q7_split run only when named
  QUERIES=($(ls "$QUERY_DIR" | grep -E '^q[0-9]+\.sql$' | sed 's/\.sql$//' | sort -V))
fi

# .dbname paths are written relative to the iPDb checkout (e.g. ../mldb/foo.db)
DB_PATH="$(<"$EVAL_DIR/queries/$BENCHMARK/$BENCHMARK.dbname")"
case "$DB_PATH" in /*) ;; *) DB_PATH="$(cd "$IPDB_DIR" && realpath -m "$DB_PATH")" ;; esac
OUT_DIR="$RESULTS_ROOT/$BENCHMARK/ipdb"
mkdir -p "$OUT_DIR"

for f in "$IPDB_BIN" "$DB_PATH"; do
  [ -e "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done

# Fail early if the DB can't be opened or the model isn't registered in it
MODELS="$("$IPDB_BIN" -readonly "$DB_PATH" -list -noheader -c "SELECT model_name FROM ipdb_models();" 2>&1)"
if [ $? -ne 0 ] || echo "$MODELS" | grep -q "Error"; then
  echo "Cannot open $DB_PATH with iPDB: $(echo "$MODELS" | head -1)" >&2
  echo "Rebuild it with: bash evaluation/setup_ipdb_icde.sh" >&2; exit 1
fi
if ! echo "$MODELS" | grep -qx "$MODEL"; then
  echo "Model '$MODEL' is not registered in $DB_PATH. Registered: $(echo $MODELS)" >&2
  echo "Register it with: bash evaluation/setup_ipdb_icde.sh" >&2; exit 1
fi

DISABLE_SQL=""
[ -n "$DISABLED_OPTIMIZERS" ] && DISABLE_SQL="SET disabled_optimizers = '${DISABLED_OPTIMIZERS}';"

echo "BENCHMARK: $BENCHMARK  MODEL: $MODEL  BATCH: $LLM_BATCH_SIZE  LLM_THREADS: $LLM_NO_THREADS  THREADS: $THREADS"
echo "DB: $DB_PATH (read-only)"
echo "OUT: $OUT_DIR"
echo "QUERIES: ${QUERIES[*]}  RUNS: ${RUNS[*]}  JOIN_DECOMPOSITION: $JOIN_DECOMPOSITION${DISABLED_OPTIMIZERS:+  DISABLED_OPTIMIZERS: $DISABLED_OPTIMIZERS}"

for RUN in "${RUNS[@]}"; do
  for QUERY in "${QUERIES[@]}"; do
    if [ ! -f "$QUERY_DIR/$QUERY.sql" ]; then
      echo "RUN -> $RUN, QUERY -> $QUERY  SKIPPED: no iPDB version ($QUERY_DIR/$QUERY.sql missing)"; continue
    fi
    STEM="$OUT_DIR/${QUERY}|model:${MODEL}|run:${RUN}"
    QUERY_CONTENT="$(grep -v '^[[:space:]]*--' "$QUERY_DIR/$QUERY.sql")"
    QUERY_CONTENT="${QUERY_CONTENT//;/}"
    QUERY_CONTENT="${QUERY_CONTENT//<model>/$MODEL}"
    echo "RUN -> $RUN, QUERY -> $QUERY"

    START=$(date +%s%N)
    (cd "$IPDB_DIR" && "$IPDB_BIN" -readonly "$DB_PATH") > "$STEM.log" 2>&1 <<QUERY_SCOPE
PRAGMA enable_profiling = 'json';
${DISABLE_SQL}
SET threads = ${THREADS};
SET ml_batch_size = ${LLM_BATCH_SIZE};
SET llm_no_threads = ${LLM_NO_THREADS};
SET join_decomposition = '${JOIN_DECOMPOSITION}';
PRAGMA profile_output = '${STEM}.json';
COPY (
${QUERY_CONTENT}
) TO '${STEM}.csv' (HEADER, DELIMITER ',');
.q
QUERY_SCOPE
    RC=$?
    WALL=$(awk -v s="$START" -v e="$(date +%s%N)" 'BEGIN{printf "%.3f", (e-s)/1e9}')
    echo "exit_code: $RC  wall_time_s: $WALL" >> "$STEM.log"
    if [ $RC -ne 0 ] || grep -q "Error" "$STEM.log"; then
      echo "  FAILED (exit $RC) after ${WALL}s, see $STEM.log"
      grep -m3 "Error" "$STEM.log" | sed 's/^/    /'
    else
      # iPDB prints "Total calls" per LLM operator (not for prompt-only table generation)
      CALLS=$(grep -o "Total calls: [0-9]*" "$STEM.log" | awk '{s+=$3; n++} END{if (n) printf ", %d LLM calls", s}')
      echo "  done in ${WALL}s, $(($(wc -l < "$STEM.csv") - 1)) rows${CALLS}"
      # The optimizer tags each LLM join predicate with "[decomp: <note>]" in the profile (queries without
      # an LLM join have none): "always" = decomposed; "..., no usable rewrite" = planning call failed, ran pairwise.
      DECOMP="$(grep -o 'decomp: [^]"]*' "$STEM.json" 2>/dev/null | sed 's/^decomp: //' | sort -u | paste -sd ';' -)"
      if [ -n "$DECOMP" ]; then
        echo "  join decomposition: $DECOMP"
        if [ "$JOIN_DECOMPOSITION" = always ] && [[ "$DECOMP" == *"no usable rewrite"* ]]; then
          echo "  WARNING: an LLM join was NOT decomposed (ran pairwise) although JOIN_DECOMPOSITION=always"
        fi
      fi
    fi
  done
done
echo "Benchmark Done!"
