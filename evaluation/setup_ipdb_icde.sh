#!/usr/bin/env bash
# Build the iPDB databases for the ICDE runs and register the LLM models in them.
#
# The May-2026 iPDB databases (sembench_sf2000.db, benchmark.db, pc_part.db) can't be opened by
# iPDB v1.5.5 (catalog format changed), so this builds fresh ones from the plain-DuckDB copies that
# flock / Sema use:
#
#   sembench_movies  sembench_sf2000_icde.db  <- movie_database.duckdb : reviews_2000 AS review
#   semantic_movies  benchmark_icde.db        <- benchmark_duckdb.db   : movie, movie_info, info_type, review
#   pc_parts         pc_part_icde.db          <- pc_part_duckdb.db     : Product, Category, Review
#
# and registers every model below in each DB (skipping ones that are already there). Re-running is
# safe: existing DBs are kept, only missing models are added. FORCE=1 rebuilds the DBs.
#
# Models use the persistent iPDB secrets `openai`, `gemini` and `anth` (check: SELECT name FROM duckdb_secrets();).
# If one is missing, create it first, e.g.
#   CREATE PERSISTENT SECRET openai (TYPE http, bearer_token '<OPENAI_API_KEY>');
#   CREATE PERSISTENT SECRET gemini (TYPE http, bearer_token '<GEMINI_API_KEY>');
#   CREATE PERSISTENT SECRET anth   (TYPE http, bearer_token '<ANTHROPIC_API_KEY>');
# Claude goes through Anthropic's OpenAI-compatible endpoint (api.anthropic.com/v1/chat/completions), which
# accepts the key as a bearer token. A model registered with a bad key returns NULLs (0 tokens), not an error.
#
# Usage: bash evaluation/setup_ipdb_icde.sh   (from the repo root or evaluation/)
set -euo pipefail

cd "$(dirname "$0")/.."                     # repo root
IPDB="../iPDb/build/release_api/ipdb"
REQ_PER_MIN="${REQ_PER_MIN:-500}"
OPENAI_API="https://api.openai.com/v1/"
GEMINI_API="https://generativelanguage.googleapis.com/v1beta/openai/"
ANTHROPIC_API="https://api.anthropic.com/v1/"

# name used in queries (LLM <name> ...) | provider model id | API | secret
MODELS="
gpt6luna      gpt-6-luna            $OPENAI_API openai
gemini38flash gemini-3.8-flash      $GEMINI_API gemini
gemini35lite  gemini-3.5-flash-lite $GEMINI_API gemini
gemini4       gemini-4-argon        $GEMINI_API gemini
sonnet55      claude-sonnet-5-5     $ANTHROPIC_API anth
"

build_db() {  # build_db <db> <source db> <SQL creating the tables from schema src>
  local db="$1" src="$2" sql="$3"
  if [ -e "$db" ] && [ -z "${FORCE:-}" ]; then
    echo "$db exists, keeping it (FORCE=1 to rebuild)"; return
  fi
  rm -f "$db" "$db.wal"
  "$IPDB" "$db" <<SQL
ATTACH '${src}' AS src (READ_ONLY);
${sql}
DETACH src;
SQL
  echo "Created $db"
}

register_models() {  # register_models <db>
  local db="$1" existing
  existing="$("$IPDB" "$db" -list -noheader -c "SELECT model_name FROM ipdb_models();")"
  while read -r name path api secret; do
    [ -n "$name" ] || continue
    if echo "$existing" | grep -qx "$name"; then continue; fi
    "$IPDB" "$db" -c "CREATE LLM MODEL ${name} PATH '${path}' ON PROMPT API '${api}' SECRET ${secret} OPTIONS {\"req_per_min\": ${REQ_PER_MIN}};" >/dev/null
    echo "  registered ${name} (${path}) in ${db}"
  done <<< "$MODELS"
}

build_db sembench_sf2000_icde.db movie_database.duckdb "
CREATE TABLE review AS SELECT * FROM src.reviews_2000;"

build_db benchmark_icde.db benchmark_duckdb.db "
CREATE TABLE movie      AS SELECT * FROM src.movie;
CREATE TABLE movie_info AS SELECT * FROM src.movie_info;
CREATE TABLE info_type  AS SELECT * FROM src.info_type;
CREATE TABLE review     AS SELECT * FROM src.review;"

build_db pc_part_icde.db pc_part_duckdb.db "
CREATE TABLE Product  AS SELECT * FROM src.Product;
CREATE TABLE Category AS SELECT * FROM src.category;
CREATE TABLE Review   AS SELECT * FROM src.review;"

for db in sembench_sf2000_icde.db benchmark_icde.db pc_part_icde.db; do
  register_models "$db"
  echo "$db: $("$IPDB" -readonly "$db" -list -noheader -c "SELECT string_agg(table_name, ', ' ORDER BY table_name) FROM duckdb_tables();") | models: $("$IPDB" -readonly "$db" -list -noheader -c "SELECT string_agg(model_name, ', ' ORDER BY model_name) FROM ipdb_models();")"
done
