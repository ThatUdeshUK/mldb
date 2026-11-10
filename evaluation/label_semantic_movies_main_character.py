#!/usr/bin/env python3
"""
Reference labels for the main character of every semantic_movies movie (q1's main_character column)
-> evaluation/results/semantic_movies/ground_truth/q1_main_character.csv
(merged into q1.csv by make_semantic_movies_ground_truth.py)

Why a model: the benchmark data has no character names (the IMDB cast table lists actors, no characters, and is
incomplete), so the main character cannot be derived from the data. This is a SILVER label from a strong
reference model (default Claude Opus 5.5) that reads all plot summaries of a movie at once, i.e. more context than
the benchmark query (which sees one plot row at a time). Aliases and co-leads are recorded so the evaluation can
accept "Kirk" for "James T. Kirk" and either lead of a two-hander.

Runs through the Message Batches API (50% price, asynchronous). The batch id is saved next to the output, so the
script can be re-run to resume polling / collect results. Movies missing from the results (errored, refused,
malformed) are listed and can be retried with --retry-missing.

Usage (needs ANTHROPIC_API_KEY or an `ant auth login` profile):
    uv run --with anthropic --with duckdb --with pandas python evaluation/label_semantic_movies_main_character.py --estimate
    uv run --with anthropic --with duckdb --with pandas python evaluation/label_semantic_movies_main_character.py
    uv run ... python evaluation/label_semantic_movies_main_character.py --retry-missing
Then: .venv/bin/python evaluation/make_semantic_movies_ground_truth.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_semantic_movies_ground_truth import DB, OUT  # noqa: E402

MODEL = "claude-opus-5-5"
PRICE_PER_MTOK = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00)}  # standard; batch = 50%
MOVIES_PER_REQUEST = 20
OUT_CSV = OUT / "q1_main_character.csv"
BATCH_FILE = OUT / "q1_main_character.batches.json"

INSTRUCTIONS = """You are building ground truth for a movie benchmark. For each movie below you get its title and \
one or more plot summaries. Identify the movie's MAIN CHARACTER: the protagonist the story follows / centres on.

Rules:
- Use the character's name as it appears in the plot summaries (full name if given, e.g. "James T. Kirk").
- aliases: other ways the summaries (or the obvious short form) refer to the SAME character, e.g. ["Kirk", "Jim"].
- co_leads: other characters that are equally central (two-hander, ensemble), so either could fairly be called the \
main character. Leave empty when one protagonist is clear.
- If the protagonist is not named but described (e.g. "a young widow"), use that description.
- For a documentary or non-fiction film, the main character is the central real person or group the film is about, \
if any.
- no_character = true only if the summaries name or describe no person or character at all (e.g. a documentary \
about a place or an idea); then main_character is "".
- Base the answer on the plot summaries; use your knowledge of the film only to break ties.
Return one entry per movie, with the movie's id exactly as given."""

SCHEMA = {
    "type": "object",
    "properties": {
        "movies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "main_character": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "co_leads": {"type": "array", "items": {"type": "string"}},
                    "no_character": {"type": "boolean"},
                },
                "required": ["id", "main_character", "aliases", "co_leads", "no_character"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["movies"],
    "additionalProperties": False,
}


def load_movies() -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    return con.sql("""
        SELECT m.id, min(m.title) AS title, list(DISTINCT mi.info) AS plots
        FROM movie m JOIN movie_info mi ON mi.movie_id = m.id
        JOIN info_type it ON it.id = mi.info_type_id AND it.info = 'plot'
        GROUP BY m.id ORDER BY m.id""").df()


def render(chunk: pd.DataFrame) -> str:
    parts = []
    for r in chunk.itertuples():
        plots = "\n".join(f"  plot {i + 1}: {p}" for i, p in enumerate(r.plots))
        parts.append(f"<movie id=\"{r.id}\" title=\"{r.title}\">\n{plots}\n</movie>")
    return "\n\n".join(parts)


def build_requests(movies: pd.DataFrame, model: str, effort: str):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    reqs = []
    for start in range(0, len(movies), MOVIES_PER_REQUEST):
        chunk = movies.iloc[start:start + MOVIES_PER_REQUEST]
        reqs.append(Request(
            custom_id=f"mc-{chunk.id.iloc[0]}-{chunk.id.iloc[-1]}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=16000,
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
                system=[{"type": "text", "text": INSTRUCTIONS, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": render(chunk)}],
            )))
    return reqs


def estimate(movies: pd.DataFrame, model: str):
    n_req = -(-len(movies) // MOVIES_PER_REQUEST)
    chars = sum(len(render(movies.iloc[s:s + MOVIES_PER_REQUEST])) for s in range(0, len(movies), MOVIES_PER_REQUEST))
    in_tok = chars / 3.5 + n_req * len(INSTRUCTIONS) / 3.5
    out_tok = len(movies) * 45 + n_req * 1500  # JSON per movie + thinking per request at low effort (rough)
    pin, pout = PRICE_PER_MTOK[model]
    cost = (in_tok * pin + out_tok * pout) / 1e6 * 0.5
    print(f"{len(movies)} movies, {n_req} requests, ~{in_tok / 1e6:.2f}M input / ~{out_tok / 1e6:.2f}M output tokens")
    print(f"estimated batch cost on {model}: ~${cost:.2f} (rough: chars/3.5 tokens, thinking guessed)")


def collect(client, batch_ids: list[str]) -> list[dict]:
    rows, problems = [], []
    for bid in batch_ids:
        while (b := client.messages.batches.retrieve(bid)).processing_status != "ended":
            print(f"{bid}: {b.processing_status}, {b.request_counts.processing} processing, "
                  f"{b.request_counts.succeeded} done", flush=True)
            time.sleep(60)
        for res in client.messages.batches.results(bid):
            if res.result.type != "succeeded":
                problems.append(f"{res.custom_id}: {res.result.type}")
                continue
            msg = res.result.message
            if msg.stop_reason != "end_turn":
                problems.append(f"{res.custom_id}: stop_reason {msg.stop_reason}")
                continue
            text = next((blk.text for blk in msg.content if blk.type == "text"), "")
            try:
                rows.extend(json.loads(text)["movies"])
            except (json.JSONDecodeError, KeyError) as e:
                problems.append(f"{res.custom_id}: bad JSON ({e})")
    for p in problems:
        print("  problem:", p)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=MODEL, choices=sorted(PRICE_PER_MTOK))
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--estimate", action="store_true", help="print a cost estimate and exit (no API calls)")
    ap.add_argument("--retry-missing", action="store_true", help="submit a new batch for movies not yet labelled")
    args = ap.parse_args()

    movies = load_movies()
    done = pd.read_csv(OUT_CSV, keep_default_na=False) if OUT_CSV.exists() else None
    todo = movies if done is None else movies[~movies.id.isin(done.id)]
    if args.estimate:
        estimate(todo, args.model)
        return

    import anthropic
    client = anthropic.Anthropic()
    OUT.mkdir(parents=True, exist_ok=True)
    state = json.loads(BATCH_FILE.read_text()) if BATCH_FILE.exists() else {"pending": [], "collected": []}
    if not state["pending"] and (done is None or args.retry_missing) and len(todo):
        batch = client.messages.batches.create(requests=build_requests(todo, args.model, args.effort))
        state["pending"].append(batch.id)
        BATCH_FILE.write_text(json.dumps(state, indent=1))
        print(f"submitted batch {batch.id}: {len(todo)} movies on {args.model} (effort {args.effort})")
    if not state["pending"]:
        print(f"nothing pending; {0 if done is None else len(done)} labelled, {len(todo)} missing "
              f"(use --retry-missing)")
        return

    rows = collect(client, state["pending"])
    valid = set(movies.id)
    new = pd.DataFrame([r for r in rows if r["id"] in valid]).drop_duplicates("id")
    if len(new):
        new["aliases"] = new.aliases.map("|".join)
        new["co_leads"] = new.co_leads.map("|".join)
        new = new.merge(movies[["id", "title"]], on="id")
        merged = pd.concat([done, new]) if done is not None else new
        merged = merged.drop_duplicates("id", keep="last").sort_values("id")
        merged[["id", "title", "main_character", "aliases", "co_leads", "no_character"]].to_csv(OUT_CSV, index=False)
    state["collected"] += state["pending"]
    state["pending"] = []
    BATCH_FILE.write_text(json.dumps(state, indent=1))
    labelled = pd.read_csv(OUT_CSV, keep_default_na=False) if OUT_CSV.exists() else pd.DataFrame(columns=["id"])
    missing = sorted(valid - set(labelled.id))
    print(f"{len(labelled)} of {len(valid)} movies labelled -> {OUT_CSV}")
    if missing:
        print(f"{len(missing)} missing (first: {missing[:10]}); re-run with --retry-missing")


if __name__ == "__main__":
    main()
