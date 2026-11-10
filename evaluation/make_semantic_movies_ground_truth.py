#!/usr/bin/env python3
"""
Generate ground truth for semantic_movies q1-q4 (the end-to-end evaluation queries)
-> evaluation/results/semantic_movies/ground_truth/

Data: benchmark_duckdb.db (plain DuckDB copy of the semantic_movies tables; same rows as benchmark_icde.db,
which evaluation/queries/semantic_movies/semantic_movies.dbname points at - checked with a hash over
movie(id, title, genre, lang) and the Star Trek reviews).

The queries read only the plot / title / review text. The answers are taken from columns the queries do not
read: Rotten Tomatoes metadata on `movie` (genre, lang) and `review` (sentiment), and IMDB `movie_info` rows
(info types 'genres', 'languages').

q1  genre and main character of every movie, from its plot (6,719 plot rows of 4,999 movies; one GT row per movie id)
    genre            the RT genre list (movie.genre), as is.
    accepted_genres  canonical tokens (see canonical_genre) of RT genre UNION IMDB 'genres'. A predicted genre is
                     correct when its canonical token is in this set. 4 movies have no genre in either source.
    main_character   NOT in the data (the IMDB cast table has actors but no character names and is incomplete),
                     so it comes from a reference-model label: label_semantic_movies_main_character.py writes
                     q1_main_character.csv, merged here when present (columns main_character, aliases,
                     co_leads, no_character). Without that file the columns are left empty.

q2  language of every movie, from its title (5,000 movie rows = 4,999 ids; "Joy" 2004619 is listed twice)
    language            RT original language (movie.lang) without the region suffix ("English (United Kingdom)"
                        -> English); if RT has none / "Unknown language", the first IMDB 'languages' row.
    accepted_languages  canonical names (see canonical_language) of RT language UNION all IMDB 'languages'.
                        29 movies have no language in either source and are left out of the GT.

q3  negative reviews of 'Star Trek' (320 reviews of the 2009 film)
    Rule: review.sentiment = 'NEGATIVE' (the critic's own RT rotten verdict): 20 reviews.
    All 320 texts were read: the 20 read negative, but some RT labels and texts disagree in tone. Those are
    written to q3_ambiguous.csv (negative label + mixed text, or positive label + negative / mixed / no-opinion
    text) so a lenient evaluation can accept them either way, as for pc_parts q4.

q4  all MPA maturity ratings in the US (pure generation, no input rows)
    The five current MPA (MPAA) film ratings G, PG, PG-13, R, NC-17 with the MPA's own descriptions.
    q4_ambiguous.csv lists labels that are not current MPA ratings but are often listed with them (NR / Unrated,
    and the retired X, M, GP); a lenient evaluation can ignore them instead of counting them as wrong.

Usage: .venv/bin/python evaluation/make_semantic_movies_ground_truth.py
       SEMANTIC_MOVIES_DB=/path/benchmark_duckdb.db SEMANTIC_MOVIES_GT_DIR=/path/out ... (overrides)
"""

import os
import re
from pathlib import Path

import duckdb
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent


def _default_db() -> Path:
    db = REPO / "benchmark_duckdb.db"
    if not db.exists() and ".claude" in REPO.parts:  # running from a git worktree: use the main checkout's copy
        db = Path(*REPO.parts[:REPO.parts.index(".claude")]) / "benchmark_duckdb.db"
    return db


DB = Path(os.environ.get("SEMANTIC_MOVIES_DB", _default_db()))
OUT = Path(os.environ.get("SEMANTIC_MOVIES_GT_DIR", EVAL_DIR / "results" / "semantic_movies" / "ground_truth"))
MAIN_CHARACTER_CSV = OUT / "q1_main_character.csv"  # written by label_semantic_movies_main_character.py

# ---------------------------------------------------------------------------------------------------------------
# Canonical genres / languages (shared with the evaluator: predictions go through the same functions)
# ---------------------------------------------------------------------------------------------------------------
GENRE_SYNONYMS = {
    "sci-fi": "sci-fi", "scifi": "sci-fi", "science fiction": "sci-fi", "science-fiction": "sci-fi",
    "kids": "family", "family": "family", "children": "family", "kids & family": "family",
    "lgbtq+": "lgbtq+", "lgbtq": "lgbtq+", "lgbt": "lgbtq+",
    "sport": "sport", "sports": "sport", "sports & fitness": "sport", "fitness": "sport",
    "anime": "animation", "animation": "animation", "animated": "animation", "cartoon": "animation",
    "music": "music", "musical": "music",
    "biography": "biography", "biopic": "biography", "biographical": "biography",
    "history": "history", "historical": "history", "period": "history",
    "romance": "romance", "romantic": "romance",
    "thriller": "thriller", "suspense": "thriller", "mystery": "mystery",
    "film-noir": "film-noir", "film noir": "film-noir", "noir": "film-noir",
    "reality-tv": "reality-tv", "reality": "reality-tv",
    "war": "war", "western": "western", "horror": "horror", "comedy": "comedy", "drama": "drama",
    "documentary": "documentary", "docudrama": "documentary", "action": "action", "adventure": "adventure",
    "crime": "crime", "fantasy": "fantasy", "holiday": "holiday", "christmas": "holiday", "short": "short",
    "news": "news",
}


def split_genres(s) -> list[str]:
    """'Mystery & thriller, Drama' / 'Romantic comedy' / 'Sci-Fi/Action' -> canonical tokens."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    text = str(s).lower().strip()
    out = []
    for compound in ("kids & family", "sports & fitness", "science fiction", "film noir"):
        if compound in text:
            out.append(GENRE_SYNONYMS[compound])
            text = text.replace(compound, ",")
    for part in re.split(r"[,/;|&]| and ", text):
        part = part.strip(" .-")
        if not part:
            continue
        g = canonical_genre(part)
        if g:
            out.append(g)
        else:  # "romantic comedy", "action thriller": canonicalise word by word
            out.extend(g for w in part.split() if (g := canonical_genre(w)))
    return list(dict.fromkeys(out))


def canonical_genre(s: str) -> str | None:
    return GENRE_SYNONYMS.get(s.lower().strip())


LANGUAGE_SYNONYMS = {
    "bangla": "bengali", "tagalog": "filipino", "mandarin": "chinese", "cantonese": "chinese",
    "farsi": "persian", "dari": "persian", "flemish": "dutch", "serbo-croatian": "serbian",
    "croatian": "serbian", "bosnian": "serbian", "central khmer": "khmer", "pashtu": "pashto",
    "letzebuergesh": "luxembourgish",
}


def canonical_language(s) -> str | None:
    """'English (United Kingdom)' -> 'english'; 'Mandarin' -> 'chinese'; None / 'Unknown language' -> None."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = re.sub(r"\s*\(.*\)\s*$", "", str(s)).strip().lower()
    if s in ("", "none", "unknown language", "unknown"):
        return None
    return LANGUAGE_SYNONYMS.get(s, s)


# ---------------------------------------------------------------------------------------------------------------
# q3 / q4 hand-curated parts
# ---------------------------------------------------------------------------------------------------------------
STAR_TREK_REVIEWS = 320
STAR_TREK_NEGATIVE = 20
# review_id -> (text prefix, note). Not changing the GT (= RT label); accepted either way by a lenient evaluation.
Q3_AMBIGUOUS = {
    # RT NEGATIVE, text mixed
    1818885: ("That jolt of energy may resuscitate", "label negative, text mixed (energetic but shallow)"),
    1820797: ("Well, at least they got it half right", "label negative, text mixed (half right)"),
    1823081: ("While entertaining as a space soap opera", "label negative, text mixed (entertaining but...)"),
    # RT POSITIVE, text reads negative
    1819147: ("Is this where the future of Trek lies", "label positive, text negative (underwhelming, stale)"),
    1864570: ("Thanks for the memories; let's leave it at that", "label positive, text reads negative"),
    102723973: ("There were times where this didn", "label positive, text negative (doesn't feel like Trek)"),
    1819326: ("If you realise that it has one aim", "label positive, text negative for the reviewer (tutting)"),
    # RT POSITIVE, text mixed
    1819281: ("All in all, the material is strong enough", "label positive, text mixed (Abrams' fumbling)"),
    1819333: ("Given Bana's baddie is about as threatening", "label positive, text mixed (weak villain/plot)"),
    1819684: ("The young cast is mostly callow", "label positive, text mixed (bland cast, presentable)"),
    1819689: ("Fun, but like most of the previous movies", "label positive, text mixed"),
    1819832: ("Great characters, nice comic touches", "label positive, text mixed (plot cries out)"),
    1821298: ("Well-cast and well-acted...however", "label positive, text mixed (falls short)"),
    1828911: ("[It] broadens the appeal of its cult property", "label positive, text mixed (shaggy dog)"),
    1820565: ("Clever, campy, even a little endearing", "label positive, text mixed (consumes itself)"),
    2107497: ("The character work is consistently ace", "label positive, text mixed (no high-minded ideas)"),
    2251882: ("Abrams had a responsibility", "label positive, text mixed (succeeds in parts)"),
    # RT POSITIVE, text expresses no opinion
    1819067: ("Parental Content Review", "label positive, text has no sentiment"),
    2334176: ("This is not your father's Star Trek (for better or for worse)", "label positive, text neutral"),
    2828438: ("One of of them set their phaser to stun", "label positive, text reports two opposite views"),
}

MPA_RATINGS = [
    ("G", "General Audiences",
     "All ages admitted. Nothing that would offend parents for viewing by children."),
    ("PG", "Parental Guidance Suggested",
     "Some material may not be suitable for children. Parents urged to give parental guidance; may contain some "
     "material parents might not like for their young children."),
    ("PG-13", "Parents Strongly Cautioned",
     "Some material may be inappropriate for children under 13. Parents are urged to be cautious; some material "
     "may be inappropriate for pre-teenagers."),
    ("R", "Restricted",
     "Under 17 requires accompanying parent or adult guardian. Contains some adult material."),
    ("NC-17", "Adults Only",
     "No one 17 and under admitted. Clearly adult content."),
]
Q4_AMBIGUOUS = [
    ("NR", "Not Rated / Unrated: a film not submitted to the MPA; not an MPA rating"),
    ("X", "retired MPA rating, replaced by NC-17 in 1990"),
    ("M", "retired MPA rating (1968-1970), replaced by GP"),
    ("GP", "retired MPA rating (1970-1972), renamed PG"),
]


def _as_list(v) -> list:
    """DuckDB list column cell -> list (a correlated list() over no rows arrives as NULL / NaN)."""
    return [] if v is None or v is pd.NA or (isinstance(v, float) and pd.isna(v)) else list(v)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB), read_only=True)
    it = dict(con.sql("SELECT info, id FROM info_type").fetchall())
    # movie has one duplicated id (2004619 "Joy": once with genre/lang, once with NULLs); one GT row per id
    con.sql("CREATE TEMP VIEW movie_u AS SELECT id, min(title) AS title, max(genre) AS genre, max(lang) AS lang "
            "FROM movie GROUP BY id")

    # q1 ------------------------------------------------------------------------------------------------------
    q1 = con.sql(f"""
        SELECT m.id, m.title, m.genre,
               (SELECT list(g.info ORDER BY g.id) FROM movie_info g
                 WHERE g.movie_id = m.id AND g.info_type_id = {it['genres']}) AS imdb_genres
        FROM movie_u m
        WHERE EXISTS (SELECT 1 FROM movie_info p WHERE p.movie_id = m.id AND p.info_type_id = {it['plot']})
        ORDER BY m.id""").df()
    assert len(q1) == 4999, f"expected 4999 movies with a plot, got {len(q1)}"
    q1["imdb_genres"] = q1.imdb_genres.map(lambda l: "|".join(_as_list(l)))
    q1["accepted_genres"] = [
        "|".join(dict.fromkeys(split_genres(rt) + [g for x in imdb.split("|") if x for g in split_genres(x)]))
        for rt, imdb in zip(q1.genre, q1.imdb_genres)]
    unknown = {t.strip() for gs in q1.genre.dropna() for t in gs.split(",")} | \
              {x for gs in q1.imdb_genres for x in gs.split("|") if x}
    unmapped = sorted(t for t in unknown if not split_genres(t))
    assert not unmapped, f"genres without a canonical token: {unmapped}"

    mc_cols = ["main_character", "aliases", "co_leads", "no_character"]
    if MAIN_CHARACTER_CSV.exists():
        mc = pd.read_csv(MAIN_CHARACTER_CSV, keep_default_na=False)
        q1 = q1.merge(mc[["id", *mc_cols]], on="id", how="left")
    else:
        for c in mc_cols:
            q1[c] = ""
    q1.to_csv(OUT / "q1.csv", index=False)

    # q2 ------------------------------------------------------------------------------------------------------
    q2 = con.sql(f"""
        SELECT m.id, m.title, m.lang AS rt_language,
               (SELECT list(l.info ORDER BY l.id) FROM movie_info l
                 WHERE l.movie_id = m.id AND l.info_type_id = {it['languages']}) AS imdb_languages
        FROM movie_u m ORDER BY m.id""").df()
    assert len(q2) == 4999
    q2["imdb_languages"] = q2.imdb_languages.map(lambda l: [x for x in _as_list(l) if x != "None"])
    q2["accepted_languages"] = [
        "|".join(dict.fromkeys(x for x in [canonical_language(rt), *map(canonical_language, imdb)] if x))
        for rt, imdb in zip(q2.rt_language, q2.imdb_languages)]

    def primary(rt, imdb):
        if canonical_language(rt):
            return re.sub(r"\s*\(.*\)\s*$", "", rt).strip()
        return imdb[0] if imdb else None

    q2["language"] = [primary(rt, imdb) for rt, imdb in zip(q2.rt_language, q2.imdb_languages)]
    q2["imdb_languages"] = q2.imdb_languages.map("|".join)
    no_lang = q2[q2.accepted_languages == ""]
    q2 = q2[q2.accepted_languages != ""]
    q2[["id", "title", "language", "accepted_languages", "rt_language", "imdb_languages"]] \
        .to_csv(OUT / "q2.csv", index=False)

    # q3 ------------------------------------------------------------------------------------------------------
    reviews = con.sql("""
        SELECT r.movie_id, r.review_id, r.sentiment, r.score, r.review
        FROM movie m JOIN review r ON m.id = r.movie_id WHERE m.title = 'Star Trek'
        ORDER BY r.review_id""").df().set_index("review_id", drop=False)
    assert len(reviews) == STAR_TREK_REVIEWS, f"expected {STAR_TREK_REVIEWS} Star Trek reviews, got {len(reviews)}"
    for rid, (prefix, _) in Q3_AMBIGUOUS.items():
        assert reviews.loc[rid, "review"].startswith(prefix), f"q3 label {rid} no longer matches the data"
    neg = reviews[reviews.sentiment == "NEGATIVE"]
    assert len(neg) == STAR_TREK_NEGATIVE, f"expected {STAR_TREK_NEGATIVE} negative reviews, got {len(neg)}"
    neg[["movie_id", "review_id", "review", "score"]].to_csv(OUT / "q3.csv", index=False)
    amb = reviews.loc[list(Q3_AMBIGUOUS)].copy()
    amb["note"] = amb.review_id.map(lambda r: Q3_AMBIGUOUS[r][1])
    amb[["movie_id", "review_id", "sentiment", "review", "note"]].to_csv(OUT / "q3_ambiguous.csv", index=False)

    # q4 ------------------------------------------------------------------------------------------------------
    pd.DataFrame(MPA_RATINGS, columns=["label", "name", "explanation"]).to_csv(OUT / "q4.csv", index=False)
    pd.DataFrame(Q4_AMBIGUOUS, columns=["label", "note"]).to_csv(OUT / "q4_ambiguous.csv", index=False)

    n_mc = int((q1.main_character.fillna("") != "").sum() + (q1.no_character.astype(str) == "True").sum())
    print(f"q1: {len(q1)} movies; accepted genres for {int((q1.accepted_genres != '').sum())} "
          f"({int(q1.genre.isna().sum())} without an RT genre); main character for {n_mc}"
          + ("" if MAIN_CHARACTER_CSV.exists() else f"  (no {MAIN_CHARACTER_CSV.name}: run the labeller)"))
    print(f"q2: {len(q2)} movies with a language ({len(no_lang)} without one left out); "
          f"top: {q2.language.value_counts().head(5).to_dict()}")
    print(f"q3: {len(neg)} negative of {len(reviews)} reviews (+ {len(amb)} ambiguous)")
    print(f"q4: {len(MPA_RATINGS)} ratings (+ {len(Q4_AMBIGUOUS)} ambiguous)")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
