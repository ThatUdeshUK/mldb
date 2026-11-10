#!/usr/bin/env python3
"""Summarise the merging evaluation: results/*.json -> analysis/{runs.csv, summary.csv, gain.csv, report.md}.

Usage: python3 sem_merge_evaluation/analyze.py
"""

import csv
import json
import re
from collections import defaultdict
from statistics import mean

from common import ANALYSIS_DIR, OPS, RESULTS_DIR, STEM_RE, STRATEGIES, WORKLOADS, load_dataset, load_prices, model_key


def norm(op_name: str, v) -> str:
    s = str(v or "").strip()
    if op_name == "rating":
        s = re.sub(r"[^A-Z0-9]", "", s.upper().removeprefix("RATED"))
        return {"PG13": "PG-13", "NC17": "NC-17"}.get(s, s)
    if op_name == "language":
        return re.split(r"[(/,;]", s.lower())[0].strip()
    return s.lower()


def gt_bool(v: str) -> bool | None:
    return {"True": True, "False": False}.get(v)


def prf(pred: set, truth: set) -> tuple[float, float, float]:
    tp = len(pred & truth)
    p = tp / len(pred) if pred else (1.0 if not truth else 0.0)
    r = tp / len(truth) if truth else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def result_set(outputs: dict, filters: list[str]) -> set:
    return {k for k, v in outputs.items() if all(v.get(f) is True for f in filters)}


def analyse_run(res: dict, data: dict[str, dict], prices: dict) -> dict:
    meta = res["meta"]
    ops = [OPS[o] for o in meta["ops"]]
    filters = [o.name for o in ops if o.kind == "filter"]
    projects = [o for o in ops if o.kind == "project"]
    outputs = res["outputs"]
    st = res["stages"]
    row = {k: meta[k] for k in ("workload", "strategy", "mode", "data", "run")}
    row["model"] = model_key(meta["model"])
    row.update(rows_in=sum(s["rows_in"] for s in st), calls=sum(s["calls"] for s in st),
               prompt_tokens=sum(s["prompt_tokens"] for s in st),
               completion_tokens=sum(s["completion_tokens"] for s in st),
               retries=sum(s["retries"] for s in st), latency_s=res["latency_s"])
    price = prices.get(row["model"])
    row["cost_usd"] = (row["prompt_tokens"] * price["input"] + row["completion_tokens"] * price["output"]) / 1e6 \
        if price else None

    # End-to-end quality of the filter part: rows whose filter labels are all known.
    universe = {k for k in outputs if all(gt_bool(data[k][OPS[f].gt]) is not None for f in filters)}
    pred = result_set(outputs, filters) & universe
    truth = {k for k in universe if all(gt_bool(data[k][OPS[f].gt]) for f in filters)}
    row["returned"] = len(result_set(outputs, filters))
    row["precision"], row["recall"], row["f1"] = prf(pred, truth)

    # Projection accuracy on the returned rows (what the query outputs), where a label exists.
    for o in projects:
        evald = [k for k in result_set(outputs, filters) if data[k][o.gt] and o.name in outputs[k]]
        row["proj_acc"] = float(mean(norm(o.name, outputs[k][o.name]) == norm(o.name, data[k][o.gt])
                                     for k in evald)) if evald else None

    # Per-operator accuracy wherever the strategy computed the operator.
    for o in ops:
        evald = [k for k, v in outputs.items() if o.name in v and data[k][o.gt] != ""]
        if o.kind == "filter":
            accs = [outputs[k][o.name] is gt_bool(data[k][o.gt]) for k in evald]
        else:
            accs = [norm(o.name, outputs[k][o.name]) == norm(o.name, data[k][o.gt]) for k in evald]
        row[f"acc_{o.name}"] = float(mean(accs)) if accs else None
        row[f"n_{o.name}"] = len(evald)
    return row


def agreement(a: dict, b: dict, ops: list[str]) -> float | None:
    """Fraction of (row, operator) values computed by both runs on which they agree."""
    triples = [(o, a[k][o], b[k][o]) for k in a.keys() & b.keys() for o in ops if o in a[k] and o in b[k]]
    return float(mean(norm(o, x) == norm(o, y) for o, x, y in triples)) if triples else None


def fmt(v, d=3):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def pct(new, base):
    return None if new is None or not base else 100.0 * (new - base) / base


def main():
    prices = load_prices()
    files = sorted(p for p in RESULTS_DIR.glob("*.json") if not p.name.endswith(".failed.json"))
    failed = sorted(RESULTS_DIR.glob("*.failed.json"))
    if not files:
        raise SystemExit(f"no results in {RESULTS_DIR}; run sem_merge_evaluation/run.py first")

    datasets, results, runs = {}, {}, []
    for p in files:
        if not STEM_RE.match(p.stem):
            continue
        res = json.loads(p.read_text())
        if res["meta"]["strategy"] not in STRATEGIES:  # dropped strategies (e.g. the old "independent")
            continue
        d = res["meta"]["data"]
        if d not in datasets:
            datasets[d] = {r["id"]: r for r in load_dataset(d)}
        row = analyse_run(res, datasets[d], prices)
        results[p.stem] = res
        runs.append(row)

    # Agreement of every strategy with "separate" (same workload/model/mode/data/run).
    by_key = {(r["workload"], r["strategy"], r["model"], r["mode"], r["data"], r["run"]): s
              for r, s in zip(runs, results)}
    for r in runs:
        base = by_key.get((r["workload"], "separate", r["model"], r["mode"], r["data"], r["run"]))
        cur = by_key[(r["workload"], r["strategy"], r["model"], r["mode"], r["data"], r["run"])]
        if base is None:
            r["agree_separate"] = r["result_jaccard_separate"] = None
            continue
        a, b = results[cur], results[base]
        r["agree_separate"] = agreement(a["outputs"], b["outputs"], a["meta"]["ops"])
        filters = [o for o in a["meta"]["ops"] if OPS[o].kind == "filter"]
        sa, sb = result_set(a["outputs"], filters), result_set(b["outputs"], filters)
        r["result_jaccard_separate"] = len(sa & sb) / len(sa | sb) if sa | sb else 1.0

    ANALYSIS_DIR.mkdir(exist_ok=True)
    cols = list(dict.fromkeys(k for r in runs for k in r))
    with open(ANALYSIS_DIR / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(runs)

    # Means per (workload, strategy, model, mode, data).
    groups = defaultdict(list)
    for r in runs:
        groups[(r["workload"], r["strategy"], r["model"], r["mode"], r["data"])].append(r)
    metrics = ["rows_in", "calls", "prompt_tokens", "completion_tokens", "cost_usd", "latency_s", "returned",
               "precision", "recall", "f1", "proj_acc", "agree_separate", "result_jaccard_separate"]
    summary = []
    for key, rs in groups.items():
        s = dict(zip(["workload", "strategy", "model", "mode", "data"], key), runs=len(rs))
        for m in metrics:
            vals = [r.get(m) for r in rs if r.get(m) is not None]
            s[m] = mean(vals) if vals else None
        summary.append(s)
    order = {w: i for i, w in enumerate(WORKLOADS)}
    summary.sort(key=lambda s: (s["model"], s["mode"], s["data"], order[s["workload"]],
                                STRATEGIES.index(s["strategy"])))
    with open(ANALYSIS_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    # Merge gain: merged vs separate, and vs the cheapest unmerged strategy.
    idx = {(s["workload"], s["strategy"], s["model"], s["mode"], s["data"]): s for s in summary}
    gains = []
    for s in summary:
        if s["strategy"] != "merged":
            continue
        k = (s["workload"], s["model"], s["mode"], s["data"])
        sep = idx.get((k[0], "separate", *k[1:]))
        unmerged = [idx[(k[0], st, *k[1:])] for st in STRATEGIES if st != "merged" and (k[0], st, *k[1:]) in idx]
        cost_key = "cost_usd" if all(u["cost_usd"] is not None for u in unmerged) else "prompt_tokens"
        best = min(unmerged, key=lambda u: (u[cost_key] if cost_key != "prompt_tokens"
                                            else u["prompt_tokens"] + u["completion_tokens"])) if unmerged else None
        g = dict(zip(["workload", "model", "mode", "data"], k))
        for name, base in (("separate", sep), ("best", best)):
            if base is None:
                continue
            if name == "best":
                g["best_unmerged"] = base["strategy"]
            for m in ("calls", "prompt_tokens", "completion_tokens", "cost_usd", "latency_s"):
                g[f"{m}_vs_{name}_%"] = pct(s[m], base[m])
            g[f"f1_vs_{name}"] = None if s["f1"] is None or base["f1"] is None else s["f1"] - base["f1"]
        gains.append(g)
    if gains:
        with open(ANALYSIS_DIR / "gain.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for g in gains for k in g)))
            w.writeheader()
            w.writerows(gains)

    # report.md
    L = ["# Semantic operator merging: results", ""]
    if failed:
        L += [f"**Failed runs ({len(failed)}):** " + ", ".join(f"`{p.stem.removesuffix('.failed')}`" for p in failed), ""]
    for (model, mode, data) in sorted({(s["model"], s["mode"], s["data"]) for s in summary}):
        L += [f"## model `{model}`, mode `{mode}`, data `{data}`", ""]
        for wl, (kind, cols_desc, ops) in WORKLOADS.items():
            rows = [s for s in summary if (s["workload"], s["model"], s["mode"], s["data"]) == (wl, model, mode, data)]
            if not rows:
                continue
            L += [f"### `{wl}`: {kind}, {cols_desc}: " + " | ".join(f"`{o}`" for o in ops), "",
                  "| strategy | runs | rows in | calls | in tok | out tok | cost $ | latency s | returned | P | R | F1 "
                  "| proj acc | agree w/ separate | result Jaccard |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for s in rows:
                L.append("| " + " | ".join([s["strategy"], str(s["runs"])] + [
                    fmt(s["rows_in"], 0), fmt(s["calls"], 0), fmt(s["prompt_tokens"], 0),
                    fmt(s["completion_tokens"], 0), fmt(s["cost_usd"], 4), fmt(s["latency_s"], 1),
                    fmt(s["returned"], 0), fmt(s["precision"]), fmt(s["recall"]), fmt(s["f1"]),
                    fmt(s["proj_acc"]), fmt(s["agree_separate"]), fmt(s["result_jaccard_separate"])]) + " |")
            L.append("")
        gs = [g for g in gains if (g["model"], g["mode"], g["data"]) == (model, mode, data)]
        if gs:
            L += ["### Merge gain (merged vs. separate / vs. cheapest unmerged strategy; negative % = merged is lower)", "",
                  "| workload | calls % | in tok % | out tok % | cost % | latency % | ΔF1 | best unmerged | cost % vs best | latency % vs best | ΔF1 vs best |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for g in gs:
                L.append("| " + " | ".join([g["workload"]] + [fmt(g.get(k), 1) for k in (
                    "calls_vs_separate_%", "prompt_tokens_vs_separate_%", "completion_tokens_vs_separate_%",
                    "cost_usd_vs_separate_%", "latency_s_vs_separate_%")] + [fmt(g.get("f1_vs_separate")),
                    g.get("best_unmerged", "–"), fmt(g.get("cost_usd_vs_best_%"), 1),
                    fmt(g.get("latency_s_vs_best_%"), 1), fmt(g.get("f1_vs_best"))]) + " |")
            L.append("")
    L += ["## Metrics", "",
          "- **rows in**: rows sent to the LLM summed over operator invocations (separate: N + survivors; merged: N).",
          "- **P/R/F1**: the filter conjunction's result set against the labels (rows whose filter labels are known).",
          "- **proj acc**: projection accuracy on the returned rows that have a label (exact match after normalisation).",
          "- **agree w/ separate**: fraction of (row, operator) values computed by both this strategy and `separate` "
          "that are equal; **result Jaccard**: overlap of the two result sets.",
          "- **cost**: tokens × `prices.json`; completion tokens include reasoning tokens for reasoning models.", ""]
    (ANALYSIS_DIR / "report.md").write_text("\n".join(L))
    print(f"{len(runs)} runs -> {ANALYSIS_DIR}/report.md")


if __name__ == "__main__":
    main()
