#!/usr/bin/env python3
"""Analyse the join-decomposition runs (stdlib only). Writes to analysis/:

    runs.csv       one row per run: cost, latency, calls, tokens, quality, the decision taken and the T2 signals
    summary.csv    mean per (query, model, arm)
    decisions.csv  oracle vs. the auto / guarded arms: wrong choices, cost of wrong choices, quality lost
    step0.csv      T2 runtime-check signals vs. the quality decomposition actually loses (calibration)
    report.md      all of the above as tables, plus the rewrites the planning call produced

Quality:  F1 against ground truth (pairs are unordered), or precision@k for LIMIT queries (any k true pairs are
          a correct answer, so recall is meaningless there).
Reference F1: F1 against the pairwise arm's result (the paper's reference result, Sec. II-C); LIMIT queries
          have none, because different correct answers are equally valid.
Oracle:   per query and model, decompose iff the decomposed arm's quality is within --delta of the pairwise
          arm's and it is cheaper (USD if the model has a price in prices.json, otherwise LLM calls).
Choice:   what a run actually executed: 'decomposed'; 'pairwise' (T1 declined, or the planning call gave no
          usable rewrite); 'fallback' (decomposed, then the T2 runtime check switched back to pairwise).
Cost:     input/output tokens x prices.json. The planning call is counted as one LLM call, but iPDB does not
          report its tokens, so they are not in the token or USD columns.

Usage: python3 join_evaluation/analyze.py [--delta 0.05] [--models gpt6luna] [--results DIR]
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ANALYSIS_DIR, GT_DIR, HERE, RESULTS_DIR, STEM_RE, load_queries  # noqa: E402

GUARD_RE = re.compile(
    r"Runtime check: (?P<decision>decomposed|fallback) \((?P<reason>.*?)\); "  # reasons can nest parentheses
    r"unknown build (?P<unk_build>[\d.]+) / probe (?P<unk_probe>[\d.]+)"
    r"(?:, overlap (?P<overlap>[\d.]+))?(?:, selectivity (?P<selectivity>[\d.]+))?"
    r", build (?P<guard_build_rows>\d+) rows, probe sample (?P<guard_probe_rows>\d+) rows")
NOTE_RE = re.compile(r"\[decomp: ([^\]\"]*)\]")


# ---------------------------------------------------------------------------
# Per-run parsing
# ---------------------------------------------------------------------------

def llm_counters(obj, acc):
    if isinstance(obj, dict):
        for k in ("llm_calls", "llm_tokens", "llm_inputs", "llm_outputs"):
            if isinstance(obj.get(k), (int, float)):
                acc[k] = acc.get(k, 0) + obj[k]
        for v in obj.values():
            llm_counters(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            llm_counters(v, acc)


def parse_profile(path: Path) -> dict:
    out = {"latency_s": None, "op_calls": None, "input_tokens": None, "output_tokens": None,
           "tokens": None, "decomp_note": ""}
    if not path.exists():
        return out
    text = path.read_text(errors="replace")
    out["decomp_note"] = "; ".join(sorted(set(NOTE_RE.findall(text))))
    try:
        prof = json.loads(text)
    except json.JSONDecodeError:
        return out
    out["latency_s"] = prof.get("latency")
    acc = {}
    llm_counters(prof, acc)
    out["op_calls"] = acc.get("llm_calls")
    inp, outp, tot = acc.get("llm_inputs"), acc.get("llm_outputs"), acc.get("llm_tokens")
    # Billed output = total - input when the output count leaves out thinking tokens (Gemini)
    if inp is not None and tot is not None and (outp is None or tot > inp + outp):
        outp = tot - inp
    out.update(input_tokens=inp, output_tokens=outp, tokens=tot)
    return out


def parse_log(path: Path) -> dict:
    out = {"log_calls": 0, "log_tokens": 0, "planning_calls": 0, "rewrite": False, "rewrite_skipped": "", "task1": "",
           "task2": "", "theta": "", "guard_decision": "", "guard_reason": "", "unk_build": None,
           "unk_probe": None, "overlap": None, "selectivity": None, "guard_build_rows": None,
           "guard_probe_rows": None}
    if not path.exists():
        return out
    text = path.read_text(errors="replace")
    out["log_calls"] = sum(int(n) for n in re.findall(r"Total calls: (\d+)", text))
    out["log_tokens"] = sum(int(n) for n in re.findall(r"Total tokens: (\d+)", text))
    if "[SplitPredictJoin] Plan rewrite" in text:
        out["rewrite"], out["planning_calls"] = True, 1
        for key, label in (("task1", "Decomposed task1 prompt"), ("task2", "Decomposed task2 prompt"),
                           ("theta", "theta")):
            m = re.search(rf"{label}\s*:\s*(.*)", text)
            out[key] = m.group(1).strip() if m else ""
    m = re.search(r"\[SplitPredictJoin\] Skipping rewrite, ([^:\n]*)", text)
    if m:
        out["rewrite_skipped"], out["planning_calls"] = m.group(1).strip(), 1
    m = GUARD_RE.search(text)
    if m:
        out["guard_decision"], out["guard_reason"] = m.group("decision"), m.group("reason")
        for k in ("unk_build", "unk_probe", "overlap", "selectivity"):
            out[k] = float(m.group(k)) if m.group(k) is not None else None
        for k in ("guard_build_rows", "guard_probe_rows"):
            out[k] = int(m.group(k))
    return out


def read_pairs(path: Path, cols: tuple[int, int]) -> set | None:
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.reader(f))[1:]
    return {frozenset((str(r[cols[0]]).strip(), str(r[cols[1]]).strip())) for r in rows if len(r) > max(cols)}


def prf(pred: set, truth: set):
    if not pred and not truth:
        return 1.0, 1.0, 1.0
    if not pred or not truth:
        return 0.0, 0.0, 0.0
    tp = len(pred & truth)
    p, r = tp / len(pred), tp / len(truth)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def choice_of(base_arm: str, log: dict) -> str:
    if base_arm in ("pairwise", "manual"):
        return base_arm
    if log["guard_decision"] == "fallback":
        return "fallback"
    return "decomposed" if log["rewrite"] else "pairwise"


def collect(results: Path, queries: dict, prices: dict, models: list | None) -> list[dict]:
    gts = {}
    rows = []
    for meta_path in sorted(results.glob("*.meta.json")):
        s = meta_path.name[:-len(".meta.json")]
        m = STEM_RE.match(s)
        if not m or m["query"] not in queries or (models and m["model"] not in models):
            continue
        q = queries[m["query"]]
        meta = json.loads(meta_path.read_text())
        base_arm = meta.get("base_arm", m["arm"].split("+")[0])
        row = {"query": q.id, "arm": m["arm"], "base_arm": base_arm, "model": m["model"], "run": int(m["run"]),
               "limit": q.limit if q.limit is not None else "",
               "status": "timeout" if meta.get("timed_out") else ("failed" if meta.get("failed") else "ok"),
               "wall_s": meta.get("wall_s")}
        prof = parse_profile(results / f"{s}.json")
        log = parse_log(results / f"{s}.log")
        row.update(prof)
        row.update(log)
        # iPDB swallows failed LLM calls (e.g. an empty or timed-out response in the pairwise join) and still
        # returns a result; LLM calls without a single token mean the answer is not the model's.
        if row["status"] == "ok" and log["log_calls"] > 0 and log["log_tokens"] == 0:
            row["status"] = "llm_failed"
        op_calls = prof["op_calls"] if prof["op_calls"] is not None else log["log_calls"]
        row["calls"] = (op_calls or 0) + log["planning_calls"]
        price = prices.get(m["model"])
        row["cost_usd"] = (prof["input_tokens"] * price["input"] / 1e6 + prof["output_tokens"] * price["output"] / 1e6
                           if price and prof["input_tokens"] is not None and prof["output_tokens"] is not None
                           else None)
        row["choice"] = choice_of(base_arm, log)
        row.update(precision=None, recall=None, f1=None, quality=None, metric="P@k" if q.limit else "F1",
                   pred_pairs=None, ref_f1=None)
        if row["status"] == "ok":
            pred = read_pairs(results / f"{s}.csv", q.pair_columns)
            if q.id not in gts:
                gts[q.id] = read_pairs(GT_DIR / f"{q.id}.csv", (0, 1))
            if pred is not None:
                row["pred_pairs"] = len(pred)
                row["_pred"] = pred
                if gts[q.id] is not None:
                    p, r, f = prf(pred, gts[q.id])
                    row.update(precision=p, recall=r, f1=f, quality=p if q.limit else f)
        rows.append(row)
    # Reference F1: against the pairwise arm (same model; same run if there is one, else its first run)
    pairwise = defaultdict(dict)
    for r in rows:
        if r["arm"] == "pairwise" and "_pred" in r:
            pairwise[(r["query"], r["model"])][r["run"]] = r["_pred"]
    for r in rows:
        ref = pairwise.get((r["query"], r["model"]), {})
        if "_pred" in r and ref and not queries[r["query"]].limit:
            r["ref_f1"] = prf(r["_pred"], ref.get(r["run"], ref[min(ref)]))[2]
    for r in rows:
        r.pop("_pred", None)
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

NUMERIC = ["wall_s", "latency_s", "calls", "planning_calls", "input_tokens", "output_tokens", "tokens", "cost_usd",
           "precision", "recall", "f1", "quality", "ref_f1", "pred_pairs"]


def mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def summarise(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["query"], r["model"], r["arm"])].append(r)
    out = []
    for (q, model, arm), rs in sorted(groups.items()):
        ok = [r for r in rs if r["status"] == "ok"]
        s = {"query": q, "model": model, "arm": arm, "base_arm": rs[0]["base_arm"], "metric": rs[0]["metric"],
             "runs": len(rs), "ok": len(ok),
             "choices": ", ".join(f"{c} x{n}" for c, n in Counter(r["choice"] for r in ok).most_common())}
        for k in NUMERIC:
            s[k] = mean(r[k] for r in ok)
        out.append(s)
    return out


def cost_key(a, b):
    return "cost_usd" if a.get("cost_usd") is not None and b.get("cost_usd") is not None else "calls"


def decisions(summary, rows, delta):
    by = {(s["query"], s["model"], s["arm"]): s for s in summary}
    out = []
    for (q, model) in sorted({(s["query"], s["model"]) for s in summary}):
        pw, dec = by.get((q, model, "pairwise")), by.get((q, model, "decomposed"))
        if not pw or not dec or not pw["ok"] or not dec["ok"]:
            continue
        key = cost_key(pw, dec)
        qp, qd = pw["quality"], dec["quality"]
        acceptable = qp is not None and qd is not None and qd >= qp - delta
        oracle = "decomposed" if acceptable and (dec[key] or 0) < (pw[key] or 0) else "pairwise"
        orc = dec if oracle == "decomposed" else pw
        for s in summary:
            if s["query"] != q or s["model"] != model or s["base_arm"] not in ("auto", "guarded") or not s["ok"]:
                continue
            arm_runs = [r for r in rows if r["query"] == q and r["model"] == model and r["arm"] == s["arm"]
                        and r["status"] == "ok"]
            effective = Counter("decomposed" if r["choice"] == "decomposed" else "pairwise" for r in arm_runs)
            wrong = sum(n for c, n in effective.items() if c != oracle)
            out.append({
                "query": q, "model": model, "arm": s["arm"], "cost_metric": key,
                "pairwise_quality": qp, "decomposed_quality": qd, "pairwise_cost": pw[key], "decomposed_cost": dec[key],
                "oracle": oracle, "arm_choices": s["choices"], "wrong_runs": wrong, "runs": len(arm_runs),
                "arm_cost": s[key], "arm_quality": s["quality"],
                "cost_regret": None if s[key] is None or orc[key] is None else s[key] - orc[key],
                "quality_regret": None if s["quality"] is None or orc["quality"] is None
                else orc["quality"] - s["quality"]})
    return out


def wasted(rows, summary):
    """Fallback runs: what the decomposition attempt cost on top of running pairwise directly."""
    pw = {(s["query"], s["model"]): s for s in summary if s["arm"] == "pairwise" and s["ok"]}
    out = []
    for r in rows:
        if r["choice"] != "fallback" or r["status"] != "ok":
            continue
        ref = pw.get((r["query"], r["model"]))
        out.append({"query": r["query"], "model": r["model"], "arm": r["arm"], "run": r["run"],
                    "guard_reason": r["guard_reason"], "calls": r["calls"],
                    "pairwise_calls": ref["calls"] if ref else None,
                    "wasted_calls": r["calls"] - ref["calls"] if ref and ref["calls"] is not None else None,
                    "cost_usd": r["cost_usd"], "pairwise_cost_usd": ref["cost_usd"] if ref else None,
                    "wasted_usd": r["cost_usd"] - ref["cost_usd"]
                    if ref and r["cost_usd"] is not None and ref["cost_usd"] is not None else None,
                    "latency_s": r["latency_s"], "pairwise_latency_s": ref["latency_s"] if ref else None})
    return out


def step0(rows, summary, delta):
    """T2 calibration: the runtime check's signals and decision vs. whether decomposition really loses quality."""
    by = {(s["query"], s["model"], s["arm"]): s for s in summary}
    out = []
    for r in rows:
        if r["base_arm"] not in ("guarded", "auto") or not r["guard_decision"]:
            continue
        pw, dec = by.get((r["query"], r["model"], "pairwise")), by.get((r["query"], r["model"], "decomposed"))
        drop = (dec["quality"] - pw["quality"]) if pw and dec and pw["quality"] is not None \
            and dec["quality"] is not None else None
        loses = None if drop is None else drop < -delta
        fallback = r["guard_decision"] == "fallback"
        outcome = "" if loses is None else {(True, True): "correct fallback", (False, False): "correct decompose",
                                            (True, False): "missed loss", (False, True): "unneeded fallback"}[(loses, fallback)]
        out.append({k: r[k] for k in ("query", "model", "arm", "run", "guard_decision", "guard_reason", "unk_build",
                                      "unk_probe", "overlap", "selectivity", "guard_build_rows", "guard_probe_rows",
                                      "theta")}
                   | {"decomposed_minus_pairwise_quality": drop, "decomposition_loses_quality": loses,
                      "outcome": outcome})
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def fmt(v, kind=""):
    if v is None or v == "":
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if kind == "usd":
            return f"${v:.4f}"
        if kind == "int":
            return f"{v:,.0f}"
        if kind == "s":
            return f"{v:.1f}"
        return f"{v:.3f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v).replace("|", "\\|")


def md_table(rows, cols):
    """cols: list of (header, key, kind)"""
    lines = ["| " + " | ".join(h for h, _, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(k), kind) for _, k, kind in cols) + " |")
    return "\n".join(lines)


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


ARM_ORDER = {"pairwise": 0, "decomposed": 1, "auto": 2, "guarded": 3, "manual": 4}


def report(rows, summary, dec, waste, s0, queries, delta, out_dir, results_dir):
    md = ["# Join decomposition evaluation", "",
          f"Generated by `join_evaluation/analyze.py` from `{results_dir}` "
          f"({len(rows)} runs). Quality is F1 against ground truth, or precision@k (P@k) for LIMIT queries; "
          f"reference F1 is agreement with the pairwise arm. The oracle tolerates a quality drop of {delta}. "
          "Planning-call tokens are not reported by iPDB, so they are missing from tokens and cost "
          "(the call itself is counted in `calls`).", ""]
    failed = [r for r in rows if r["status"] != "ok"]
    if failed:
        md += ["**Failed or timed-out runs** (excluded from the means): " +
               ", ".join(f"{r['query']}/{r['arm']}/run {r['run']} ({r['status']})" for r in failed), ""]

    md += ["## Per query and arm", ""]
    for q in queries.values():
        qs = sorted((s for s in summary if s["query"] == q.id), key=lambda s: (s["model"], ARM_ORDER.get(s["base_arm"], 9), s["arm"]))
        if not qs:
            continue
        md += [f"### {q.id}", "", q.meta.get("description", ""), ""]
        md += [md_table(qs, [("model", "model", ""), ("arm", "arm", ""), ("runs ok", "ok", ""),
                             ("executed as", "choices", ""), ("latency (s)", "latency_s", "s"),
                             ("LLM calls", "calls", "int"), ("tokens", "tokens", "int"), ("cost", "cost_usd", "usd"),
                             (qs[0]["metric"], "quality", ""), ("ref F1", "ref_f1", ""),
                             ("result pairs", "pred_pairs", "int")]), ""]

    md += ["## Decisions vs. the oracle (E5)", "",
           "`wrong` counts runs whose executed plan (a fallback counts as pairwise) differs from the oracle's. "
           "Cost regret is the arm's mean cost minus the oracle's; quality regret is the oracle's quality minus "
           "the arm's.", ""]
    md += [md_table(dec, [("query", "query", ""), ("model", "model", ""), ("arm", "arm", ""),
                          ("pairwise q", "pairwise_quality", ""), ("decomposed q", "decomposed_quality", ""),
                          ("pairwise cost", "pairwise_cost", ""), ("decomposed cost", "decomposed_cost", ""),
                          ("cost in", "cost_metric", ""), ("oracle", "oracle", ""), ("arm executed", "arm_choices", ""),
                          ("wrong", "wrong_runs", ""), ("cost regret", "cost_regret", ""),
                          ("quality regret", "quality_regret", "")]) if dec else "_Needs the pairwise and decomposed arms._", ""]

    md += ["## Wasted fallback cost", "",
           "Runs where the runtime check fell back to pairwise, against running pairwise directly.", ""]
    md += [md_table(waste, [("query", "query", ""), ("model", "model", ""), ("arm", "arm", ""), ("run", "run", ""),
                            ("reason", "guard_reason", ""), ("calls", "calls", "int"),
                            ("pairwise calls", "pairwise_calls", "int"), ("wasted calls", "wasted_calls", "int"),
                            ("wasted cost", "wasted_usd", "usd"), ("latency (s)", "latency_s", "s"),
                            ("pairwise latency (s)", "pairwise_latency_s", "s")]) if waste else "_No fallbacks._", ""]

    md += ["## Step 0: runtime-check signals vs. actual quality loss", "",
           f"One row per run with a runtime check (guarded, and auto when T1 chose to decompose). "
           f"`loses quality` means the decomposed arm's quality is more than {delta} below pairwise. Current "
           "thresholds: fall back if unknown rate > 0.5, overlap < 0.2 (`=` only), identity ratio >= 0.95, or "
           "selectivity > 0.5.", ""]
    md += [md_table(s0, [("query", "query", ""), ("model", "model", ""), ("arm", "arm", ""), ("run", "run", ""),
                         ("theta", "theta", ""), ("unknown build", "unk_build", ""), ("unknown probe", "unk_probe", ""),
                         ("overlap", "overlap", ""), ("selectivity", "selectivity", ""),
                         ("decision", "guard_decision", ""), ("reason", "guard_reason", ""),
                         ("dec - pair quality", "decomposed_minus_pairwise_quality", ""),
                         ("loses quality", "decomposition_loses_quality", ""), ("outcome", "outcome", "")])
           if s0 else "_No runtime checks yet (run the guarded arm)._", ""]

    rewrites = [r for r in rows if r["rewrite"] and r["status"] == "ok"]
    md += ["## Rewrites produced by the planning call", ""]
    md += [md_table(sorted(rewrites, key=lambda r: (r["query"], r["model"], r["arm"], r["run"])),
                    [("query", "query", ""), ("model", "model", ""), ("arm", "arm", ""), ("run", "run", ""),
                     ("theta", "theta", ""), ("task 1", "task1", ""), ("task 2", "task2", "")])
           if rewrites else "_None yet._", ""]
    skipped = [r for r in rows if r["rewrite_skipped"]]
    if skipped:
        md += ["Planning calls without a usable rewrite (the join ran pairwise): " +
               ", ".join(f"{r['query']}/{r['arm']}/run {r['run']} ({r['rewrite_skipped']})" for r in skipped), ""]
    (out_dir / "report.md").write_text("\n".join(md) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out", type=Path, default=ANALYSIS_DIR)
    ap.add_argument("--models", nargs="*", help="only these models")
    ap.add_argument("--delta", type=float, default=0.05, help="quality drop the oracle tolerates (default 0.05)")
    args = ap.parse_args()

    queries = {q.id: q for q in load_queries()}
    prices = {k: v for k, v in json.loads((HERE / "prices.json").read_text()).items() if not k.startswith("_")}
    missing_gt = [q for q in queries if not (GT_DIR / f"{q}.csv").exists()]
    if missing_gt:
        print(f"[warn] no ground truth for {missing_gt}: run make_ground_truth.py (quality left empty)", file=sys.stderr)
    rows = collect(args.results, queries, prices, args.models)
    if not rows:
        sys.exit(f"No runs found in {args.results}")
    args.out.mkdir(parents=True, exist_ok=True)
    summary = summarise(rows)
    dec = decisions(summary, rows, args.delta)
    waste = wasted(rows, summary)
    s0 = step0(rows, summary, args.delta)
    for name, data in (("runs", rows), ("summary", summary), ("decisions", dec), ("step0", s0), ("wasted", waste)):
        write_csv(args.out / f"{name}.csv", data)
    report(rows, summary, dec, waste, s0, queries, args.delta, args.out, args.results)
    unpriced = sorted({r["model"] for r in rows} - set(prices))
    if unpriced:
        print(f"[info] no price for {unpriced} in prices.json: cost left empty, the oracle compares LLM calls")
    print(f"{len(rows)} runs ({sum(r['status'] == 'ok' for r in rows)} ok) -> {args.out}/report.md")


if __name__ == "__main__":
    main()
