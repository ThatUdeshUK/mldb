#!/usr/bin/env python3
"""Run the join-decomposition evaluation on iPDB: every query x arm x run, one fresh iPDB session each.

Per run it writes results/<query>|arm:<arm>|model:<model>|run:<run>.{csv,json,log,meta.json}:
    .csv        query result
    .json       DuckDB JSON profile (latency, llm_* counters, the "[decomp: ...]" decision note)
    .log        iPDB stdout/stderr: "Total calls", the planning call's rewrite ("[SplitPredictJoin] Plan rewrite")
                and the runtime check ("[SplitPredictJoin] Runtime check: ...")
    .meta.json  arm, settings, exit code, wall time, timeout flag

Runs that already finished successfully are skipped (resume after an interruption); --force reruns them.

Examples (from the repo root):
    python3 join_evaluation/run.py --explain                      # T1's plan-time decision per query (cheap)
    python3 join_evaluation/run.py --queries pc_q5 sm_tagline     # all arms, run 1, gpt6luna
    python3 join_evaluation/run.py --model sonnet55 --runs 1 2 3
    python3 join_evaluation/run.py --arms guarded --set join_decomposition_max_unknown=0.3 --tag unk0.3
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARMS, IPDB_BIN, RESULTS_DIR, load_queries, stem  # noqa: E402


def session_sql(q, model, arm_settings, extra, csv_path, json_path, threads, batch, llm_threads):
    lines = [
        "PRAGMA enable_profiling = 'json';",
        f"SET threads = {threads};",
        f"SET ml_batch_size = {batch};",
        f"SET llm_no_threads = {llm_threads};",
        *q.settings,
        *arm_settings,
        *extra,
        f"PRAGMA profile_output = '{json_path}';",
        "COPY (",
        q.render(model),
        f") TO '{csv_path}' (HEADER, DELIMITER ',');",
        ".q",
    ]
    return "\n".join(lines) + "\n"


def run_ipdb(ipdb_bin, db, sql, log_path, timeout, cwd):
    start = time.monotonic()
    with open(log_path, "w") as log:
        try:
            proc = subprocess.run([str(ipdb_bin), "-readonly", str(db)], input=sql, text=True,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=timeout, cwd=cwd)
            rc, timed_out = proc.returncode, False
        except subprocess.TimeoutExpired:
            rc, timed_out = -1, True
    wall = time.monotonic() - start
    text = log_path.read_text(errors="replace")
    failed = rc != 0 or timed_out or "Error:" in text
    return rc, timed_out, failed, wall


def _sum_counter(log_path: Path, label: str) -> int:
    return sum(int(n) for n in re.findall(rf"{label}: (\d+)", log_path.read_text(errors="replace")))


def total_calls(log_path: Path) -> int:
    return _sum_counter(log_path, "Total calls")


def total_tokens(log_path: Path) -> int:
    return _sum_counter(log_path, "Total tokens")


def explain(args, queries):
    """EXPLAIN each query under 'auto': shows T1's estimate and decision ([decomp: ...]). If T1 decides to
    decompose, the optimizer makes its planning call (1 LLM call); nothing else is executed."""
    out_dir = args.out / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)
    for q in queries:
        if "manual" in q.arms:
            continue
        sql = "\n".join([*q.settings, *ARMS["auto"], *args.set_sql, "EXPLAIN", q.render(args.model) + ";", ".q"]) + "\n"
        log = out_dir / f"{q.id}|model:{args.model}.explain.txt"
        rc, timed_out, failed, wall = run_ipdb(args.ipdb_bin, q.db_path(), sql, log, args.timeout, out_dir)
        text = log.read_text(errors="replace")
        # EXPLAIN draws boxes and wraps long text across lines: drop the box characters and join the lines
        flat = " ".join(re.sub(r"[│┌┐└┘─┬┴├┤]", " ", text).split())
        notes = sorted({seg.split("]")[0] for seg in flat.split("[decomp: ")[1:]})
        rewrite = "planning call made" if "[SplitPredictJoin]" in text else "no planning call"
        status = "FAILED" if failed else f"{wall:.1f}s"
        print(f"{q.id:<15} {status:<8} decomp: {'; '.join(notes) or '-'}  ({rewrite})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt6luna", help="LLM model name registered in the iPDB databases")
    ap.add_argument("--queries", nargs="*", help="query ids (default: all in queries/)")
    ap.add_argument("--arms", nargs="*", help=f"arms (default: each query's arms; known: {list(ARMS)})")
    ap.add_argument("--runs", nargs="*", type=int, default=[1])
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                    help="extra iPDB settings for every run, e.g. join_decomposition_max_unknown=0.3")
    ap.add_argument("--tag", default="", help="suffix for the arm name in result files (use with --set)")
    ap.add_argument("--threads", type=int, default=1, help="SET threads (DuckDB execution threads)")
    ap.add_argument("--batch-size", type=int, default=16, help="SET ml_batch_size")
    ap.add_argument("--llm-threads", type=int, default=16, help="SET llm_no_threads")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per run before it is killed")
    ap.add_argument("--ipdb-bin", type=Path, default=IPDB_BIN)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--force", action="store_true", help="rerun runs that already succeeded")
    ap.add_argument("--explain", action="store_true", help="only EXPLAIN under 'auto' (T1 decisions)")
    ap.add_argument("--dry-run", action="store_true", help="print the session SQL instead of running it")
    args = ap.parse_args()

    if args.set and not args.tag:
        ap.error("--set changes the arm's behaviour; give it a --tag so its results don't mix with the default arm")
    args.set_sql = []
    for item in args.set:
        name, _, value = item.partition("=")
        args.set_sql.append(f"SET {name.strip()} = {value.strip()};")
    if not args.ipdb_bin.exists():
        sys.exit(f"iPDB binary not found: {args.ipdb_bin} (set --ipdb-bin or IPDB_BIN)")

    queries = load_queries(args.queries)
    for q in queries:
        if not q.db_path().exists():
            sys.exit(f"Database for {q.id} not found: {q.db_path()} (build it with evaluation/setup_ipdb_icde.sh)")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.explain:
        explain(args, queries)
        return

    print(f"model={args.model} runs={args.runs} threads={args.threads} batch={args.batch_size} "
          f"llm_threads={args.llm_threads} timeout={args.timeout}s out={args.out}")
    for run in args.runs:
        for q in queries:
            arms = [a for a in (args.arms or q.arms) if a in q.arms]
            for arm in arms:
                if arm not in ARMS:
                    sys.exit(f"Unknown arm '{arm}'. Known: {list(ARMS)}")
                arm_name = f"{arm}+{args.tag}" if args.tag else arm
                s = stem(q.id, arm_name, args.model, run)
                paths = {ext: args.out / f"{s}.{ext}" for ext in ("csv", "json", "log", "meta.json")}
                if paths["meta.json"].exists() and not args.force:
                    prev = json.loads(paths["meta.json"].read_text())
                    if not prev.get("failed"):
                        print(f"[skip] {s} (done; --force to rerun)")
                        continue
                sql = session_sql(q, args.model, ARMS[arm], args.set_sql, paths["csv"], paths["json"],
                                  args.threads, args.batch_size, args.llm_threads)
                if args.dry_run:
                    print(f"-- {s}\n{sql}")
                    continue
                for p in (paths["csv"], paths["json"]):
                    p.unlink(missing_ok=True)
                print(f"[run ] {s} ...", flush=True)
                rc, timed_out, failed, wall = run_ipdb(args.ipdb_bin, q.db_path(), sql, paths["log"],
                                                       args.timeout, args.out)
                meta = {"query": q.id, "arm": arm_name, "base_arm": arm, "model": args.model, "run": run,
                        "settings": q.settings + ARMS[arm] + args.set_sql, "threads": args.threads,
                        "batch_size": args.batch_size, "llm_threads": args.llm_threads,
                        "exit_code": rc, "timed_out": timed_out, "failed": failed, "wall_s": round(wall, 3),
                        "db": str(q.db_path()), "ipdb_bin": str(args.ipdb_bin),
                        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                paths["meta.json"].write_text(json.dumps(meta, indent=2))
                if failed:
                    why = "timeout" if timed_out else f"exit {rc}"
                    print(f"       FAILED ({why}) after {wall:.1f}s, see {paths['log']}")
                else:
                    rows = max(0, sum(1 for _ in open(paths["csv"])) - 1) if paths["csv"].exists() else 0
                    calls, tokens = total_calls(paths["log"]), total_tokens(paths["log"])
                    print(f"       done in {wall:.1f}s, {rows} rows, {calls} LLM calls, {tokens} tokens")
                    if calls and not tokens:
                        print("       WARNING: LLM calls returned no tokens (failed calls iPDB ignores); "
                              "analyze.py marks this run llm_failed")


if __name__ == "__main__":
    main()
