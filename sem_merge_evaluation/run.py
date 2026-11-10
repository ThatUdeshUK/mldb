#!/usr/bin/env python3
"""Run the semantic operator merging evaluation with WSIAF's semantic_filter / semantic_project operators.

Must run in WSIAF's environment (litellm, pydantic and WSIAF's `operators` / `models` packages):

    uv run --project ~/Projects/WSIAF python sem_merge_evaluation/run.py [options]

The model is --model (default gpt-6-luna); the endpoint and key come from WSIAF's .env (LLM_BASE_URL,
LLM_API_KEY), --base-url overrides the endpoint. See README.md for the workloads and strategies.
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from common import (DEFAULT_N, DEFAULT_POOL, DEFAULT_SEED, OPS, RESULTS_DIR, WORKLOADS, WSIAF_DIR, Op,
                    dataset_name, dataset_path, load_dataset, stem, strategies_for)

try:
    from dotenv import load_dotenv
    from models.models import ModelConfig, TableData
    from operators.sem_filter import semantic_filter_operator
    from operators.sem_project import semantic_project_operator
    import litellm
except ImportError as e:
    raise SystemExit(f"{e}\nRun inside WSIAF's environment: "
                     "uv run --project ~/Projects/WSIAF python sem_merge_evaluation/run.py ...")


DEFAULT_MODEL = "gpt-6-luna"  # WSIAF's .env still names o4-mini, which OpenAI has deprecated

# WSIAF's per-row operators get litellm.batch_completion's per-item errors back as exception objects:
# semantic_filter silently keeps such rows, semantic_project crashes. Make any failed item raise, so the chunk is
# retried (see _chunked) instead of a wrong result being recorded.
_batch_completion = litellm.batch_completion


def _strict_batch_completion(*args, **kwargs):
    responses = _batch_completion(*args, **kwargs)
    errors = [r for r in responses if isinstance(r, Exception)]
    if errors:
        raise RuntimeError(f"{len(errors)}/{len(responses)} LLM calls failed, first: {errors[0]!r}"[:1000])
    return responses


litellm.batch_completion = _strict_batch_completion


@dataclass
class Stage:
    name: str
    ops: list[str]
    operator: str            # WSIAF operator used: semantic_filter | semantic_project
    rows_in: int = 0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    latency_s: float = 0.0


@dataclass
class Ctx:
    cfg: "ModelConfig"
    batched: bool            # WSIAF row_marshal: several rows per prompt instead of one call per row
    chunk_size: int
    parallel: int
    max_retries: int
    stages: list[Stage] = field(default_factory=list)


def to_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "yes", "1"):
        return True
    if isinstance(v, str) and v.strip().lower() in ("false", "no", "0"):
        return False
    return None


def project_instruction(ops: list[Op]) -> str:
    """Instruction of a (multi-)output semantic project; one bullet per extracted output."""
    lines = [f"- {op.output} ({op.out_type}): {op.instruction}" for op in ops]
    return "Extract the following from the row:\n" + "\n".join(lines)


def columns_of(ops: list[Op]) -> list[str]:
    return list(dict.fromkeys(op.column for op in ops))


def _chunked(ctx: Ctx, stage: Stage, rows: list[dict], columns: list[str], call):
    """Run `call(table) -> OperatorResult` over row chunks (concurrently), retrying failed chunks.

    Returns the list of (chunk_rows, OperatorResult) in input order."""
    chunks = [rows[i:i + ctx.chunk_size] for i in range(0, len(rows), ctx.chunk_size)]

    def one(chunk):
        table = TableData(attributes=columns, ids=[str(r["id"]) for r in chunk],
                          data=[[r[c] for c in columns] for r in chunk])
        for attempt in range(ctx.max_retries + 1):
            try:
                return chunk, call(table), attempt
            except Exception as e:  # a failed LLM response surfaces as an exception inside the operator
                if attempt == ctx.max_retries:
                    raise RuntimeError(f"stage {stage.name}: chunk failed after {attempt + 1} attempts: {e!r}")

    stage.rows_in = len(rows)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, ctx.parallel)) as pool:
        done = list(pool.map(one, chunks))
    stage.latency_s = time.perf_counter() - t0
    for _, res, attempt in done:
        stage.calls += res.usage.calls
        stage.prompt_tokens += res.usage.prompt_tokens
        stage.completion_tokens += res.usage.completion_tokens
        stage.retries += attempt
    ctx.stages.append(stage)
    return [(c, r) for c, r, _ in done]


def run_filter(ctx: Ctx, op: Op, rows: list[dict], name: str) -> dict[str, bool]:
    stage = Stage(name, [op.name], "semantic_filter")
    done = _chunked(ctx, stage, rows, [op.column],
                    lambda t: semantic_filter_operator(ctx.cfg, op.instruction, t, row_marshal=ctx.batched))
    out = {}
    for chunk, res in done:
        kept = set(res.data.ids or [])
        out.update({str(r["id"]): str(r["id"]) in kept for r in chunk})
    return out


def run_project(ctx: Ctx, ops: list[Op], rows: list[dict], name: str) -> dict[str, dict]:
    """One semantic project extracting every op's output; filters' outputs are parsed to booleans."""
    stage = Stage(name, [op.name for op in ops], "semantic_project")
    schema = {op.output: op.out_type for op in ops}
    inst = project_instruction(ops)

    def call(table):
        res = semantic_project_operator(ctx.cfg, inst, table, schema, row_marshal=ctx.batched)
        if len(res.data.data) != len(table.data):
            raise ValueError(f"projection returned {len(res.data.data)} rows for {len(table.data)}")
        return res

    done = _chunked(ctx, stage, rows, columns_of(ops), call)
    out = {}
    for chunk, res in done:
        for r, vals in zip(chunk, res.data.data):
            rec = dict(zip(res.data.attributes, vals))
            out[str(r["id"])] = {op.name: (to_bool(rec.get(op.output)) if op.kind == "filter"
                                           else str(rec.get(op.output, "")).strip()) for op in ops}
    return out


def execute(ctx: Ctx, op: Op, rows: list[dict], name: str, single_output: bool) -> dict[str, object]:
    """One operator on rows -> {id: value}. Filters use semantic_filter unless single_output is set."""
    if op.kind == "filter" and not single_output:
        return run_filter(ctx, op, rows, name)
    return {k: v[op.name] for k, v in run_project(ctx, [op], rows, name).items()}


def run_strategy(ctx: Ctx, strategy: str, ops: list[Op], rows: list[dict]) -> dict[str, dict]:
    """Returns {id: {op_name: value}} for every value the strategy computed."""
    outputs = {str(r["id"]): {} for r in rows}

    def record(op, vals):
        for k, v in vals.items():
            outputs[k][op.name] = v

    def pipeline(first: Op, second: Op, single_output: bool):
        v1 = execute(ctx, first, rows, f"1:{first.name}", single_output)
        record(first, v1)
        survivors = [r for r in rows if v1.get(str(r["id"])) is True] if first.kind == "filter" else rows
        record(second, execute(ctx, second, survivors, f"2:{second.name}", single_output))

    op1, op2 = ops
    if strategy == "separate":
        pipeline(op1, op2, single_output=False)
    elif strategy == "separate_rev":
        pipeline(op2, op1, single_output=False)
    elif strategy == "separate_1out":
        pipeline(op1, op2, single_output=True)
    elif strategy == "independent":
        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(execute, ctx, op1, rows, f"a:{op1.name}", False)
            f2 = pool.submit(execute, ctx, op2, rows, f"b:{op2.name}", False)
            record(op1, f1.result())
            record(op2, f2.result())
    elif strategy == "merged":
        merged = run_project(ctx, ops, rows, f"merged:{op1.name}+{op2.name}")
        for k, vals in merged.items():
            outputs[k].update(vals)
    else:
        raise ValueError(strategy)
    return outputs


def wsiaf_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(WSIAF_DIR), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workloads", nargs="+", choices=list(WORKLOADS), default=list(WORKLOADS))
    ap.add_argument("--strategies", nargs="+", help="default: every strategy that applies to the workload")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="dataset size (default %(default)s)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--pool", choices=["rated", "all"], default=DEFAULT_POOL)
    ap.add_argument("--runs", type=int, nargs="+", default=[1])
    ap.add_argument("--model", default=DEFAULT_MODEL, help="litellm model (default %(default)s)")
    ap.add_argument("--base-url", help="default: LLM_BASE_URL from WSIAF's .env")
    ap.add_argument("--mode", choices=["row", "batch"], default="row",
                    help="row: one LLM call per row (WSIAF's default); batch: chunk-size rows per call "
                         "(WSIAF row_marshal=True)")
    ap.add_argument("--chunk-size", type=int, help="rows per operator invocation (default: 50 row, 20 batch)")
    ap.add_argument("--parallel", type=int, default=4, help="concurrent chunks per stage (default %(default)s)")
    ap.add_argument("--retries", type=int, default=2, help="retries of a failed chunk (default %(default)s)")
    ap.add_argument("--force", action="store_true", help="rerun finished runs")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the prompts, no LLM calls")
    args = ap.parse_args()

    load_dotenv(WSIAF_DIR / ".env")
    model = args.model
    cfg = ModelConfig(model=model, base_url=args.base_url or os.environ.get("LLM_BASE_URL", ""),
                      secret=os.environ.get("LLM_API_KEY", ""))

    data = dataset_name(args.n, args.seed, args.pool)
    if not dataset_path(data).exists():
        subprocess.run([sys.executable, str(dataset_path(data).parent.parent / "make_dataset.py"),
                        "--n", str(args.n), "--seed", str(args.seed), "--pool", args.pool], check=True)
    rows = load_dataset(data)
    chunk_size = args.chunk_size or (50 if args.mode == "row" else 20)

    RESULTS_DIR.mkdir(exist_ok=True)
    for workload in args.workloads:
        ops = [OPS[o] for o in WORKLOADS[workload][2]]
        strategies = [s for s in (args.strategies or strategies_for(workload)) if s in strategies_for(workload)]
        for strategy in strategies:
            for run in args.runs:
                s = stem(workload, strategy, model, args.mode, data, run)
                out = RESULTS_DIR / f"{s}.json"
                if out.exists() and not args.force:
                    print(f"skip  {s} (done)")
                    continue
                if args.dry_run:
                    print(f"plan  {s}")
                    if strategy in ("merged", "separate_1out"):
                        for group in ([ops] if strategy == "merged" else [[o] for o in ops]):
                            print(f"      semantic_project{columns_of(group)} schema="
                                  f"{ {o.output: o.out_type for o in group} }\n"
                                  + "\n".join("        " + l for l in project_instruction(group).splitlines()))
                    else:
                        for o in ops:
                            inst = o.instruction if o.kind == "filter" else project_instruction([o])
                            print(f"      semantic_{o.kind}[{o.column}] {inst}")
                    continue

                ctx = Ctx(cfg, args.mode == "batch", chunk_size, args.parallel, args.retries)
                print(f"run   {s} ...", end="", flush=True)
                t0 = time.perf_counter()
                status, error, outputs = "ok", None, {}
                try:
                    outputs = run_strategy(ctx, strategy, ops, rows)
                except Exception as e:
                    status, error = "failed", "".join(traceback.format_exception(e))[-4000:]
                wall = time.perf_counter() - t0
                result = {
                    "meta": {"workload": workload, "strategy": strategy, "ops": [o.name for o in ops],
                             "model": model, "mode": args.mode, "data": data, "n": len(rows), "run": run,
                             "chunk_size": chunk_size, "parallel": args.parallel,
                             "started": dt.datetime.now().isoformat(timespec="seconds"),
                             "wsiaf_commit": wsiaf_commit()},
                    "status": status, "error": error, "latency_s": wall,
                    "stages": [vars(st) for st in ctx.stages],
                    "outputs": outputs,
                }
                if status == "ok":
                    out.write_text(json.dumps(result, indent=1))
                else:
                    (RESULTS_DIR / f"{s}.failed.json").write_text(json.dumps(result, indent=1))
                calls = sum(st.calls for st in ctx.stages)
                toks = sum(st.prompt_tokens + st.completion_tokens for st in ctx.stages)
                print(f" {status} {wall:.1f}s, {calls} calls, {toks} tokens" + (f"\n{error}" if error else ""))


if __name__ == "__main__":
    main()
