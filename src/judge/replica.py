"""One judging task: serve one judge on 2 GPUs, meet the other tasks, judge this task's shard."""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.timing import RunRecorder  # noqa: E402
from src.generate.replica import PP_SIZE, barrier, stop_server, wait_ready  # noqa: E402
from src.judge.inputs import load_items  # noqa: E402
from src.judge.prompts import load_prompts  # noqa: E402
from src.judge.smoke import build_calls, judge_call, kv_lines, server_command, start_server  # noqa: E402
from src.train.data import load_records, ordered_pool  # noqa: E402

MAX_LEN = 6144
MAX_ERROR_SHARE = 0.005


def plan(index, ntasks, serial):
    """Steps this task runs, as (judge, shard index, shard count): serial runs both judges in turn."""
    if serial:
        if ntasks != 1:
            raise ValueError("serial baseline uses exactly one task")
        return [("qwen", 0, 1), ("granite", 0, 1)]
    if ntasks % 2:
        raise ValueError("parallel judging needs an even number of tasks (units of 1 Qwen + 1 Granite)")
    units = ntasks // 2
    return [("qwen" if index < units else "granite", index % units, units)]


def shard_items(items, shard_index, shards):
    """Round-robin by position: response i goes to shard i % shards."""
    return [it for pos, it in enumerate(items) if pos % shards == shard_index]


def port_for(job_id, index):
    return 20000 + (job_id % 500) * 8 + index % 8


def ensure_fresh(run_dir, steps):
    for judge, k, _ in steps:
        for name in (f"judging__{judge}{k}.jsonl", f"judgments_{judge}_replica{k}.ndjson"):
            if (Path(run_dir) / name).exists():
                raise FileExistsError(f"{run_dir}/{name} already exists; run ids must be unique")


def run_calls(port, served_name, calls, ids, concurrency, out_path):
    """Send every call with `concurrency` in flight; a failed call is written as a missing score."""
    lock = threading.Lock()
    stats = {"calls": 0, "bad": 0, "errors": 0}
    with open(out_path, "w", encoding="utf-8") as fh:
        def work(call):
            index, metric, score_type, prompt = call
            try:
                out = judge_call(port, served_name, prompt, score_type)
                err = None
            except Exception as exc:
                out = {"seconds": 0.0, "finish_reason": "error", "prompt_tokens": 0,
                       "completion_tokens": 0, "ok": False, "score": None, "reasoning": ""}
                err = str(exc)
            row = {"index": index, "id": ids[index], "metric": metric, **out}
            if err:
                row["error"] = err
            with lock:
                stats["calls"] += 1
                stats["bad"] += not row["ok"]
                stats["errors"] += err is not None
                fh.write(json.dumps(row) + "\n")
                fh.flush()

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(work, calls))
    return stats


def run_step(a, judge, k, shards, model_dir, run_dir, port, items, registry, recorder, sync):
    """Startup, optional barrier, processing for one judge; the server always stops at the end."""
    mine = shard_items(items, k, shards)
    calls = build_calls(mine, registry)
    ids = {index: row["id"] for index, _, row in mine}
    proc = None
    try:
        with recorder.interval("startup"):
            proc = start_server(server_command(model_dir, judge, port, MAX_LEN),
                                run_dir / f"server_{judge}{k}.log")
            wait_ready(port, proc, a.ready_timeout)
        if sync:
            with recorder.interval("barrier"):
                barrier(a.index, a.ntasks, a.head_host, a.barrier_port, a.barrier_timeout)
        with recorder.interval("processing"):
            stats = run_calls(port, judge, calls, ids, a.concurrency,
                              run_dir / f"judgments_{judge}_replica{k}.ndjson")
            if stats["errors"] > MAX_ERROR_SHARE * max(stats["calls"], 1):
                raise RuntimeError(f"{judge}{k}: {stats['errors']} of {stats['calls']} calls raised errors")
        print(f"{judge}{k}: {len(mine)} responses, {stats['calls']} calls, "
              f"{stats['bad']} missing scores", flush=True)
        return kv_lines(run_dir / f"server_{judge}{k}.log")
    finally:
        stop_server(proc)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", required=True)
    p.add_argument("--n_responses", type=int, required=True)
    p.add_argument("--responses", required=True)
    p.add_argument("--qwen_dir", required=True)
    p.add_argument("--granite_dir", required=True)
    p.add_argument("--serial", action="store_true")
    p.add_argument("--head_host", default="localhost")
    p.add_argument("--barrier_port", type=int, default=30000)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--prompts_dir", default="configs/judge_prompts")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--index", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    p.add_argument("--ntasks", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)))
    p.add_argument("--ready_timeout", type=int, default=1200)
    p.add_argument("--barrier_timeout", type=int, default=1800)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    steps = plan(a.index, a.ntasks, a.serial)
    job_id = int(os.environ.get("SLURM_JOB_ID", 0))
    port = port_for(job_id, a.index)
    run_dir = Path(a.results_dir) / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_fresh(run_dir, steps)
    print(f"task {a.index}: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"port={port} steps={steps}", flush=True)
    records, _ = load_records(a.data_dir)
    items = load_items(a.responses, ordered_pool(records), a.n_responses)
    registry = load_prompts(a.prompts_dir)
    model_dirs = {"qwen": a.qwen_dir, "granite": a.granite_dir}
    for judge, k, shards in steps:
        recorder = RunRecorder(a.results_dir, a.run_id, "judging", f"{judge}{k}",
                               {"gpus": PP_SIZE * a.ntasks, "n_responses": a.n_responses,
                                "concurrency": a.concurrency, "judge": judge, "pp_size": PP_SIZE,
                                "mode": "serial" if a.serial else "parallel",
                                "units": 0 if a.serial else a.ntasks // 2})
        run_step(a, judge, k, shards, model_dirs[judge], run_dir, port, items, registry,
                 recorder, sync=not a.serial)


if __name__ == "__main__":
    main()
