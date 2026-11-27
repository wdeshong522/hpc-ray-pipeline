"""Summarize raw timing records, one row per run.

Proposal link: runtime, throughput, startup time as a share of total time.
Reads the files written by src/common/timing.py; computes nothing about
speedup yet (that needs several runs and comes later).

Definitions (per run, all tags together):
  startup_s    = last startup end - first startup start
  processing_s = last processing end - first processing start
  barrier_s    = longest barrier wait of any replica (generation only)
  makespan_s   = last end - first start over every record
Training runs have rank* tags and n_examples; generation runs have replica*
tags and n_scenarios; judging runs have qwen*/granite* tags and n_responses
(serial baseline: both judges' times add; otherwise the slowest task counts). startup_s is a span over startup records only, so
barrier waiting is never counted as startup or processing (D34).
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.timing import load_records  # noqa: E402


def span(records):
    if not records:
        return 0.0
    return (max(r["end_unix"] for r in records)
            - min(r["start_unix"] for r in records))


WORKER_PREFIXES = ("rank", "replica", "qwen", "granite")
SCALING_STAGES = ("lora", "generation", "judging")


def is_worker(record):
    return record["tag"].startswith(WORKER_PREFIXES)


def workload(records):
    """Items processed in the run: examples (training) or scenarios (generation)."""
    for r in records:
        for key in ("n_examples", "n_scenarios", "n_responses"):
            if key in r["params"]:
                return r["params"][key]
    raise ValueError("no workload size in params")


def gpu_count(records):
    for r in records:
        if "gpus" in r["params"] and r["stage"] in ("generation", "judging"):
            return r["params"]["gpus"]
    return len({r["tag"] for r in records if r["tag"].startswith("rank")})


def tag_sums(records, phase):
    """Total duration of one phase per tag (a serial judging task has two startups)."""
    sums = defaultdict(float)
    for r in records:
        if r["phase"] == phase:
            sums[r["tag"]] += r["duration_s"]
    return sums


def judging_times(records):
    """Judging run: serial adds the two judges' times, parallel takes the slowest task (D37, D41).

    Also returns per-judge processing time and the share of the slower judge's
    time the faster judge's GPUs sat idle (parallel runs only).
    """
    serial = records[0]["params"].get("mode") == "serial"
    out = {}
    for phase in ("startup", "barrier", "processing"):
        values = list(tag_sums(records, phase).values())
        out[phase] = (sum(values) if serial else max(values, default=0.0))
    by_judge = defaultdict(float)
    for tag, d in tag_sums(records, "processing").items():
        judge = "qwen" if tag.startswith("qwen") else "granite"
        by_judge[judge] = max(by_judge[judge], d)
    out["qwen_s"], out["granite_s"] = by_judge["qwen"], by_judge["granite"]
    slow = max(out["qwen_s"], out["granite_s"])
    out["idle_share"] = None if serial or not slow else (slow - min(out["qwen_s"], out["granite_s"])) / slow
    return out


def summarize_run(records):
    startup = [r for r in records if r["phase"] == "startup"]
    barrier = [r for r in records if r["phase"] == "barrier"]
    processing = [r for r in records if r["phase"] == "processing"]
    proc_durations = [r["duration_s"] for r in processing if is_worker(r)]
    n_items = workload(records)
    startup_s = span(startup)
    processing_s = span(processing)
    extra = {}
    if records[0]["stage"] == "judging":
        times = judging_times(records)
        startup_s, processing_s = times["startup"], times["processing"]
        extra = {"qwen_s": times["qwen_s"], "granite_s": times["granite_s"],
                 "idle_share": times["idle_share"]}
        barrier_s = times["barrier"]
        serial = records[0]["params"].get("mode") == "serial"
        proc_durations = [] if serial else list(tag_sums(records, "processing").values())
    else:
        barrier_s = max((r["duration_s"] for r in barrier), default=0.0)
    row = {
        "run_id": records[0]["run_id"],
        "stage": records[0]["stage"],
        "gpus": gpu_count(records),
        "n_items": n_items,
        "makespan_s": span(records),
        "startup_s": startup_s,
        "barrier_s": barrier_s,
        "processing_s": processing_s,
        "startup_share": startup_s / (startup_s + processing_s),
        "items_per_s": n_items / processing_s if processing_s else None,
        "spread_s": (max(proc_durations) - min(proc_durations)
                     if proc_durations else 0.0),
        "completed": all(r["completed"] for r in records),
    }
    row.update(extra)
    return row


def summarize_all(records):
    by_run = defaultdict(list)
    for r in records:
        by_run[r["run_id"]].append(r)
    rows = [summarize_run(rs) for rs in by_run.values()
            if rs[0]["stage"] in SCALING_STAGES]
    return sorted(rows, key=lambda row: (row["stage"], row["gpus"],
                                         row["n_items"], row["run_id"]))


def main(results_dir="results"):
    rows = summarize_all(load_records(results_dir))
    print(f"{'run_id':24s} {'gpus':>4s} {'n':>6s} {'makespan':>9s} "
          f"{'startup':>8s} {'barrier':>8s} {'process':>8s} {'start%':>6s} "
          f"{'item/s':>7s} {'spread':>6s} ok")
    for r in rows:
        print(f"{r['run_id']:24s} {r['gpus']:4d} {r['n_items']:6d} "
              f"{r['makespan_s']:9.1f} {r['startup_s']:8.1f} "
              f"{r['barrier_s']:8.1f} {r['processing_s']:8.1f} "
              f"{100 * r['startup_share']:6.1f} {r['items_per_s']:7.2f} "
              f"{r['spread_s']:6.2f} {'yes' if r['completed'] else 'NO'}")
    judged = [r for r in rows if r["stage"] == "judging"]
    if judged:
        print(f"\n{'judging run':24s} {'qwen_s':>8s} {'granite_s':>9s} {'idle%':>6s}")
        for r in judged:
            idle = "-" if r["idle_share"] is None else f"{100 * r['idle_share']:.1f}"
            print(f"{r['run_id']:24s} {r['qwen_s']:8.1f} {r['granite_s']:9.1f} {idle:>6s}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
