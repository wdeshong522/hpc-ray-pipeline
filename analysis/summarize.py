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
tags and n_scenarios. startup_s is a span over startup records only, so
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


WORKER_PREFIXES = ("rank", "replica")


def is_worker(record):
    return record["tag"].startswith(WORKER_PREFIXES)


def workload(records):
    """Items processed in the run: examples (training) or scenarios (generation)."""
    for r in records:
        for key in ("n_examples", "n_scenarios"):
            if key in r["params"]:
                return r["params"][key]
    raise ValueError("no workload size in params")


def gpu_count(records):
    for r in records:
        if "gpus" in r["params"] and r["stage"] == "generation":
            return r["params"]["gpus"]
    return len({r["tag"] for r in records if r["tag"].startswith("rank")})


def summarize_run(records):
    startup = [r for r in records if r["phase"] == "startup"]
    barrier = [r for r in records if r["phase"] == "barrier"]
    processing = [r for r in records if r["phase"] == "processing"]
    proc_durations = [r["duration_s"] for r in processing if is_worker(r)]
    n_items = workload(records)
    startup_s = span(startup)
    processing_s = span(processing)
    return {
        "run_id": records[0]["run_id"],
        "stage": records[0]["stage"],
        "gpus": gpu_count(records),
        "n_items": n_items,
        "makespan_s": span(records),
        "startup_s": startup_s,
        "barrier_s": max((r["duration_s"] for r in barrier), default=0.0),
        "processing_s": processing_s,
        "startup_share": startup_s / (startup_s + processing_s),
        "items_per_s": n_items / processing_s if processing_s else None,
        "spread_s": (max(proc_durations) - min(proc_durations)
                     if proc_durations else 0.0),
        "completed": all(r["completed"] for r in records),
    }


def summarize_all(records):
    by_run = defaultdict(list)
    for r in records:
        by_run[r["run_id"]].append(r)
    rows = [summarize_run(rs) for rs in by_run.values()]
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


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
