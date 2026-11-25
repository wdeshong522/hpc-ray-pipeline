"""Summarize raw timing records, one row per run.

Proposal link: runtime, throughput, startup time as a share of total time.
Reads the files written by src/common/timing.py; computes nothing about
speedup yet (that needs several runs and comes later).

Definitions (per run, all tags together):
  startup_s    = last startup end - first startup start
  processing_s = last processing end - first processing start
  makespan_s   = last end - first start over every record
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


def summarize_run(records):
    ranks = [r for r in records if r["tag"].startswith("rank")]
    startup = [r for r in records if r["phase"] == "startup"]
    processing = [r for r in records if r["phase"] == "processing"]
    proc_durations = [r["duration_s"] for r in processing
                      if r["tag"].startswith("rank")]
    n_examples = next(r["params"]["n_examples"] for r in records
                      if "n_examples" in r["params"])
    startup_s = span(startup)
    processing_s = span(processing)
    return {
        "run_id": records[0]["run_id"],
        "gpus": len({r["tag"] for r in ranks}),
        "n_examples": n_examples,
        "makespan_s": span(records),
        "startup_s": startup_s,
        "processing_s": processing_s,
        "startup_share": startup_s / (startup_s + processing_s),
        "examples_per_s": n_examples / processing_s if processing_s else None,
        "rank_spread_s": (max(proc_durations) - min(proc_durations)
                          if proc_durations else 0.0),
        "completed": all(r["completed"] for r in records),
    }


def summarize_all(records):
    by_run = defaultdict(list)
    for r in records:
        by_run[r["run_id"]].append(r)
    rows = [summarize_run(rs) for rs in by_run.values()]
    return sorted(rows, key=lambda row: (row["gpus"], row["n_examples"],
                                         row["run_id"]))


def main(results_dir="results"):
    rows = summarize_all(load_records(results_dir))
    print(f"{'run_id':24s} {'gpus':>4s} {'n':>6s} {'makespan':>9s} "
          f"{'startup':>8s} {'process':>8s} {'start%':>6s} {'ex/s':>6s} "
          f"{'spread':>6s} ok")
    for r in rows:
        print(f"{r['run_id']:24s} {r['gpus']:4d} {r['n_examples']:6d} "
              f"{r['makespan_s']:9.1f} {r['startup_s']:8.1f} "
              f"{r['processing_s']:8.1f} {100 * r['startup_share']:6.1f} "
              f"{r['examples_per_s']:6.2f} {r['rank_spread_s']:6.2f} "
              f"{'yes' if r['completed'] else 'NO'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
