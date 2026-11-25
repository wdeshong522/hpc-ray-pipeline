"""Run from the repo root:  python tests/test_summarize.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.summarize import summarize_run, summarize_all  # noqa: E402


def rec(run, tag, phase, start, end, n=100, ok=True):
    return {"run_id": run, "stage": "lora", "tag": tag, "phase": phase,
            "start_unix": start, "end_unix": end, "duration_s": end - start,
            "completed": ok, "params": {"n_examples": n}}


def two_gpu_run(run="lora_g2_n100_rep0", ok=True):
    return [
        rec(run, "driver", "startup", 0.0, 50.0),
        rec(run, "rank0", "startup", 50.0, 250.0),
        rec(run, "rank1", "startup", 50.0, 250.0),
        rec(run, "rank0", "processing", 250.0, 350.0, ok=ok),
        rec(run, "rank1", "processing", 250.0, 352.0, ok=ok),
    ]


def test_spans_and_shares():
    s = summarize_run(two_gpu_run())
    assert s["gpus"] == 2
    assert s["startup_s"] == 250.0
    assert s["processing_s"] == 102.0
    assert s["makespan_s"] == 352.0
    assert abs(s["startup_share"] - 250.0 / 352.0) < 1e-9
    assert abs(s["examples_per_s"] - 100 / 102.0) < 1e-9
    assert s["rank_spread_s"] == 2.0
    assert s["completed"] is True


def test_incomplete_run_is_flagged():
    assert summarize_run(two_gpu_run(ok=False))["completed"] is False


def test_runs_are_separated_and_sorted_by_gpus():
    one = [rec("lora_g1_n10_rep0", "driver", "startup", 0, 10, n=10),
           rec("lora_g1_n10_rep0", "rank0", "startup", 10, 30, n=10),
           rec("lora_g1_n10_rep0", "rank0", "processing", 30, 40, n=10)]
    rows = summarize_all(two_gpu_run() + one)
    assert [r["gpus"] for r in rows] == [1, 2]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("summarize tests passed")
