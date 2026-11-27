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
    assert abs(s["items_per_s"] - 100 / 102.0) < 1e-9
    assert s["spread_s"] == 2.0
    assert s["completed"] is True


def gen_rec(run, tag, phase, start, end, replicas=2, n=100):
    return {"run_id": run, "stage": "generation", "tag": tag, "phase": phase,
            "start_unix": start, "end_unix": end, "duration_s": end - start,
            "completed": True,
            "params": {"replicas": replicas, "gpus": 2 * replicas,
                       "n_scenarios": n}}


def test_generation_run_barrier_is_not_startup_or_processing():
    run = "gen_r2_n100_rep1"
    recs = [
        gen_rec(run, "replica0", "startup", 0.0, 130.0),
        gen_rec(run, "replica0", "barrier", 130.0, 290.0),
        gen_rec(run, "replica1", "startup", 0.0, 290.0),
        gen_rec(run, "replica1", "barrier", 290.0, 290.0),
        gen_rec(run, "replica0", "processing", 290.0, 337.0),
        gen_rec(run, "replica1", "processing", 290.0, 333.0),
    ]
    s = summarize_run(recs)
    assert s["gpus"] == 4 and s["n_items"] == 100
    assert s["startup_s"] == 290.0
    assert s["barrier_s"] == 160.0
    assert s["processing_s"] == 47.0
    assert s["spread_s"] == 4.0


def test_merge_run_is_skipped():
    merge = [{"run_id": "m", "stage": "merge", "tag": "merge",
              "phase": "processing", "start_unix": 0.0, "end_unix": 351.0,
              "duration_s": 351.0, "completed": True, "params": {}}]
    assert summarize_all(two_gpu_run() + merge)[0]["run_id"] == "lora_g2_n100_rep0"
    assert len(summarize_all(two_gpu_run() + merge)) == 1


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


def judge_rec(run, tag, phase, start, end, mode="parallel", gpus=4, n=100):
    return {"run_id": run, "stage": "judging", "tag": tag, "phase": phase,
            "start_unix": start, "end_unix": end, "duration_s": end - start,
            "completed": True,
            "params": {"gpus": gpus, "n_responses": n, "mode": mode}}


def test_judging_parallel_takes_slowest_task_and_idle_share():
    run = "judge_u1_n100_rep1"
    recs = [judge_rec(run, "qwen0", "startup", 0, 300), judge_rec(run, "granite0", "startup", 0, 360),
            judge_rec(run, "qwen0", "barrier", 300, 360), judge_rec(run, "granite0", "barrier", 360, 360),
            judge_rec(run, "qwen0", "processing", 360, 460), judge_rec(run, "granite0", "processing", 360, 480)]
    s = summarize_run(recs)
    assert s["gpus"] == 4 and s["n_items"] == 100
    assert s["startup_s"] == 360 and s["barrier_s"] == 60 and s["processing_s"] == 120
    assert s["qwen_s"] == 100 and s["granite_s"] == 120
    assert abs(s["idle_share"] - 20 / 120) < 1e-9
    assert s["spread_s"] == 20


def test_judging_serial_adds_both_judges():
    run = "judge_serial_n100_rep1"
    recs = [judge_rec(run, "qwen0", "startup", 0, 300, "serial", 2),
            judge_rec(run, "qwen0", "processing", 300, 400, "serial", 2),
            judge_rec(run, "granite0", "startup", 400, 700, "serial", 2),
            judge_rec(run, "granite0", "processing", 700, 820, "serial", 2)]
    s = summarize_run(recs)
    assert s["gpus"] == 2
    assert s["startup_s"] == 600 and s["processing_s"] == 220
    assert s["idle_share"] is None and s["spread_s"] == 0.0
