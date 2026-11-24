"""Run from the repo root:  python tests/test_launch.py

Checks the launcher's pure helpers; the Ray launch itself needs a cluster.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train.launch import (  # noqa: E402
    derive_cpus_per_worker, worker_launch_end, parse_args)


def test_cpus_split_across_workers_on_one_node():
    assert derive_cpus_per_worker(32, 1) == 32
    assert derive_cpus_per_worker(32, 4) == 8
    assert derive_cpus_per_worker(32, 2) == 16


def test_eight_gpus_use_four_workers_per_node_for_cpu_split():
    assert derive_cpus_per_worker(32, 8) == 8


def test_at_least_one_cpu():
    assert derive_cpus_per_worker(2, 4) == 1


def rec(tag, phase, start, stage="lora"):
    return {"stage": stage, "tag": tag, "phase": phase, "start_unix": start}


def test_launch_end_is_first_worker_startup():
    records = [
        rec("rank1", "startup", 105.0),
        rec("rank0", "startup", 102.0),
        rec("rank0", "processing", 500.0),
        rec("driver", "startup", 50.0),
    ]
    assert worker_launch_end(records, fallback=999.0) == 102.0


def test_launch_end_falls_back_when_no_worker_started():
    assert worker_launch_end([], fallback=7.0) == 7.0
    assert worker_launch_end([rec("driver", "startup", 1.0)], 7.0) == 7.0


def test_args_parse():
    a = parse_args(["--gpus", "4", "--n_examples", "1000", "--run_id", "x"])
    assert a.gpus == 4 and a.per_device_batch == 1 and a.grad_accum == 8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("launch tests passed")
