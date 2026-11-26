"""Run from the repo root:  python tests/test_timing.py

Plain asserts, no test framework, so it runs anywhere Python runs.
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.timing import RunRecorder, load_records  # noqa: E402


def test_two_phases_recorded_and_separate():
    with tempfile.TemporaryDirectory() as d:
        rec = RunRecorder(d, "toy_rep1", "generation", tag="replica0",
                          params={"replicas": 1})
        with rec.interval("startup"):
            time.sleep(0.20)
        with rec.interval("processing"):
            time.sleep(0.40)
        s = rec.summary()
        assert 0.19 < s["startup"] < 0.35, s
        assert 0.39 < s["processing"] < 0.55, s
        assert 0.25 < s["startup_share"] < 0.45, s
        records = load_records(d)
        assert [r["phase"] for r in records] == ["startup", "processing"]
        assert all(r["completed"] for r in records)
        assert records[0]["params"] == {"replicas": 1}


def test_failed_block_is_still_recorded_and_marked():
    with tempfile.TemporaryDirectory() as d:
        rec = RunRecorder(d, "toy_rep2", "lora")
        try:
            with rec.interval("processing"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        (r,) = load_records(d)
        assert r["completed"] is False


def test_unknown_phase_rejected():
    with tempfile.TemporaryDirectory() as d:
        rec = RunRecorder(d, "toy_rep3", "judging")
        try:
            with rec.interval("teardown"):
                pass
        except ValueError:
            return
        raise AssertionError("expected ValueError")


def test_add_interval_records_explicit_span():
    with tempfile.TemporaryDirectory() as d:
        rec = RunRecorder(d, "toy_rep1", "lora", tag="driver")
        rec.add_interval("startup", 100.0, 130.5)
        got = load_records(d)
        assert len(got) == 1
        assert got[0]["duration_s"] == 30.5
        assert got[0]["start_unix"] == 100.0 and got[0]["end_unix"] == 130.5
        try:
            rec.add_interval("startup", 10.0, 5.0)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for reversed span")
        try:
            rec.add_interval("loading", 1.0, 2.0)
        except ValueError:
            return
        raise AssertionError("expected ValueError for bad phase")


def test_barrier_phase_allowed():
    with tempfile.TemporaryDirectory() as d:
        rec = RunRecorder(d, "r", "generation", "replica0")
        with rec.interval("barrier"):
            pass
        assert rec.summary()["barrier"] >= 0.0



if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("timing tests passed")

