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


if __name__ == "__main__":
    test_two_phases_recorded_and_separate()
    test_failed_block_is_still_recorded_and_marked()
    test_unknown_phase_rejected()
    print("timing tests passed")
