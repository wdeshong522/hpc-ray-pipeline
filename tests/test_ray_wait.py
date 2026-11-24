"""Run from the repo root:  python tests/test_ray_wait.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.ray_wait import wait_for_gpus  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_ready_immediately():
    c = FakeClock()
    assert wait_for_gpus(4, 60, lambda: 4, sleep=c.sleep, clock=c.now)


def test_ready_after_nodes_join():
    c = FakeClock()
    counts = iter([0, 4, 8])
    assert wait_for_gpus(8, 60, lambda: next(counts),
                         sleep=c.sleep, clock=c.now)


def test_times_out_when_gpus_never_arrive():
    c = FakeClock()
    assert not wait_for_gpus(8, 30, lambda: 4, sleep=c.sleep, clock=c.now)
    assert c.t >= 30


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ray_wait tests passed")
