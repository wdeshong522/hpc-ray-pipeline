"""Run from the repo root:  python tests/test_worker.py

Checks the pure configuration logic only; the Ray worker itself needs GPUs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train.worker import with_defaults, sft_kwargs  # noqa: E402

BASE = dict(model_name="m", data_dir="d", n_examples=1000, output_dir="o",
            results_dir="r", run_id="x")


def test_defaults_filled():
    cfg = with_defaults(BASE)
    assert cfg["epochs"] == 1 and cfg["lora_r"] == 16
    assert cfg["per_device_batch"] == 1 and cfg["grad_accum"] == 8


def test_tunables_can_be_overridden():
    cfg = with_defaults({**BASE, "per_device_batch": 2, "grad_accum": 4})
    assert cfg["per_device_batch"] == 2 and cfg["grad_accum"] == 4


def test_fixed_settings_cannot_be_overridden():
    try:
        with_defaults({**BASE, "lora_r": 32})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_missing_keys_raise():
    try:
        with_defaults({"model_name": "m"})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_trainer_has_no_eval_checkpoints_or_reporting():
    kw = sft_kwargs(with_defaults(BASE))
    assert kw["eval_strategy"] == "no"
    assert kw["save_strategy"] == "no"
    assert kw["report_to"] == "none"
    assert kw["fp16"] is True and kw["bf16"] is False
    assert kw["num_train_epochs"] == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("worker tests passed")
