"""
timing.py
=========
Records wall-clock intervals for pipeline stages, so every scaling run
produces the raw numbers the proposal's analysis needs.

Proposal elements served
------------------------
- Scaling Experiments: "Startup time (model loading and server
  initialization) is measured separately from processing time."
- Analysis Plan: runtime, and "start up time as a share of the total time".

Design (deliberately small)
---------------------------
- Three phases exist: "startup" and "processing" (the two the proposal
  names) and "barrier" (time a ready replica waits for the others, D34).
  Anything else raises an error, so we can't quietly invent categories.
- Each interval is appended as ONE JSON line to a file. Appending a line is
  safe even when several processes (e.g. several replicas) write at once,
  as long as each process writes to its own file, which `tag` is for.
- Nothing here depends on Ray, vLLM, or torch, so it works in any stage.
- Analysis later reads these files; this module never computes speedup.

Usage
-----
    rec = RunRecorder(results_dir="results", run_id="gen_strong_r2_rep1",
                      stage="generation", tag="replica0",
                      params={"replicas": 2, "pp_size": 2})
    with rec.interval("startup"):
        start_vllm_server()
    with rec.interval("processing"):
        generate_all()
    print(rec.summary())
"""

import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path

ALLOWED_PHASES = ("startup", "barrier", "processing")


def _slurm_context() -> dict:
    """Pulls SLURM identity from the environment, if running under SLURM.

    Returns None values (not errors) outside SLURM, so a laptop test works.
    SLURM_JOB_NODELIST is kept as a string; we don't expand it here.
    """
    keys = {
        "slurm_job_id": "SLURM_JOB_ID",
        "slurm_job_name": "SLURM_JOB_NAME",
        "slurm_nodelist": "SLURM_JOB_NODELIST",
        "slurm_cpus_per_task": "SLURM_CPUS_PER_TASK",
    }
    return {name: os.environ.get(var) for name, var in keys.items()}


class RunRecorder:
    def __init__(self, results_dir, run_id, stage, tag="main", params=None):
        """
        results_dir : root folder for raw results.
        run_id      : unique name for ONE run of ONE configuration, including
                      the repeat number (e.g. "lora_strong_g4_rep2"). The
                      run manifest (later) is what generates these.
        stage       : "lora", "generation", "judging", "pipeline", ...
        tag         : distinguishes writers within a run (e.g. "replica3").
                      Each (run_id, stage, tag) gets its own file.
        params      : the configuration under test (GPUs, replicas, batch
                      size ...). Stored on every record so a result file
                      explains itself without the manifest.
        """
        self.run_id = run_id
        self.stage = stage
        self.tag = tag
        self.params = dict(params or {})
        out_dir = Path(results_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"{stage}__{tag}.jsonl"
        self._durations = {phase: 0.0 for phase in ALLOWED_PHASES}

    @contextmanager
    def interval(self, phase: str):
        """Times the `with` block and appends one record when it ends.

        The record is written even if the block raises, with
        "completed": false, because a failed run's timing is still useful
        when diagnosing it, and we must not mistake it for a good run.
        """
        if phase not in ALLOWED_PHASES:
            raise ValueError(
                f"phase must be one of {ALLOWED_PHASES}, got {phase!r}")
        start_unix = time.time()          # wall clock: comparable across nodes
        start_perf = time.perf_counter()  # monotonic: accurate duration
        completed = False
        try:
            yield
            completed = True
        finally:
            self._append(phase, start_unix, time.time(),
                         time.perf_counter() - start_perf, completed)

    def add_interval(self, phase: str, start_unix: float, end_unix: float,
                     completed: bool = True):
        """Records an interval whose start and end were measured elsewhere.

        Used when a span cannot be wrapped in a `with` block, for example
        cluster start plus worker launch, which begins in the SLURM script
        and ends inside another process. Both times must be wall-clock
        (time.time()) values so they compare across nodes.
        """
        if phase not in ALLOWED_PHASES:
            raise ValueError(
                f"phase must be one of {ALLOWED_PHASES}, got {phase!r}")
        if end_unix < start_unix:
            raise ValueError("end_unix is before start_unix")
        self._append(phase, start_unix, end_unix, end_unix - start_unix,
                     completed)

    def _append(self, phase, start_unix, end_unix, duration, completed):
        self._durations[phase] += duration
        record = {
            "run_id": self.run_id,
            "stage": self.stage,
            "tag": self.tag,
            "phase": phase,
            "start_unix": start_unix,
            "end_unix": end_unix,
            "duration_s": duration,
            "completed": completed,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "params": self.params,
            **_slurm_context(),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def summary(self) -> dict:
        """Totals per phase for THIS recorder instance, plus startup share."""
        total = sum(self._durations.values())
        return {
            **self._durations,
            "total": total,
            "startup_share": (self._durations["startup"] / total) if total else None,
        }


def load_records(results_dir) -> list:
    """Reads every *.jsonl under results_dir into a list of dicts.

    The analysis stage will start from this. Kept here so the writer and
    reader of the format live in one file.
    """
    records = []
    for path in sorted(Path(results_dir).rglob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return records
