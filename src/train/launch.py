"""Driver for Stage 1: run data-parallel LoRA on an existing Ray cluster.

Proposal link: Ray Train data-parallel training across GPUs and nodes.  The
Ray cluster itself is started by the SLURM script (slurm/lora.sbatch); this
driver connects to it, launches one worker per GPU, and records the part of
startup no worker can see: cluster start plus worker launch (DECISIONS D24).
"""
import argparse
import os
import time
from pathlib import Path

GPUS_PER_NODE = 4
REPO_ROOT = Path(__file__).resolve().parents[2]


def derive_cpus_per_worker(cpus_per_node, gpus):
    """Split a node's CPUs evenly across the workers placed on it."""
    workers_per_node = min(gpus, GPUS_PER_NODE)
    return max(1, cpus_per_node // workers_per_node)


def worker_launch_end(records, fallback):
    """Wall-clock time the first worker began its own startup."""
    starts = [r["start_unix"] for r in records
              if r["stage"] == "lora" and r["tag"].startswith("rank")
              and r["phase"] == "startup"]
    return min(starts) if starts else fallback


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=int, required=True)
    p.add_argument("--n_examples", type=int, required=True)
    p.add_argument("--run_id", required=True)
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--per_device_batch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--cpus_per_worker", type=int, default=None)
    p.add_argument("--ray_address", default="auto")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = float(os.environ.get("RUN_T0", time.time()))

    import ray
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    from src.common.timing import RunRecorder, load_records
    from src.train.worker import train_worker

    results_dir = Path(args.results_dir).resolve()
    out_dir = results_dir / args.run_id
    cpus = args.cpus_per_worker or derive_cpus_per_worker(
        int(os.environ.get("SLURM_CPUS_PER_TASK", "4")), args.gpus)

    ray.init(
        address=args.ray_address,
        runtime_env={"env_vars": {
            "PYTHONPATH": str(REPO_ROOT),
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": str(cpus),
        }},
    )

    cfg = {
        "model_name": args.model_name,
        "data_dir": str(Path(args.data_dir).resolve()),
        "n_examples": args.n_examples,
        "output_dir": str(out_dir / "adapter"),
        "results_dir": str(results_dir),
        "run_id": args.run_id,
        "per_device_batch": args.per_device_batch,
        "grad_accum": args.grad_accum,
    }
    trainer = TorchTrainer(
        train_loop_per_worker=train_worker,
        train_loop_config=cfg,
        scaling_config=ScalingConfig(
            num_workers=args.gpus,
            use_gpu=True,
            resources_per_worker={"CPU": cpus, "GPU": 1},
            placement_strategy="PACK",
        ),
        run_config=RunConfig(name=args.run_id,
                             storage_path=str(out_dir / "ray")),
    )

    driver = RunRecorder(
        results_dir, args.run_id, "lora", tag="driver",
        params={"gpus": args.gpus, "n_examples": args.n_examples,
                "cpus_per_worker": cpus})
    try:
        trainer.fit()
    finally:
        records = load_records(out_dir)
        end = worker_launch_end(records, fallback=time.time())
        driver.add_interval("startup", min(t0, end), end)


if __name__ == "__main__":
    main()
