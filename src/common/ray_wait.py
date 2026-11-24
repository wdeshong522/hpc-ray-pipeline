"""Wait until a Ray cluster has registered the expected number of GPUs.

Used by SLURM scripts between "start Ray on every node" and "run the driver".
Exit status 0 means ready, 1 means timed out.
"""
import argparse
import sys
import time


def wait_for_gpus(expected, timeout_s, get_gpus, poll_s=5,
                  sleep=time.sleep, clock=time.monotonic):
    deadline = clock() + timeout_s
    while True:
        found = get_gpus()
        if found >= expected:
            return True
        if clock() >= deadline:
            return False
        sleep(poll_s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--address", required=True)
    p.add_argument("--gpus", type=int, required=True)
    p.add_argument("--timeout", type=float, default=300)
    args = p.parse_args()

    import ray
    ray.init(address=args.address)
    ok = wait_for_gpus(args.gpus, args.timeout,
                       lambda: ray.cluster_resources().get("GPU", 0))
    print(f"ray cluster resources: {ray.cluster_resources()}")
    ray.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
