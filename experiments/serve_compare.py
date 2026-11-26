"""Throughput comparison for models served by one vLLM server, with requests in flight at once."""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.serve_smoke import prompt_messages
from src.train.data import load_records, ordered_pool, take

TEMPERATURE = 0.3
TOP_P = 0.9
MAX_TOKENS = 1536
SEED = 42


def request(port, model, messages, seed):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": MAX_TOKENS,
                       "temperature": TEMPERATURE, "top_p": TOP_P, "seed": seed}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.load(resp)
    return data["choices"][0]["finish_reason"], data["usage"]["completion_tokens"]


def run_model(port, model, prompts, concurrency):
    """Warm up with two requests, then time all prompts sent with the given concurrency."""
    for i, msgs in enumerate(prompts[:2]):
        request(port, model, msgs, SEED + 1000 + i)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda a: request(port, model, a[1], SEED + a[0]),
                                enumerate(prompts)))
    wall = time.time() - t0
    tokens = sum(t for _, t in results)
    return {"model": model, "requests": len(results), "concurrency": concurrency,
            "wall_s": round(wall, 1), "tokens": tokens,
            "tokens_per_s": round(tokens / wall, 1),
            "mean_tokens": round(tokens / len(results), 1),
            "stop": sum(1 for f, _ in results if f == "stop"),
            "length": sum(1 for f, _ in results if f == "length")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--concurrency", type=int, default=12)
    args = p.parse_args()

    records, _ = load_records(args.data_dir)
    prompts = [prompt_messages(r) for r in take(ordered_pool(records), args.n)]
    for model in args.models:
        print(json.dumps(run_model(args.port, model, prompts, args.concurrency)), flush=True)


if __name__ == "__main__":
    main()
