"""Judge smoke test: serve one judge on a 2-GPU replica and sweep client concurrency."""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.generate.replica import CU13_LIB, PP_SIZE, stop_server, wait_ready  # noqa: E402
from src.judge.inputs import load_items  # noqa: E402
from src.judge.prompts import build_prompt, load_prompts, parse_score, score_schema, split_record  # noqa: E402
from src.train.data import load_records, ordered_pool  # noqa: E402

MAX_TOKENS = 400
REPETITION_PENALTY = 1.15


def server_command(model_dir, served_name, port, max_len):
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_dir, "--served-model-name", served_name,
            "--pipeline-parallel-size", str(PP_SIZE), "--dtype", "float16",
            "--max-model-len", str(max_len), "--port", str(port)]


def start_server(cmd, log_path):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = CU13_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    return subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env)


def kv_lines(log_path):
    """Lines of the server log that report KV cache size and maximum concurrency."""
    pat = re.compile(r"KV cache size|Maximum concurrency")
    return [line.strip() for line in open(log_path, errors="replace") if pat.search(line)]


def judge_call(port, served_name, prompt, score_type):
    body = {"model": served_name, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": MAX_TOKENS,
            "repetition_penalty": REPETITION_PENALTY,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "judge_score", "strict": True,
                                                "schema": score_schema(score_type)}}}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.load(resp)
    choice = data["choices"][0]
    parsed = parse_score(choice["message"]["content"] or "", score_type)
    return {"seconds": time.time() - t0, "finish_reason": choice["finish_reason"],
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
            "ok": parsed is not None, "score": parsed[0] if parsed else None,
            "reasoning": parsed[1] if parsed else choice["message"]["content"]}


def build_calls(items, registry):
    calls = []
    for index, rec, row in items:
        scenario, truth = split_record(rec)
        for metric, entry in registry.items():
            if entry["requires_ground_truth"] and not truth:
                continue
            calls.append((index, metric, entry["score_type"],
                          build_prompt(entry, scenario, row["text"], truth)))
    return calls


def sweep_level(port, served_name, calls, concurrency, out_path):
    rows, lock = [], threading.Lock()

    def work(call):
        index, metric, score_type, prompt = call
        try:
            out = judge_call(port, served_name, prompt, score_type)
        except Exception as exc:
            out = {"ok": False, "error": str(exc), "seconds": 0.0, "finish_reason": "error",
                   "prompt_tokens": 0, "completion_tokens": 0, "score": None, "reasoning": ""}
        with lock:
            rows.append({"index": index, "metric": metric, **out})

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(work, calls))
    wall = time.time() - t0
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda r: (r["index"], r["metric"])):
            fh.write(json.dumps(r) + "\n")
    n = len(rows)
    return {"concurrency": concurrency, "calls": n, "wall_s": round(wall, 1),
            "calls_per_s": round(n / wall, 2),
            "completion_tok_per_s": round(sum(r["completion_tokens"] for r in rows) / wall, 1),
            "mean_latency_s": round(sum(r["seconds"] for r in rows) / n, 1),
            "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / n),
            "mean_completion_tokens": round(sum(r["completion_tokens"] for r in rows) / n),
            "bad_json": sum(not r["ok"] for r in rows),
            "truncated": sum(r["finish_reason"] == "length" for r in rows)}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--served_name", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--responses", required=True)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--concurrencies", default="12,32,64,128")
    p.add_argument("--prompts_dir", default="configs/judge_prompts")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="results/judge_smoke")
    p.add_argument("--max_len", type=int, default=6144)
    p.add_argument("--ready_timeout", type=int, default=1200)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    out_dir = Path(a.out_dir) / a.tag
    out_dir.mkdir(parents=True, exist_ok=False)
    port = 8000 + int(os.environ.get("SLURM_JOB_ID", 0)) % 1000
    records, _ = load_records(a.data_dir)
    items = load_items(a.responses, ordered_pool(records), a.n)
    calls = build_calls(items, load_prompts(a.prompts_dir))
    log_path = out_dir / "server.log"
    proc = start_server(server_command(a.model_dir, a.served_name, port, a.max_len), log_path)
    summary = {"tag": a.tag, "model_dir": a.model_dir, "n_responses": a.n, "levels": []}
    try:
        t0 = time.time()
        wait_ready(port, proc, a.ready_timeout)
        summary["startup_s"] = round(time.time() - t0, 1)
        summary["kv_lines"] = kv_lines(log_path)
        for c in (int(x) for x in a.concurrencies.split(",")):
            level = sweep_level(port, a.served_name, calls, c, out_dir / f"calls_c{c}.ndjson")
            summary["levels"].append(level)
            print(json.dumps(level), flush=True)
    finally:
        stop_server(proc)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(summary["kv_lines"]))


if __name__ == "__main__":
    main()
