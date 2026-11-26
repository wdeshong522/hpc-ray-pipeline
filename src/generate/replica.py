"""One generation replica: start a vLLM server on this node, meet the other replicas, serve this node's shard."""
import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.timing import RunRecorder  # noqa: E402
from src.train.data import load_records, ordered_pool, stable_id, take, to_messages  # noqa: E402

CU13_LIB = "/opt/venvs/serve/lib/python3.12/site-packages/nvidia/cu13/lib"
SERVED_NAME = "merged"
PP_SIZE = 2
TEMPERATURE = 0.3
TOP_P = 0.9
MAX_TOKENS = 1536
SEED = 42


def shard(pool, index, replicas):
    """Round-robin shard: scenario i goes to replica i % replicas."""
    return [(i, rec) for i, rec in enumerate(pool) if i % replicas == index]


def prompt_messages(record):
    """System and user turns of a record: everything before the first assistant turn."""
    out = []
    for m in to_messages(record):
        if m["role"] == "assistant":
            break
        out.append(m)
    return out


def barrier(index, replicas, host, port, timeout_s, sleep=time.sleep, clock=time.time):
    """Block until every replica has called this; replica 0 listens, the others connect."""
    if replicas == 1:
        return
    deadline = clock() + timeout_s
    if index == 0:
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen(replicas)
        conns = []
        try:
            while len(conns) < replicas - 1:
                srv.settimeout(max(deadline - clock(), 0.01))
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    raise TimeoutError("barrier: not every replica arrived")
                conns.append(conn)
            for conn in conns:
                conn.sendall(b"go")
        finally:
            for conn in conns:
                conn.close()
            srv.close()
        return
    while True:
        try:
            conn = socket.create_connection((host, port), timeout=5)
            break
        except OSError:
            if clock() > deadline:
                raise TimeoutError("barrier: replica 0 never listened")
            sleep(1)
    try:
        conn.settimeout(max(deadline - clock(), 1))
        if conn.recv(2) != b"go":
            raise RuntimeError("barrier: bad release message")
    finally:
        conn.close()


def server_command(model_dir, port):
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_dir, "--served-model-name", SERVED_NAME,
            "--pipeline-parallel-size", str(PP_SIZE), "--dtype", "float16",
            "--max-model-len", "4096", "--port", str(port)]


def start_server(model_dir, port, log_path):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = CU13_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    log = open(log_path, "w")
    return subprocess.Popen(server_command(model_dir, port), stdout=log,
                            stderr=subprocess.STDOUT, env=env)


def wait_ready(port, proc, timeout_s, poll_s=5, sleep=time.sleep, clock=time.time):
    """Poll the server's model list until it answers; fail if the process dies or time runs out."""
    deadline = clock() + timeout_s
    while clock() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5):
                return
        except OSError:
            sleep(poll_s)
    raise TimeoutError("vLLM server not ready in time")


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def request(port, messages, seed, timeout_s=1800):
    body = json.dumps({"model": SERVED_NAME, "messages": messages, "max_tokens": MAX_TOKENS,
                       "temperature": TEMPERATURE, "top_p": TOP_P, "seed": seed}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.load(resp)
    choice = data["choices"][0]
    return {"finish_reason": choice["finish_reason"],
            "completion_tokens": data["usage"]["completion_tokens"],
            "text": choice["message"]["content"]}


def run_client(port, items, concurrency, out_path):
    """Send every (position, record) item with `concurrency` in flight; return the list of failures."""
    errors = []
    lock = threading.Lock()
    with open(out_path, "w", encoding="utf-8") as fh:
        def work(item):
            pos, rec = item
            t0 = time.time()
            try:
                out = request(port, prompt_messages(rec), SEED + pos)
            except Exception as exc:
                with lock:
                    errors.append((pos, str(exc)))
                return
            row = {"index": pos, "id": stable_id(rec), **out, "seconds": time.time() - t0}
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(work, items))
    return errors


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", required=True)
    p.add_argument("--n_scenarios", type=int, required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--head_host", default="localhost")
    p.add_argument("--barrier_port", type=int, default=30000)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--index", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    p.add_argument("--replicas", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)))
    p.add_argument("--ready_timeout", type=int, default=1200)
    p.add_argument("--barrier_timeout", type=int, default=1800)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    port = 8000 + int(os.environ.get("SLURM_JOB_ID", 0)) % 1000
    run_dir = Path(a.results_dir) / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = RunRecorder(a.results_dir, a.run_id, "generation", f"replica{a.index}",
                      {"replicas": a.replicas, "gpus": PP_SIZE * a.replicas,
                       "n_scenarios": a.n_scenarios, "pp_size": PP_SIZE,
                       "concurrency": a.concurrency})
    proc = None
    try:
        with rec.interval("startup"):
            records, _ = load_records(a.data_dir)
            items = shard(take(ordered_pool(records), a.n_scenarios), a.index, a.replicas)
            proc = start_server(a.model_dir, port, run_dir / f"server_replica{a.index}.log")
            wait_ready(port, proc, a.ready_timeout)
            barrier(a.index, a.replicas, a.head_host, a.barrier_port, a.barrier_timeout)
        with rec.interval("processing"):
            errors = run_client(port, items, a.concurrency,
                                run_dir / f"responses_replica{a.index}.ndjson")
            if errors:
                raise RuntimeError(f"{len(errors)} of {len(items)} requests failed: {errors[:3]}")
        print(f"replica {a.index}: {len(items)} scenarios done")
    finally:
        stop_server(proc)


if __name__ == "__main__":
    main()
