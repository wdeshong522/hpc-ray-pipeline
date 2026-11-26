"""Smoke test for one vLLM replica: send real scenarios to the adapter and to the base model."""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.data import load_records, ordered_pool, take, to_messages


def prompt_messages(record):
    """System and user turns of a record: everything before the first assistant turn."""
    out = []
    for m in to_messages(record):
        if m["role"] == "assistant":
            break
        out.append(m)
    return out


def chat(port, model, messages, max_tokens, temperature):
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    choice = data["choices"][0]
    return {"seconds": time.time() - t0, "finish_reason": choice["finish_reason"],
            "tokens": data["usage"]["completion_tokens"], "text": choice["message"]["content"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--adapter_name", required=True)
    p.add_argument("--base_name", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--max_tokens", type=int, default=1536)
    p.add_argument("--temperature", type=float, default=0.3)
    args = p.parse_args()

    records, _ = load_records(args.data_dir)
    for i, rec in enumerate(take(ordered_pool(records), args.n)):
        messages = prompt_messages(rec)
        for name in (args.adapter_name, args.base_name):
            r = chat(args.port, name, messages, args.max_tokens, args.temperature)
            tail = r["text"][-120:].replace("\n", " ")
            print(f"[{i}] model={name} finish={r['finish_reason']} tokens={r['tokens']} "
                  f"sec={r['seconds']:.1f} special_in_text={'<|' in r['text']}")
            print(f"     tail: ...{tail}")


if __name__ == "__main__":
    main()
