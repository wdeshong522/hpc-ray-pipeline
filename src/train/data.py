"""Training data: load ShareGPT records, fix one order, take a nested prefix.

Proposal link: LoRA workloads are defined by example count (4,000 for strong
scaling, 1,000 per GPU for weak scaling).  Every smaller workload must be a
prefix of the larger one, and every worker must see the same order.
"""
import glob
import hashlib
import json
import os
import random

SHUFFLE_SEED = 42

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def stable_id(record):
    """Content hash of a record, independent of file name and position."""
    turns = record["conversations"]
    text = "||".join(f"{t.get('from', '')}:{t.get('value', '')}" for t in turns)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ordered_pool(records, seed=SHUFFLE_SEED):
    """One fixed shuffle: sort by content hash, then seeded shuffle.

    The result does not depend on file names or the order files were read.
    """
    pool = sorted(records, key=stable_id)
    random.Random(seed).shuffle(pool)
    return pool


def take(pool, n):
    """First n records of the ordered pool, so smaller sets nest in larger."""
    if n > len(pool):
        raise ValueError(f"asked for {n} examples, pool has {len(pool)}")
    return pool[:n]


def to_messages(record):
    messages = []
    for turn in record["conversations"]:
        role = ROLE_MAP.get(turn.get("from", "").lower())
        text = turn.get("value", "").strip()
        if role is None or not text:
            continue
        messages.append({"role": role, "content": text})
    return messages


def build_dataset(records, tokenizer):
    """Apply the tokenizer's chat template; one 'text' column for SFTTrainer."""
    texts = []
    for i, rec in enumerate(records):
        messages = to_messages(rec)
        if len(messages) < 2:
            raise ValueError(f"record {i}: fewer than two usable turns")
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        texts.append({"text": text})
    from datasets import Dataset
    return Dataset.from_list(texts)


def load_records(data_dir):
    """Read every .jsonl under data_dir.

    Returns (records, problems). Malformed or unusable lines are skipped and
    listed in problems as (file, line number, reason), never hidden.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.jsonl"),
                             recursive=True))
    if not files:
        raise FileNotFoundError(f"no .jsonl files under {data_dir}")
    records, problems = [], []
    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    problems.append((path, lineno, f"bad json: {exc}"))
                    continue
                if not isinstance(obj.get("conversations"), list):
                    problems.append((path, lineno, "no conversations list"))
                    continue
                if len(to_messages(obj)) < 2:
                    problems.append((path, lineno, "fewer than two turns"))
                    continue
                records.append(obj)
    return records, problems


if __name__ == "__main__":
    import sys

    good, problems = load_records(sys.argv[1])
    print(f"good={len(good)} problems={len(problems)} "
          f"unique_good={len({stable_id(r) for r in good})}")
    for path, lineno, why in problems:
        print(f"  {path} line {lineno}: {why[:100]}")
