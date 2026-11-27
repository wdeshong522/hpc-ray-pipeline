"""Frozen judging input: merge one generation run's replica files into one ordered file."""
import argparse
import glob
import json
import os
from pathlib import Path

from src.train.data import stable_id


def read_rows(run_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, "responses_replica*.ndjson"))):
        with open(path, encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def freeze(run_dir, n, out_path):
    """Write rows 0..n-1 in index order; raise on gaps, duplicates, or empty text."""
    rows = read_rows(run_dir)
    by_index = {}
    for row in rows:
        if row["index"] in by_index:
            raise ValueError(f"duplicate index {row['index']}")
        by_index[row["index"]] = row
    missing = [i for i in range(n) if i not in by_index]
    if missing:
        raise ValueError(f"{len(missing)} indexes missing, first {missing[:5]}")
    empty = [i for i in range(n) if not by_index[i]["text"].strip()]
    if empty:
        raise ValueError(f"{len(empty)} empty responses, first {empty[:5]}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for i in range(n):
            r = by_index[i]
            fh.write(json.dumps({"index": i, "id": r["id"], "finish_reason": r["finish_reason"],
                                 "text": r["text"]}) + "\n")
    return n


def load_items(path, pool, n):
    """First n frozen responses paired with their records; raise if an id does not match the pool."""
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if len(items) == n:
                break
            row = json.loads(line)
            rec = pool[row["index"]]
            if stable_id(rec) != row["id"]:
                raise ValueError(f"response {row['index']} does not match the ordered pool")
            items.append((row["index"], rec, row))
    if len(items) < n:
        raise ValueError(f"asked for {n} responses, file has {len(items)}")
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(f"froze {freeze(a.run_dir, a.n, a.out)} responses to {a.out}")


if __name__ == "__main__":
    main()
