"""Join the two judges' per-call scores into one table; a missing score is allowed and counted."""
import argparse
import glob
import json
import os
from collections import defaultdict


def read_judge(run_dir, judge):
    rows = {}
    for path in sorted(glob.glob(os.path.join(run_dir, f"judgments_{judge}_replica*.ndjson"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["index"], r["metric"])
                if key in rows:
                    raise ValueError(f"{judge}: duplicate call {key}")
                rows[key] = r
    return rows


def join(run_dir, out_path=None):
    """Write one row per (response, metric); raise if the judges did not cover the same calls."""
    judges = {j: read_judge(run_dir, j) for j in ("qwen", "granite")}
    if set(judges["qwen"]) != set(judges["granite"]):
        only_q = len(set(judges["qwen"]) - set(judges["granite"]))
        only_g = len(set(judges["granite"]) - set(judges["qwen"]))
        raise ValueError(f"judges cover different calls: {only_q} only Qwen, {only_g} only Granite")
    keys = sorted(judges["qwen"])
    out_path = out_path or os.path.join(run_dir, "judgments_joined.ndjson")
    missing = {"qwen": 0, "granite": 0}
    both_missing = 0
    diffs = defaultdict(list)
    with open(out_path, "w", encoding="utf-8") as fh:
        for key in keys:
            q, g = judges["qwen"][key], judges["granite"][key]
            scores = [r["score"] for r in (q, g) if r["ok"] and r["score"] is not None]
            missing["qwen"] += not (q["ok"] and q["score"] is not None)
            missing["granite"] += not (g["ok"] and g["score"] is not None)
            both_missing += not scores
            if len(scores) == 2:
                diffs[key[1]].append(abs(scores[0] - scores[1]))
            fh.write(json.dumps({
                "index": key[0], "id": q["id"], "metric": key[1],
                "qwen": q["score"] if q["ok"] else None,
                "granite": g["score"] if g["ok"] else None,
                "mean": sum(scores) / len(scores) if scores else None,
                "n_scores": len(scores)}) + "\n")
    summary = {"calls_per_judge": len(keys), "missing": missing, "missing_both": both_missing,
               "mean_abs_diff": {m: round(sum(d) / len(d), 4) for m, d in sorted(diffs.items())}}
    with open(os.path.join(run_dir, "join_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    a = p.parse_args()
    print(json.dumps(join(a.run_dir), indent=2))


if __name__ == "__main__":
    main()
