"""Run from the repo root:  python tests/test_data.py

Plain asserts, no test framework.  Uses synthetic records only.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train.data import (  # noqa: E402
    load_records, ordered_pool, take, to_messages, build_dataset, stable_id)


def make_records(n):
    return [{"conversations": [
        {"from": "human", "value": f"question {i}"},
        {"from": "gpt", "value": f"answer {i}"}]} for i in range(n)]


def write_jsonl(directory, name, records):
    with open(Path(directory) / name, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def test_nested_prefixes():
    pool = ordered_pool(make_records(200))
    assert take(pool, 50) == take(pool, 100)[:50]
    assert take(pool, 100) == take(pool, 200)[:100]


def test_order_independent_of_input_order():
    recs = make_records(100)
    a = ordered_pool(recs)
    b = ordered_pool(list(reversed(recs)))
    assert [stable_id(r) for r in a] == [stable_id(r) for r in b]


def test_order_is_not_file_order():
    recs = make_records(100)
    assert ordered_pool(recs) != recs


def test_take_too_many_raises():
    try:
        take(ordered_pool(make_records(10)), 11)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_load_records_reads_nested_files_and_rejects_bad_lines():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "sub").mkdir()
        write_jsonl(d, "a.jsonl", make_records(3))
        write_jsonl(Path(d) / "sub", "b.jsonl", make_records(2))
        assert len(load_records(d)) == 5
        with open(Path(d) / "bad.jsonl", "w") as fh:
            fh.write("{not json}\n")
        try:
            load_records(d)
        except ValueError:
            return
        raise AssertionError("expected ValueError")


def test_scan_reports_every_problem_without_stopping():
    from src.train.data import scan_records
    with tempfile.TemporaryDirectory() as d:
        write_jsonl(d, "a.jsonl", make_records(3))
        with open(Path(d) / "b.jsonl", "w") as fh:
            fh.write("{not json}\n")
            fh.write(json.dumps({"conversations": [
                {"from": "human", "value": "alone"}]}) + "\n")
            fh.write(json.dumps({"other": 1}) + "\n")
        good, problems = scan_records(d)
        assert len(good) == 3
        assert [p[1] for p in problems] == [1, 2, 3]


def test_to_messages_maps_roles_and_skips_empty():
    rec = {"conversations": [
        {"from": "system", "value": "s"},
        {"from": "human", "value": "  hi  "},
        {"from": "gpt", "value": ""},
        {"from": "gpt", "value": "yo"}]}
    assert to_messages(rec) == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"}]


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


def test_build_dataset_rejects_short_records():
    bad = [{"conversations": [{"from": "human", "value": "only one"}]}]
    try:
        build_dataset(bad, FakeTokenizer())
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("data tests passed")
