import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.judge import inputs, smoke
from src.judge.prompts import (METRICS, build_prompt, load_prompts, parse_score,
                               score_schema, split_record)
from src.train.data import ordered_pool, stable_id

PROMPTS = "configs/judge_prompts"


def record(i):
    return {"conversations": [{"from": "system", "value": "coach"},
                              {"from": "human", "value": f"scenario {i}"},
                              {"from": "gpt", "value": f"truth {i}"}]}


def test_load_prompts_has_all_metrics():
    reg = load_prompts(PROMPTS)
    assert tuple(reg) == METRICS
    assert reg["hallucination"]["score_type"] == "binary"
    assert reg["correctness"]["requires_ground_truth"] is True


def test_split_record_drops_system():
    assert split_record(record(1)) == ("scenario 1", "truth 1")


def test_build_prompt_fills_fields_and_adds_instruction():
    reg = load_prompts(PROMPTS)
    p = build_prompt(reg["correctness"], "SCEN", "ANS", "TRUTH")
    assert "SCEN" in p and "ANS" in p and "TRUTH" in p
    assert p.rstrip().endswith("}")
    assert "$scenario" not in p


def test_parse_score_cases():
    assert parse_score('{"reasoning": "ok", "score": 0.75}', "continuous") == (0.75, "ok")
    assert parse_score('{"reasoning": "", "score": 1.0}', "binary") == (1, "")
    assert parse_score('{"reasoning": "", "score": 1.5}', "continuous") is None
    assert parse_score("not json", "continuous") is None
    assert parse_score('{"reasoning": ""}', "continuous") is None


def test_score_schema_binary_is_enum():
    assert score_schema("binary")["properties"]["score"]["enum"] == [0, 1]
    assert score_schema("continuous")["properties"]["reasoning"]["maxLength"] == 600


def write_run(tmp_path, n, drop=None, dup=None):
    run = tmp_path / "gen"
    run.mkdir()
    rows = [{"index": i, "id": f"id{i}", "finish_reason": "stop", "text": f"t{i}"}
            for i in range(n) if i != drop]
    if dup is not None:
        rows.append(dict(rows[dup]))
    for r in range(2):
        with open(run / f"responses_replica{r}.ndjson", "w") as fh:
            for row in rows[r::2]:
                fh.write(json.dumps(row) + "\n")
    return run


def test_freeze_orders_rows(tmp_path):
    run = write_run(tmp_path, 6)
    out = tmp_path / "out" / "r.ndjson"
    assert inputs.freeze(run, 6, out) == 6
    assert [json.loads(l)["index"] for l in open(out)] == list(range(6))


def test_freeze_rejects_gap_and_duplicate(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        inputs.freeze(write_run(tmp_path, 6, drop=2), 6, tmp_path / "a.ndjson")
    (tmp_path / "second").mkdir()
    with pytest.raises(ValueError, match="duplicate"):
        inputs.freeze(write_run(tmp_path / "second", 6, dup=1), 6, tmp_path / "b.ndjson")


def test_load_items_checks_ids(tmp_path):
    pool = ordered_pool([record(i) for i in range(5)])
    path = tmp_path / "f.ndjson"
    with open(path, "w") as fh:
        for i, rec in enumerate(pool):
            fh.write(json.dumps({"index": i, "id": stable_id(rec), "text": "x"}) + "\n")
    assert len(inputs.load_items(path, pool, 3)) == 3
    with pytest.raises(ValueError, match="file has"):
        inputs.load_items(path, pool, 9)
    with open(path, "w") as fh:
        fh.write(json.dumps({"index": 0, "id": "wrong", "text": "x"}) + "\n")
    with pytest.raises(ValueError, match="does not match"):
        inputs.load_items(path, pool, 1)


def test_build_calls_one_per_metric():
    reg = load_prompts(PROMPTS)
    pool = [record(i) for i in range(3)]
    items = [(i, pool[i], {"text": f"ans{i}"}) for i in range(3)]
    assert len(smoke.build_calls(items, reg)) == 3 * len(METRICS)


class Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"reasoning": "r", "score": 1})}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_sweep_level_against_fake_server(tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    reg = load_prompts(PROMPTS)
    items = [(i, record(i), {"text": "a"}) for i in range(2)]
    calls = smoke.build_calls(items, reg)
    out = smoke.sweep_level(srv.server_port, "m", calls, 4, tmp_path / "c.ndjson")
    srv.shutdown()
    assert out["calls"] == 10 and out["bad_json"] == 0 and out["truncated"] == 0
    assert len(open(tmp_path / "c.ndjson").readlines()) == 10


def test_plan_roles():
    from src.judge.replica import plan
    assert plan(0, 1, True) == [("qwen", 0, 1), ("granite", 0, 1)]
    assert [plan(i, 4, False) for i in range(4)] == [
        [("qwen", 0, 2)], [("qwen", 1, 2)], [("granite", 0, 2)], [("granite", 1, 2)]]
    with pytest.raises(ValueError):
        plan(0, 3, False)


def test_shard_items_cover_once():
    from src.judge.replica import shard_items
    items = list(range(10))
    parts = [shard_items(items, k, 3) for k in range(3)]
    assert sorted(x for p in parts for x in p) == items and parts[1] == [1, 4, 7]


def test_ports_differ_within_job_and_ensure_fresh(tmp_path):
    from src.judge.replica import ensure_fresh, port_for
    assert len({port_for(8160, i) for i in range(6)}) == 6
    ensure_fresh(tmp_path, [("qwen", 0, 1)])
    (tmp_path / "judgments_qwen_replica0.ndjson").write_text("")
    with pytest.raises(FileExistsError):
        ensure_fresh(tmp_path, [("qwen", 0, 1)])


def test_run_calls_and_join(tmp_path):
    from src.judge import join, replica
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    reg = load_prompts(PROMPTS)
    items = [(i, record(i), {"id": f"id{i}", "text": "a"}) for i in range(2)]
    calls = smoke.build_calls(items, reg)
    ids = {i: f"id{i}" for i in range(2)}
    stats = replica.run_calls(srv.server_port, "m", calls, ids, 4, tmp_path / "judgments_qwen_replica0.ndjson")
    # granite: same calls, but one missing score
    replica.run_calls(srv.server_port, "m", calls, ids, 4, tmp_path / "judgments_granite_replica0.ndjson")
    srv.shutdown()
    assert stats == {"calls": 10, "bad": 0, "errors": 0}
    path = tmp_path / "judgments_granite_replica0.ndjson"
    rows = [json.loads(x) for x in open(path)]
    rows[0].update(ok=False, score=None)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = join.join(str(tmp_path))
    assert summary["calls_per_judge"] == 10 and summary["missing"] == {"qwen": 0, "granite": 1}
    assert summary["missing_both"] == 0
    joined = [json.loads(x) for x in open(tmp_path / "judgments_joined.ndjson")]
    assert sum(r["n_scores"] == 1 for r in joined) == 1 and all(r["mean"] == 1 for r in joined)


def test_join_rejects_unequal_coverage(tmp_path):
    from src.judge import join
    row = {"index": 0, "id": "a", "metric": "correctness", "ok": True, "score": 1}
    (tmp_path / "judgments_qwen_replica0.ndjson").write_text(json.dumps(row) + "\n")
    (tmp_path / "judgments_granite_replica0.ndjson").write_text("")
    with pytest.raises(ValueError):
        join.join(str(tmp_path))
