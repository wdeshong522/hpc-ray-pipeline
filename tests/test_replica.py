"""Run from the repo root:  python tests/test_replica.py"""
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generate.replica import barrier, prompt_messages, run_client, shard  # noqa: E402


def test_shard_covers_everything_once():
    pool = list(range(10))
    parts = [shard(pool, i, 3) for i in range(3)]
    seen = sorted(pos for part in parts for pos, _ in part)
    assert seen == pool
    assert [pos for pos, _ in parts[1]] == [1, 4, 7]


def test_prompt_stops_before_assistant():
    rec = {"conversations": [{"from": "system", "value": "s"}, {"from": "human", "value": "u"},
                             {"from": "gpt", "value": "a"}]}
    assert [m["role"] for m in prompt_messages(rec)] == ["system", "user"]


def test_barrier_releases_all():
    port, done = 31777, []

    def go(i):
        barrier(i, 3, "localhost", port, 20)
        done.append(i)

    threads = [threading.Thread(target=go, args=(i,)) for i in (2, 1, 0)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(done) == [0, 1, 2]


def test_barrier_times_out_without_the_others():
    try:
        barrier(0, 2, "localhost", 31778, 1)
    except TimeoutError:
        return
    raise AssertionError("expected TimeoutError")


def test_single_replica_barrier_returns():
    barrier(0, 1, "localhost", 31779, 1)


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        bad = "FAIL" in body["messages"][-1]["content"]
        out = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                          "usage": {"completion_tokens": 7}}).encode()
        self.send_response(500 if bad else 200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


def _serve(port):
    srv = HTTPServer(("localhost", port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _item(pos, text):
    return pos, {"conversations": [{"from": "human", "value": text}, {"from": "gpt", "value": "a"}]}


def test_run_client_writes_rows_and_reports_failures():
    srv = _serve(31780)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.ndjson"
        items = [_item(0, "a"), _item(1, "FAIL"), _item(2, "c")]
        errors = run_client(31780, items, 2, out)
        rows = [json.loads(line) for line in out.read_text().splitlines()]
    srv.shutdown()
    assert sorted(r["index"] for r in rows) == [0, 2]
    assert [pos for pos, _ in errors] == [1]
    assert rows[0]["completion_tokens"] == 7 and rows[0]["finish_reason"] == "stop"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("replica tests passed")
