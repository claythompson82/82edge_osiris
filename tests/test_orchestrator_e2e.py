import os
import subprocess
import time
import json
import shutil

import requests
import lancedb
import redis

COMPOSE_FILE = os.path.join(os.path.dirname(__file__), "docker-compose.orchestrator-e2e.yaml")
DB_DIR = os.path.join(os.path.dirname(__file__), "e2e_lancedb_data")


def _compose_cmd(*args):
    return ["docker", "compose", "-f", COMPOSE_FILE, *args]


def setup_module(module):
    os.makedirs(DB_DIR, exist_ok=True)
    subprocess.run(_compose_cmd("up", "-d"), check=True)
    # wait for sidecar to be ready
    for _ in range(30):
        try:
            r = requests.post("http://localhost:8000/generate/", json={"prompt": "ping"})
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        raise RuntimeError("services did not start in time")


def teardown_module(module):
    subprocess.run(_compose_cmd("down", "-v"), check=True)
    shutil.rmtree(DB_DIR, ignore_errors=True)


def test_full_policy_cycle():
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    r.ping()
    tick = {"timestamp": "2024-01-01T00:00:00Z", "symbol": "E2E", "close": 1.23}
    r.publish("market.ticks", json.dumps(tick))

    db = lancedb.connect(DB_DIR)
    table = None
    for _ in range(30):
        try:
            table = db.open_table("orchestrator_runs")
            if table.count_rows() > 0:
                break
        except FileNotFoundError:
            pass
        time.sleep(1)
    else:
        assert False, "orchestrator run not logged"

    rows = table.search().limit(1).to_list()
    assert rows, "no rows found"
    final_output = json.loads(rows[0]["final_output"])
    assert "phi3_proposal" in final_output

    logs = subprocess.check_output(_compose_cmd("logs", "llm-sidecar"), text=True)
    assert "/generate?model_id=phi3" in logs
    assert "propose_trade_adjustments" in logs
