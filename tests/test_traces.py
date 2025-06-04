import subprocess
import time
import requests
import os
import re
from collections import Counter

COMPOSE_FILE = os.path.join(os.path.dirname(__file__), "docker-compose.traces.yaml")


def _compose_cmd(*args):
    return ["docker", "compose", "-f", COMPOSE_FILE, *args]


def setup_module(module):
    subprocess.run(_compose_cmd("up", "-d"), check=True)
    # wait for sidecar
    for _ in range(30):
        try:
            r = requests.post("http://localhost:8000/generate/", json={"prompt": "hi"})
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        raise RuntimeError("llm-sidecar did not start")


def teardown_module(module):
    subprocess.run(_compose_cmd("down", "-v"), check=True)


def test_traces_collected():
    requests.post("http://localhost:8000/generate/", json={"prompt": "test"})
    time.sleep(3)
    logs = subprocess.check_output(_compose_cmd("logs", "otel-collector"), text=True)
    span_names = re.findall(r"Name:\s+(.*)", logs)
    counts = Counter(span_names)
    assert any("/generate" in name for name in counts), "generate span missing"
    assert counts.get("orchestrator.run", 0) >= 1
