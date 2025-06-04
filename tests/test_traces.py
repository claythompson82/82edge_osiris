import subprocess
import time
import requests
import os
import re
from collections import Counter, defaultdict

COMPOSE_FILE = os.path.join(os.path.dirname(__file__), "docker-compose.traces.yaml")


def _compose_cmd(*args):
    return ["docker", "compose", "-f", COMPOSE_FILE, *args]


SPAN_ID_RE = re.compile(r"Span ID:\s*(\w+)")
TRACE_ID_RE = re.compile(r"Trace ID:\s*(\w+)")
PARENT_ID_RE = re.compile(r"Parent ID:\s*(\w+)")
NAME_RE = re.compile(r"Name:\s*(.*)")
SERVICE_RE = re.compile(r"service.name:\s*\w*\(?([^)\s]+)\)?")


def _parse_spans(log_text: str):
    spans = []
    current = {}
    service_name = None
    for line in log_text.splitlines():
        if "service.name" in line:
            m = SERVICE_RE.search(line)
            if m:
                service_name = m.group(1)
        m = TRACE_ID_RE.search(line)
        if m:
            current = {"trace_id": m.group(1), "service_name": service_name}
            continue
        m = SPAN_ID_RE.search(line)
        if m:
            current["span_id"] = m.group(1)
            continue
        m = PARENT_ID_RE.search(line)
        if m:
            current["parent_id"] = m.group(1)
            continue
        m = NAME_RE.search(line)
        if m and current:
            current["name"] = m.group(1).strip()
            spans.append(current)
            current = {}
    return spans


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
    spans = _parse_spans(logs)
    names = [s["name"] for s in spans]
    counts = Counter(names)
    assert any("/generate" in name for name in names), "generate span missing"
    assert counts.get("orchestrator.run", 0) >= 1


def test_trace_correlation_and_parenting():
    requests.post("http://localhost:8000/generate/", json={"prompt": "chain"})
    time.sleep(3)
    logs = subprocess.check_output(_compose_cmd("logs", "otel-collector"), text=True)
    spans = _parse_spans(logs)

    traces = defaultdict(list)
    for s in spans:
        traces[s["trace_id"].lower()].append(s)

    correlated = None
    for spans_in_trace in traces.values():
        names = {sp["name"] for sp in spans_in_trace}
        if any("/generate" in n for n in names) and "orchestrator.meta_flow" in names:
            correlated = spans_in_trace
            break

    assert correlated is not None, "No trace spanning both services found"

    span_by_id = {s["span_id"].lower(): s for s in correlated}
    run_span = next(s for s in correlated if s["name"] == "orchestrator.run")
    meta_span = next(s for s in correlated if s["name"] == "orchestrator.meta_flow")

    assert meta_span["parent_id"].lower() == run_span["span_id"].lower()
    assert run_span.get("service_name") == "orchestrator_test"
    sidecar_span = next(s for s in correlated if "/generate" in s["name"])
    assert sidecar_span.get("service_name") == "llm_sidecar_test"
