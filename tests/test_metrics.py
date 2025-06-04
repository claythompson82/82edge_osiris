import subprocess
import time

import pytest
import requests


@pytest.fixture(scope="session")
def run_compose():
    subprocess.check_call(
        [
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "up",
            "-d",
            "llm-sidecar",
        ]
    )

    for _ in range(30):
        try:
            r = requests.get("http://localhost:8000/health")
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                "docker/compose.yaml",
                "logs",
            ]
        )
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                "docker/compose.yaml",
                "down",
            ]
        )
        raise RuntimeError("llm-sidecar failed to start")

    yield

    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "down",
        ]
    )


@pytest.mark.compose
def test_metrics_endpoint(run_compose):
    response = requests.get("http://localhost:8000/metrics")
    assert response.status_code == 200
    assert "python_gc_objects_collected_total" in response.text
