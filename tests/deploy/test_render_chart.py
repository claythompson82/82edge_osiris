import shutil
import subprocess
from pathlib import Path
import yaml
import pytest

CHART_DIR = Path(__file__).resolve().parents[2] / "deploy" / "dgm-kernel-chart"


def render_chart(tmp_path, values):
    values_file = tmp_path / "values.yaml"
    values_file.write_text(yaml.safe_dump(values))
    cmd = [
        "helm",
        "template",
        "test",
        str(CHART_DIR),
        "--values",
        str(values_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return list(yaml.safe_load_all(result.stdout))


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")
def test_servicemonitor_and_alerts_render(tmp_path):
    docs = render_chart(
        tmp_path,
        {"servicemonitor": {"enabled": True}, "alerts": {"enabled": True}},
    )
    kinds = [d.get("kind") for d in docs if isinstance(d, dict)]
    assert "ServiceMonitor" in kinds
    assert "PrometheusRule" in kinds

    sm = next(d for d in docs if d.get("kind") == "ServiceMonitor")
    ep = sm["spec"]["endpoints"][0]
    assert ep["path"] == "/metrics"
    assert ep["interval"] == "15s"

    pr = next(d for d in docs if d.get("kind") == "PrometheusRule")
    rule_names = [r["alert"] for g in pr["spec"]["groups"] for r in g["rules"]]
    assert "HighPatchFailureRate" in rule_names
    assert "UnsafeTokenSpike" in rule_names
