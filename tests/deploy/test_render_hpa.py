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
def test_hpa_render(tmp_path):
    docs = render_chart(
        tmp_path,
        {"autoscaling": {"enabled": True, "minReplicas": 2, "maxReplicas": 4}},
    )
    kinds = [d.get("kind") for d in docs if isinstance(d, dict)]
    assert "HorizontalPodAutoscaler" in kinds
    hpa = next(d for d in docs if d.get("kind") == "HorizontalPodAutoscaler")
    assert hpa["spec"]["minReplicas"] == 2
    assert hpa["spec"]["maxReplicas"] == 4
