import yaml
from pathlib import Path


def test_hitl_workflow_parses():
    wf = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "hitl.yaml"
    data = yaml.load(wf.read_text(), Loader=yaml.BaseLoader)

    assert data["on"]["pull_request"]["types"] == ["labeled"]
    job = data["jobs"]["hitl-comment"]
    assert "contains(github.event.pull_request.labels.*.name, 'dgm-hitl')" in job["if"]

    steps = job["steps"]
    run_step = next(s for s in steps if s.get("name") == "Post HITL comment")
    expected = (
        "python -m dgm_kernel.hitl_pr --pr ${{ github.event.pull_request.number }} --msg \"Human-in-the-loop review requested\""
    )
    assert expected in run_step["run"]
