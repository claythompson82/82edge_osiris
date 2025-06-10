import subprocess
from pathlib import Path
from dgm_kernel import sandbox


def test_run_patch_in_sandbox_success(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "mod.py"
    target.write_text("print('old')\n")

    patch = {"target": str(target), "after": "print('new')"}

    def fake_run(cmd, capture_output=True, text=True):
        assert "--network=none" in cmd
        assert "--memory" in cmd
        return subprocess.CompletedProcess(cmd, 0, "out", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    passed, logs, code = sandbox.run_patch_in_sandbox(patch, repo_root=repo)
    assert passed is True
    assert code == 0


def test_run_patch_in_sandbox_failure(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "mod.py"
    target.write_text("print('old')\n")

    patch = {"target": str(target), "after": "print('new')"}

    def fake_run(cmd, capture_output=True, text=True):
        return subprocess.CompletedProcess(cmd, 1, "", "error")

    monkeypatch.setattr(subprocess, "run", fake_run)

    passed, logs, code = sandbox.run_patch_in_sandbox(patch, repo_root=repo)
    assert passed is False
    assert code == 1

