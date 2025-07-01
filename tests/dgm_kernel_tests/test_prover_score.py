import subprocess
import sys
from pathlib import Path
from dgm_kernel import prover


def _setup_monkey(monkeypatch, tmp_path, run_results):
    monkeypatch.setattr(prover.shutil, "copytree", lambda s, d, dirs_exist_ok=True: Path(d).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(prover, "_patched_files", lambda diff: [str(tmp_path / "dummy.py")])

    def fake_run(args, cwd=None, capture_output=False, text=False, timeout=None):
        cmd = "pylint" if "pylint" in args else args[0]
        res = run_results.get(cmd)
        if res is None:
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, res[0], res[1], res[2])

    monkeypatch.setattr(prover.subprocess, "run", fake_run)


def test_score_good_patch(monkeypatch, tmp_path):
    run_results = {
        "patch": (0, "", ""),
        sys.executable: (0, "", ""),  # for py_compile
        "pytest": (0, "", ""),
        "pylint": (0, "Your code has been rated at 9.50/10\n", ""),
    }
    _setup_monkey(monkeypatch, tmp_path, run_results)
    score = prover.prove_patch("diff")
    assert score >= 0.9


def test_score_bad_patch(monkeypatch, tmp_path):
    run_results = {
        "patch": (0, "", ""),
        sys.executable: (1, "", "error"),  # py_compile fails
        "pytest": (1, "", ""),
        "pylint": (0, "Your code has been rated at 5.00/10\n", ""),
    }
    _setup_monkey(monkeypatch, tmp_path, run_results)
    score = prover.prove_patch("diff")
    assert score <= 0.4

