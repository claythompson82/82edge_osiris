"""
Tests for dgm_kernel.prover.prove_patch – scoring semantics.

We cover two aspects:

1.  The “black-box” contract — prove_patch must always return a float
    ∈ [0, 1] for any unified diff (quick smoke-test).

2.  Branch-level behaviour via monkey-patching subprocess calls so we can
    simulate *good* and *bad* tool-chain outcomes without spawning heavy
    processes.  We verify the resulting score is high (≥ 0.9) for a clean
    patch and low (≤ 0.4) for a clearly broken one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from dgm_kernel import prover


# --------------------------------------------------------------------------- #
# 1. black-box contract: always returns float between 0‒1
# --------------------------------------------------------------------------- #
def test_prover_returns_float_in_range() -> None:
    # Minimal valid unified diff adding a dummy function
    diff = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -0,0 +1,3 @@\n"
        "+def foo():\n"
        "+    return 42\n"
        "+\n"
    )
    score = prover.prove_patch(diff)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# --------------------------------------------------------------------------- #
# 2. monkey-patched subprocess path – simulate tool results
# --------------------------------------------------------------------------- #
def _setup_monkey(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_results: Dict[str, Tuple[int, str, str]],
) -> None:
    """Patch shutil.copytree & subprocess.run inside prover for fast tests."""
    # copytree → just create the dir
    monkeypatch.setattr(
        prover.shutil,
        "copytree",
        lambda s, d, **__: Path(d).mkdir(parents=True, exist_ok=True),
    )

    # patched_files → pretend we always touch one dummy file
    monkeypatch.setattr(
        prover, "_patched_files", lambda _diff: [str(tmp_path / "dummy.py")]
    )

    def fake_run(
        args: Any, *, cwd: Any = None, capture_output: bool = False, text: bool = False, timeout: Any = None
    ) -> subprocess.CompletedProcess[str]:
        """Return pre-canned exit-code/stdout/stderr triplets."""
        cmd = "pylint" if "pylint" in args else args[0]  # rough dispatch
        rc, out, err = run_results.get(cmd, (0, "", ""))
        return subprocess.CompletedProcess(args, rc, out, err)

    monkeypatch.setattr(prover.subprocess, "run", fake_run)


def test_score_good_patch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """All tool-chain checks succeed → high score (≥ 0.9)."""
    run_ok = {
        "patch": (0, "", ""),
        sys.executable: (0, "", ""),  # py_compile
        "pytest": (0, "", ""),
        "pylint": (0, "Your code has been rated at 9.50/10\n", ""),
    }
    _setup_monkey(monkeypatch, tmp_path, run_ok)
    assert prover.prove_patch("diff") >= 0.9


def test_score_bad_patch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Compilation fails → low score (≤ 0.4)."""
    run_bad = {
        "patch": (0, "", ""),
        sys.executable: (1, "", "error"),  # py_compile fails
        "pytest": (1, "", ""),
        "pylint": (0, "Your code has been rated at 5.00/10\n", ""),
    }
    _setup_monkey(monkeypatch, tmp_path, run_bad)
    assert prover.prove_patch("diff") <= 0.4
