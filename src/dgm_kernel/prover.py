"""Patch verification helpers used by the DGM kernel.

`prove_patch()` now returns a floating point *score* between ``0.0`` and
``1.0`` representing how confidently a patch is proven safe.  Inside a
temporary copy of the repository it executes three checks:

1. ``python -m py_compile`` on each patched ``.py`` file (``+0.4``).
2. ``pytest -q tests/dgm_kernel_tests/test_meta_loop.py::sanity_only``
   (``+0.4``).
3. ``pylint`` on only the patched ``.py`` files giving ``+0.2`` when the
   reported score is at least ``8.0``.

The partial scores are summed and clamped to the ``0–1`` range.  Any
subprocess errors are logged but result in ``0.0`` being returned.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
import re
import logging
log = logging.getLogger(__name__)


def prove_patch(patch_text: str) -> float:
    """Validate a unified diff in an isolated temporary directory and
    return a score between 0.0 and 1.0.

    The function copies the current repository to a temporary directory,
    applies the provided ``patch_text`` using the ``patch`` command and then
    performs two checks:

    1. ``python -m py_compile`` on each modified ``.py`` file (+0.4).
    2. ``pytest -q tests/dgm_kernel_tests/test_meta_loop.py::sanity_only``
       (+0.4).
    3. ``pylint`` on the patched ``.py`` files with score >= ``8.0`` (+0.2).

    The partial points are summed and clamped to the ``0–1`` range.  On any
    unexpected error ``0.0`` is returned.
    """

    repo_root = Path(__file__).resolve().parents[1]
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_repo = Path(tmpdir) / "repo"
            shutil.copytree(repo_root, tmp_repo, dirs_exist_ok=True)

            patch_file = Path(tmpdir) / "patch.diff"
            patch_file.write_text(patch_text)

            apply_proc = subprocess.run(
                ["patch", "-p1", "-i", str(patch_file)],
                cwd=tmp_repo,
                capture_output=True,
                text=True,
            )
            if apply_proc.returncode != 0:
                log.error("patch failed: %s", apply_proc.stderr.strip())
                return 0.0

            score = 0.0

            modified = _patched_files(patch_text)
            py_files = [p for p in modified if p.endswith(".py")]
            compile_ok = True
            for path in py_files:
                proc = subprocess.run(
                    [sys.executable, "-m", "py_compile", path],
                    cwd=tmp_repo,
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    log.error(
                        "py_compile failed for %s: %s", path, proc.stderr.strip()
                    )
                    compile_ok = False
                    break

            if compile_ok:
                score += 0.4

            test_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/dgm_kernel_tests/test_meta_loop.py::sanity_only",
                ],
                cwd=tmp_repo,
                capture_output=True,
                text=True,
            )
            if test_proc.returncode == 0:
                score += 0.4
            else:
                log.error(
                    "pytest failed: %s", (test_proc.stdout + test_proc.stderr).strip()
                )

            if py_files:
                pylint_proc = subprocess.run(
                    ["pylint", *py_files],
                    cwd=tmp_repo,
                    capture_output=True,
                    text=True,
                )
                m = re.search(r"rated at (-?\d+\.?\d*)/10", pylint_proc.stdout)
                if m:
                    pylint_score = float(m.group(1))
                    log.info("Pylint score %.2f", pylint_score)
                    if pylint_score >= 8.0:
                        score += 0.2
                else:
                    log.warning(
                        "Could not parse Pylint score from output. stdout: %s",
                        pylint_proc.stdout[:500],
                    )
            else:
                score += 0.2

            return max(0.0, min(score, 1.0))
    except Exception as e:  # pragma: no cover - unexpected issues
        log.error("prove_patch error: %s", e)
        return 0.0


def _patched_files(diff: str) -> list[str]:
    """Extract file paths touched by a unified diff."""
    files: set[str] = set()
    for line in diff.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            path = line[4:].split()[0]
            if path.startswith("a/") or path.startswith("b/"):
                path = path[2:]
            if path != "/dev/null":
                files.add(path)
    return sorted(files)


def _get_pylint_score(patch_code: str) -> float:
    """Run pylint on the given Python code string and return the score.

    Returns 0.0 if pylint is not found, fails, or the score cannot be parsed.
    """
    score = 0.0
    # Create a temporary file path variable to ensure it's defined for finally block
    tmp_file_path_for_pylint = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp_file:
            tmp_file.write(patch_code)
            tmp_file_path_for_pylint = tmp_file.name

        process = subprocess.run(
            ["pylint", tmp_file_path_for_pylint],
            capture_output=True,
            text=True,
            timeout=30,
        )

        match = re.search(
            r"Your code has been rated at (-?\d+\.?\d*)/10", process.stdout
        )
        if match:
            score = float(match.group(1))
            log.info(f"Pylint score for temp patch: {score}/10")
        else:
            log.warning(
                f"Could not parse Pylint score from output. stdout: {process.stdout[:500]}, stderr: {process.stderr[:500]}"
            )

    except FileNotFoundError:
        log.error(
            "pylint command not found. Please ensure pylint is installed and in PATH."
        )
    except subprocess.TimeoutExpired:
        log.error(f"Pylint execution timed out for {tmp_file_path_for_pylint}.")
    except Exception as e:
        log.error(f"Error running pylint on temporary file: {e}")
    finally:
        if tmp_file_path_for_pylint and Path(tmp_file_path_for_pylint).exists():
            Path(tmp_file_path_for_pylint).unlink()
    return score
