"""
Patch-verification helpers used by the DGM kernel.

`prove_patch()` returns a floating-point *score* between ``0.0`` and ``1.0``
expressing how confidently a patch is proven safe.  Inside a temporary copy of
the repository it performs three checks:

1. ``python -m py_compile`` on each patched ``.py`` file (+0.4)
2. ``pytest -q tests/dgm_kernel_tests/test_meta_loop.py::sanity_only`` (+0.4)
3. ``pylint`` on only the patched ``.py`` files; if overall score ≥ 8.0 (+0.2)

Partial scores are summed and clamped to ``0–1``.  Any subprocess error is
logged and the function returns ``0.0``.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# public API
# ──────────────────────────────────────────────────────────────────────────────
def prove_patch(patch_text: str) -> float:
    """
    Validate *patch_text* (a unified diff) inside an isolated tmp-copy of the
    repo and return a score ∈ [0.0, 1.0].

    Steps
    -----
    1. clone $REPO into tmpdir
    2. apply diff with ``patch -p1``
    3. py-compile each modified *.py*
    4. run sanity-only pytest shard
    5. run pylint on the modified *.py*

    On any failure the corresponding partial credit is skipped; unexpected
    exceptions yield 0.0.
    """
    repo_root = Path(__file__).resolve().parents[1]

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_repo = Path(tmpdir) / "repo"
            shutil.copytree(repo_root, tmp_repo, dirs_exist_ok=True)

            patch_file = Path(tmpdir) / "patch.diff"
            patch_file.write_text(patch_text)

            # ── apply diff ───────────────────────────────────────────────
            proc = subprocess.run(
                ["patch", "-p1", "-i", str(patch_file)],
                cwd=tmp_repo,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                log.error("patch failed: %s", proc.stderr.strip())
                return 0.0

            score: float = 0.0
            modified = _patched_files(patch_text)
            py_files = [p for p in modified if p.endswith(".py")]

            # ── 1) py_compile ────────────────────────────────────────────
            compile_ok = True
            for path in py_files:
                cproc = subprocess.run(
                    [sys.executable, "-m", "py_compile", path],
                    cwd=tmp_repo,
                    capture_output=True,
                    text=True,
                )
                if cproc.returncode != 0:
                    log.error("py_compile failed for %s: %s", path, cproc.stderr.strip())
                    compile_ok = False
                    break
            if compile_ok:
                score += 0.4

            # ── 2) pytest shard ──────────────────────────────────────────
            tproc = subprocess.run(
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
            if tproc.returncode == 0:
                score += 0.4
            else:
                log.error(
                    "pytest failed: %s", (tproc.stdout + tproc.stderr).strip()
                )

            # ── 3) pylint quality gate ───────────────────────────────────
            if py_files:
                lproc = subprocess.run(
                    ["pylint", *py_files],
                    cwd=tmp_repo,
                    capture_output=True,
                    text=True,
                )
                m = re.search(r"rated at (-?\d+\.?\d*)/10", lproc.stdout)
                if m:
                    pylint_score = float(m.group(1))
                    log.info("Pylint score %.2f", pylint_score)
                    if pylint_score >= 8.0:
                        score += 0.2
                else:
                    log.warning("Could not parse Pylint score from output.")

            return max(0.0, min(score, 1.0))

    except Exception as exc:  # pragma: no cover – unexpected issues
        log.error("prove_patch error: %s", exc)
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def _patched_files(diff: str) -> list[str]:
    """Return the list of file paths touched by *diff* (unified format)."""
    files: set[str] = set()
    for line in diff.splitlines():
        if line.startswith(("+++ ", "--- ")):
            path = line[4:].split()[0]
            if path.startswith(("a/", "b/")):
                path = path[2:]
            if path != "/dev/null":
                files.add(path)
    return sorted(files)


def _get_pylint_score(patch_code: str) -> float:
    """
    Run pylint on *patch_code* and return its score, or 0.0 on failure / timeout.
    """
    score = 0.0
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fp:
            fp.write(patch_code)
            tmp_path = fp.name

        proc = subprocess.run(
            ["pylint", tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        m = re.search(r"Your code has been rated at (-?\d+\.?\d*)/10", proc.stdout)
        if m:
            score = float(m.group(1))
        else:
            log.warning("Could not parse pylint score from output.")
    except FileNotFoundError:
        log.error("pylint not found – skipping lint score.")
    except subprocess.TimeoutExpired:
        log.error("pylint timed-out.")
    except Exception as exc:
        log.error("Unexpected pylint error: %s", exc)
    finally:
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink(missing_ok=True)
    return score
