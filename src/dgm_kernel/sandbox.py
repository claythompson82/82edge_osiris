"""Sandbox execution utilities.

This sandbox executes arbitrary patch code under strict safety controls:

- **Filesystem isolation** – execution happens inside a fresh temporary
  directory so no existing files are touched.
- **Resource caps** – CPU time is limited to 5 seconds and memory to
  256 MiB using ``resource.setrlimit`` on POSIX systems.
- **Built‑ins whitelist** – patch code runs with a minimal ``safe_builtins``
  dictionary, preventing usage of dangerous functions like ``open``.
- **Automatic cleanup** – the temporary directory is deleted once
  execution finishes.
"""

from __future__ import annotations

import builtins
import logging
import resource
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple

log = logging.getLogger(__name__)


class SandboxError(Exception):
    """Controlled error returned on sandbox failures."""


SAFE_BUILTINS: Dict[str, object] = {name: getattr(builtins, name) for name in [
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "print",
    "range",
    "str",
    "sum",
    "Exception",
]}


class Sandbox:
    """Execute patch code in a restricted subprocess."""

    def __init__(self, cpu_seconds: int = 5, memory_mb: int = 256) -> None:
        self.cpu_seconds = cpu_seconds
        self.memory_bytes = memory_mb * 1024 * 1024

    def _build_runner(self, patch_path: Path) -> str:
        """Return Python script that executes the patch."""
        names = list(SAFE_BUILTINS.keys())
        return f"""
import resource, pathlib, sys, builtins

def _limits():
    try:
        resource.setrlimit(resource.RLIMIT_CPU, ({self.cpu_seconds}, {self.cpu_seconds}))
        resource.setrlimit(resource.RLIMIT_AS, ({self.memory_bytes}, {self.memory_bytes}))
    except Exception:
        pass

_limits()
code = pathlib.Path('{patch_path.name}').read_text()
safe = {{n: getattr(builtins, n) for n in {names!r}}}
try:
    exec(compile(code, '{patch_path.name}', 'exec'), {{'__builtins__': safe}})
except Exception as e:
    print('SandboxError:', e)
    sys.exit(1)
"""

    def run(self, patch: Dict[str, str]) -> Tuple[bool, str, int]:
        """Run ``patch['after']`` in isolation.

        Returns ``(success, output, exit_code)`` without raising exceptions.
        """
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                tgt = tmp_path / "patch.py"
                tgt.write_text(patch.get("after", ""))
                runner = tmp_path / "runner.py"
                runner.write_text(self._build_runner(tgt))
                proc = subprocess.run(
                    ["python", str(runner)],
                    capture_output=True,
                    text=True,
                    cwd=tmp_path,
                )
                output = proc.stdout + proc.stderr
                success = proc.returncode == 0
                return success, output, proc.returncode
        except Exception as e:  # pragma: no cover - safety net
            log.error("Sandbox execution failed: %s", e)
            return False, f"SandboxError: {e}", 1


def run_patch_in_sandbox(patch: Dict[str, str]) -> Tuple[bool, str, int]:
    """Backward compatible helper calling :class:`Sandbox`."""
    return Sandbox().run(patch)
