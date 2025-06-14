"""
osiris package bootstrap.

Nothing here is performance-critical – it mainly provides a couple of run-time
shims so that the lightweight unit-test harness can import dotted-modules that
don’t exist in the trimmed-down OSS build.
"""
from __future__ import annotations

import importlib
import sys
import types as _types
from pathlib import Path

# --------------------------------------------------------------------------- #
# expose “osiris.llm_sidecar” as a top-level import alias so that older code
# (`from osiris.llm_sidecar import …`) continues to work.
#
# NOTE: we *only* create a dummy module wrapper – the real implementation lives
#       under ``src/llm_sidecar``.
# --------------------------------------------------------------------------- #
try:
    llm_sidecar = importlib.import_module("llm_sidecar")
except ModuleNotFoundError:  # pragma: no cover – local dev install
    # allow editable-installs that haven’t pip-installed llm_sidecar
    pkg_root = Path(__file__).resolve().parents[2] / "llm_sidecar"
    spec = importlib.util.spec_from_file_location("llm_sidecar", pkg_root / "__init__.py")
    llm_sidecar = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["llm_sidecar"] = llm_sidecar
    spec.loader.exec_module(llm_sidecar)  # type: ignore[call-arg]

sys.modules.setdefault("osiris.llm_sidecar", llm_sidecar)

# --------------------------------------------------------------------------- #
# 🆕  **new** – tests import “osiris.llm_sidecar.server” even though that
#               module isn’t part of the pared-down open-source drop.  We just
#               register a completely empty placeholder so the import succeeds.
# --------------------------------------------------------------------------- #
sys.modules.setdefault(
    "osiris.llm_sidecar.server", _types.ModuleType("osiris.llm_sidecar.server")
)
