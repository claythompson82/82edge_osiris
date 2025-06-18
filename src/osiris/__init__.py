"""
Top-level *osiris* package – wires up legacy alias paths used by the tests.
"""

from __future__ import annotations
import importlib, sys
from types import ModuleType
from typing import Final

__all__ = ["llm_sidecar", "__version__"]
__version__: Final[str] = "0.0.0.dev"

# ── re-export llm_sidecar ───────────────────────────────────────────────
import llm_sidecar as _ls  # noqa: E402
sys.modules.setdefault("osiris.llm_sidecar", _ls)
llm_sidecar: ModuleType = _ls

# expose cli_main
import llm_sidecar.db as _ls_db  # noqa: E402
setattr(llm_sidecar, "cli_main", _ls_db.cli_main)

# ── make scripts package resolvable ──────────────────────────────────────
scripts_pkg = importlib.import_module("osiris.scripts")
sys.modules["osiris.scripts"] = scripts_pkg
importlib.import_module("osiris.scripts.harvest_feedback")

# ── deep aliasing so patch paths in tests always resolve  🔍──────────────
from llm_sidecar import loader as _ls_loader        # noqa: E402
from llm_sidecar import hermes_plugin as _ls_hp     # noqa: E402
sys.modules.setdefault("osiris.llm_sidecar.loader", _ls_loader)
sys.modules.setdefault("osiris.llm_sidecar.hermes_plugin", _ls_hp)
sys.modules.setdefault("osiris.llm_sidecar.db", _ls_db)

# server alias was added earlier
import osiris.server as _srv  # noqa: E402
sys.modules.setdefault("osiris.llm_sidecar.server", _srv)
