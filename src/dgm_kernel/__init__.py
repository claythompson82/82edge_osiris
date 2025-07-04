"""
Public re-exports so legacy tests like
`import dgm_kernel.meta_loop` still resolve after the src/ migration.
"""

from importlib import import_module as _im

from pydantic import BaseModel, ConfigDict

# ──────────────────────── global settings ────────────────────────────
# Apply strict validation across all Pydantic models in this package.
BaseModel.model_config = ConfigDict(strict=True, extra="forbid")

for _name in ("prover", "meta_loop", "sandbox", "hitl_pr"):
    try:
        globals()[_name] = _im(f".{_name}", __name__)
    except ModuleNotFoundError:
        # Module might truly be absent; tests will handle that case.
        pass
