# ---------- TEST DATA DIRS ----------
import os, tempfile, pathlib

# point LanceDB to a temp dir the user can write to
TEMP_DATA_DIR = pathlib.Path(tempfile.gettempdir()) / "osiris_lancedb_test"
os.environ.setdefault("DB_ROOT", str(TEMP_DATA_DIR))
os.environ.setdefault("LANCEDB_DATA_PATH", str(TEMP_DATA_DIR))  # fallback name

TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
# ------------------------------------
"""
pytest bootstrap — adds the project’s src/ folder to sys.path
without clobbering site-packages.

Why it’s safe:
• Uses insert(0) so stdlib & site-packages stay visible.
• Only inserts if the path isn’t already present (idempotent).
"""

import sys
from pathlib import Path

SRC_DIR = (Path(__file__).resolve().parent.parent / "src").resolve()
SRC_STR = str(SRC_DIR)

# Remove any prior occurrences, then add to index 0
try:
    sys.path.remove(SRC_STR)
except ValueError:
    pass

sys.path.insert(0, SRC_STR)

# --- LanceDB Monkeypatch for AZR-02 ---
import sys as _sys_mp
import types as _types_mp

# Define dummy classes at a scope accessible by the lambda for connect
class _MP_DummyLanceTable:
    def add(self, *args, **kwargs) -> None: pass
    def search(self, *args, **kwargs) -> '_MP_DummyLanceTable': return self
    def to_pandas(self, *args, **kwargs):
        try:
            import pandas as _pd_mp
            return _pd_mp.DataFrame()
        except ImportError:
            return {}
    def count_rows(self, *args, **kwargs) -> int: return 0
    def __len__(self) -> int: return 0 # For LanceTable.create if it checks len of dummy data

class _MP_DummyLanceDBConnection:
    def table_names(self, *args, **kwargs) -> list: return []
    def open_table(self, name: str, *args, **kwargs) -> _MP_DummyLanceTable: return _MP_DummyLanceTable()
    def create_table(self, name: str, *args, **kwargs) -> _MP_DummyLanceTable:
        # The create_table in llm_sidecar uses schema=PydanticModel
        # This dummy needs to accept that.
        return _MP_DummyLanceTable()
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass

_dummy_lancedb_module = _types_mp.ModuleType("lancedb")
_dummy_pydantic_submodule = _types_mp.ModuleType("lancedb.pydantic")
_dummy_pydantic_submodule.pydantic_to_schema = lambda *a, **k: None
_dummy_pydantic_submodule.LanceModel = type("LanceModel", (object,), {})
_dummy_lancedb_module.pydantic = _dummy_pydantic_submodule
_dummy_lancedb_module.pydantic_to_arrow_schema = lambda *a, **k: None
_dummy_lancedb_module.connect = lambda *a, **k: _MP_DummyLanceDBConnection()

_sys_mp.modules["lancedb"] = _dummy_lancedb_module
_sys_mp.modules["lancedb.pydantic"] = _dummy_pydantic_submodule

print("LOG_CONFTEST: Applied monkey-patch for lancedb (dummy connect, LanceModel, pydantic_to_schema).", file=_sys_mp.stderr)

# Clean up some names from conftest's global scope if they were only for the patch
# Note: _MP_Dummy classes need to remain in scope for the lambda.
# They will be cleaned when conftest.py module scope itself is cleaned.
del _sys_mp, _types_mp
# --- End LanceDB Monkeypatch ---
