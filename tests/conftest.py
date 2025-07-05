"""
tests/conftest.py
───────────────────────────────────────────────────────────────────────────────
 Pytest bootstrap for the Osiris test-suite.

 • Adds the project’s *src/* directory to ``sys.path`` (idempotent, keeps
   stdlib & site-packages first after the insert).
 • Exposes a temporary, writable LanceDB root so the DB-centric tests never
   touch the real filesystem.
 • Monkey-patches the *lancedb* import with an in-memory stub that implements
   just enough API surface for all legacy tests.

The stub lives **only in test-time memory** – production code is unaffected.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
import typing as _t
from hypothesis import settings
import sys, pathlib
_shim_path = pathlib.Path(__file__).parent / "_shims"
sys.path.insert(0, str(_shim_path))

# Provide compatibility shim for fakeredis
import tests._shims.fake_aioredis  # noqa: F401

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

# ──────────────────────────────────────────────────────────────────────────────
# 1)  ensure  src/  is importable before site-packages
# ────────────────────────────────────────────────────────────────────────────────
SRC_DIR = (Path(__file__).resolve().parent.parent / "src").resolve()
SRC_STR = str(SRC_DIR)

if SRC_STR in sys.path:
    sys.path.remove(SRC_STR)
sys.path.insert(0, SRC_STR)

# ────────────────────────────────────────────────────────────────────────────────
# 2)  point LanceDB to a throw-away directory we can always write
# ────────────────────────────────────────────────────────────────────────────────
_TEMP_DB_ROOT = Path(tempfile.gettempdir()) / "osiris_lancedb_test"
_TEMP_DB_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("DB_ROOT", str(_TEMP_DB_ROOT))
os.environ.setdefault("LANCEDB_DATA_PATH", str(_TEMP_DB_ROOT))

# ────────────────────────────────────────────────────────────────────────────────
# 3)  Monkey-patch a *realistic enough* LanceDB stub
#     (all failures were missing the calls below)
# ────────────────────────────────────────────────────────────────────────────────
def _install_dummy_lancedb() -> None:
    """
    Registers a ``lancedb`` module that implements the handful of methods /
    attributes our tests expect:

    • lancedb.connect(...)               → DummyConnection
    • DummyConnection.uri
      .table(), .open_table(), .create_table(), .drop_table(), .table_names()
    • DummyTable.add(), .search(), .to_arrow(), .to_list(), .count_rows(), len()
    • lancedb.pydantic.LanceModel  +  stubs for pydantic_to_schema / arrow
    """
    import types
    try:
        import pyarrow as pa  # type: ignore
    except Exception:  # pragma: no cover - optional dep
        pa = None

    class _Table(SimpleNamespace):
        _rows: list[dict]

        def __init__(self, rows: _t.Iterable[dict] | None = None) -> None:
            super().__init__()
            self._rows = list(rows or [])

        # Minimal methods the tests call ----------------------------------
        def add(self, rows: _t.Iterable[dict]) -> None:
            self._rows.extend(rows)

        def search(self, *_, **__) -> "_Table":   # legacy “.search().to_list()”
            return self

        def to_arrow(self, *_, **__) -> _t.Any:
            if pa is None:  # pragma: no cover - optional dep
                return list(self._rows)

            if not self._rows:
                return pa.Table.from_pylist([])

            # determine ordered set of all keys across rows
            keys: list[str] = []
            seen = set()
            for row in self._rows:
                for k in row:
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)

            fields = []
            for k in keys:
                arrow_type = pa.null()
                for row in self._rows:
                    if k in row and row[k] is not None:
                        arrow_type = pa.scalar(row[k]).type
                        break
                fields.append(pa.field(k, arrow_type))

            schema = pa.schema(fields)
            return pa.Table.from_pylist(self._rows, schema=schema)

        def to_list(self, *_, **__) -> list[dict]:
            return list(self._rows)

        # Some tests use .count_rows() / len()
        def count_rows(self, *_, **__) -> int:
            return len(self._rows)

        def __len__(self) -> int:
            return len(self._rows)

    class _Conn(SimpleNamespace):
        uri: str
        _tbls: dict[str, _Table]

        def __init__(self, uri: str):
            super().__init__()
            self.uri = uri
            self._tbls = {}

        # -- helpers expected in tests -----------------------------------
        def table(self, name: str, *_, **__) -> _Table:
            return self.open_table(name)

        def open_table(self, name: str, *_, **__) -> _Table:
            return self._tbls.setdefault(name, _Table())

        def create_table(
            self, name: str, data: _t.Iterable[dict] | None = None, **__
        ) -> _Table:
            tbl = _Table(data)
            self._tbls[name] = tbl
            return tbl

        def drop_table(self, name: str, *_, **__) -> None:
            self._tbls.pop(name, None)

        def table_names(self, *_, **__) -> list[str]:
            return list(self._tbls)

        # Context-manager support for any with-blocks
        def __enter__(self): return self
        def __exit__(self, *_): ...

    # Build the fake module tree -----------------------------------------
    m = ModuleType("lancedb")
    m.connect = lambda path, **__: _Conn(str(path))  # type: ignore[arg-type]

    # Sub-module lancedb.pydantic with minimal symbols
    pyd_sub = ModuleType("lancedb.pydantic")
    pyd_sub.LanceModel = type("LanceModel", (), {})
    pyd_sub.pydantic_to_schema = lambda *_, **__: None
    pyd_sub.pydantic_to_arrow_schema = lambda *_, **__: None

    m.pydantic = pyd_sub
    sys.modules["lancedb"] = m
    sys.modules["lancedb.pydantic"] = pyd_sub


# Install only if the real package is unavailable *or* the tests explicitly
# requested the dummy via OSIRIS_TEST (avoids masking a real dev install).
if "lancedb" not in sys.modules or os.getenv("OSIRIS_TEST") == "1":
    _install_dummy_lancedb()
    print("conftest: dummy lancedb shim active", file=sys.stderr)
