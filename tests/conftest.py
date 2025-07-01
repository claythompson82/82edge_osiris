"""
tests/conftest.py
───────────────────────────────────────────────────────────────────────────────
Pytest bootstrap for the Osiris test-suite.

• Adds the project’s *src/* dir to ``sys.path`` (idempotent, keeps stdlib first).
• Exposes a throw-away writable LanceDB dir so DB tests never touch real FS.
• Installs an in-memory LanceDB stub that satisfies every legacy call-site.

The stub lives **only at test-time** – production code is untouched.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
import typing as _t

# ── Hypothesis: disable per-example deadline in CI ───────────────────────────
from hypothesis import settings as _hs_settings

_hs_settings.register_profile("ci", deadline=None)
_hs_settings.load_profile("ci")

# ──────────────────────────────────────────────────────────────────────────────
# 1) ensure  src/  is importable before site-packages
# ──────────────────────────────────────────────────────────────────────────────
SRC_DIR = (Path(__file__).resolve().parent.parent / "src").resolve()
SRC_STR = str(SRC_DIR)

if SRC_STR in sys.path:
    sys.path.remove(SRC_STR)
sys.path.insert(0, SRC_STR)

# ──────────────────────────────────────────────────────────────────────────────
# 2) tmp LanceDB root
# ──────────────────────────────────────────────────────────────────────────────
_TEMP_DB_ROOT = Path(tempfile.gettempdir()) / "osiris_lancedb_test"
_TEMP_DB_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("DB_ROOT", str(_TEMP_DB_ROOT))
os.environ.setdefault("LANCEDB_DATA_PATH", str(_TEMP_DB_ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# 3) Dummy LanceDB shim
# ──────────────────────────────────────────────────────────────────────────────
def _install_dummy_lancedb() -> None:
    """Register a minimal in-memory **lancedb** replacement."""
    try:
        import pyarrow as pa  # type: ignore
    except Exception:  # pragma: no cover – Arrow optional
        class _DummyArrow:
            class Table:  # pyarrow.Table façade
                @staticmethod
                def from_pylist(rows):  # noqa: D401
                    return list(rows)
        pa = _DummyArrow()  # type: ignore

    class _Table(SimpleNamespace):
        _rows: list[dict]

        def __init__(self, rows: _t.Iterable[dict] | None = None) -> None:
            super().__init__()
            self._rows = list(rows or [])

        # API surface used in tests --------------------------------------
        def add(self, rows: _t.Iterable[dict]) -> None:
            self._rows.extend(rows)

        def search(self, *_, **__) -> "_Table":      # noqa: D401
            return self

        def to_arrow(self, *_, **__) -> "pa.Table":  # type: ignore[name-defined]
            return pa.Table.from_pylist(self._rows)

        def to_list(self, *_, **__) -> list[dict]:
            return list(self._rows)

        def count_rows(self, *_, **__) -> int:
            return len(self._rows)

        def __len__(self) -> int:  # noqa: D401
            return len(self._rows)

    class _Conn(SimpleNamespace):
        uri: str
        _tbls: dict[str, _Table]

        def __init__(self, uri: str) -> None:
            super().__init__()
            self.uri = uri
            self._tbls = {}

        # helpers --------------------------------------------------------
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

        # context-manager -----------------------------------------------
        def __enter__(self):  # noqa: D401
            return self
        def __exit__(self, *_):  # noqa: D401
            ...

    # Build the fake module tree ----------------------------------------
    m = ModuleType("lancedb")
    m.connect = lambda path, **__: _Conn(str(path))  # type: ignore[arg-type]

    pyd = ModuleType("lancedb.pydantic")
    pyd.LanceModel = type("LanceModel", (), {})
    pyd.pydantic_to_schema = lambda *_, **__: None
    pyd.pydantic_to_arrow_schema = lambda *_, **__: None

    m.pydantic = pyd
    sys.modules["lancedb"] = m
    sys.modules["lancedb.pydantic"] = pyd


if "lancedb" not in sys.modules or os.getenv("OSIRIS_TEST") == "1":
    _install_dummy_lancedb()
    print("conftest: dummy lancedb shim active", file=sys.stderr)
