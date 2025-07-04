import sys
import builtins
import importlib

import pytest

import tests.conftest as ct


def test_to_arrow_with_pyarrow():
    pa = pytest.importorskip("pyarrow")
    ct._install_dummy_lancedb()
    from lancedb import connect

    tbl = connect("dummy").create_table("t", [{"a": 1}, {"a": 2}])
    result = tbl.to_arrow()
    assert isinstance(result, pa.Table)
    assert result.num_rows == 2


def test_to_arrow_without_pyarrow(monkeypatch):
    import tests.conftest as ct

    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyarrow":
            raise ImportError("forced")
        return orig_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "pyarrow", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    ct._install_dummy_lancedb()
    from lancedb import connect
    tbl = connect("dummy").create_table("t", [{"a": 1}])
    result = tbl.to_arrow()
    assert isinstance(result, list)
    assert result == [{"a": 1}]

    monkeypatch.undo()
    ct._install_dummy_lancedb()
