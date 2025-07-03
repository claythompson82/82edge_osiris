import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace


def test_package_cli_runs_once(tmp_path: Path):
    env = os.environ.copy()
    env["OSIRIS_TEST"] = "1"
    repo_root = Path(__file__).resolve().parents[2]
    module_dir = tmp_path
    lancedb_stub = module_dir / "lancedb"
    lancedb_stub.mkdir()
    (lancedb_stub / "__init__.py").write_text(
        """
class _Table:
    def __init__(self, rows=None):
        self._rows = list(rows or [])
    def add(self, rows):
        self._rows.extend(rows)
    def search(self, *a, **k):
        return self
    def to_list(self, *a, **k):
        return list(self._rows)
    def to_arrow(self, *a, **k):
        return self._rows
    def count_rows(self, *a, **k):
        return len(self._rows)
    def __len__(self):
        return len(self._rows)

class _Conn:
    def __init__(self, uri):
        self.uri = uri
        self._tbls = {}
    def table(self, name, *a, **k):
        return self.open_table(name)
    def open_table(self, name, *a, **k):
        return self._tbls.setdefault(name, _Table())
    def create_table(self, name, data=None, **k):
        tbl = _Table(data)
        self._tbls[name] = tbl
        return tbl
    def drop_table(self, name, *a, **k):
        self._tbls.pop(name, None)
    def table_names(self, *a, **k):
        return list(self._tbls)
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass

def connect(path, **k):
    return _Conn(str(path))
"""
    )
    (lancedb_stub / "pydantic.py").write_text("class LanceModel: pass\n")
    (module_dir / "redis.py").write_text(
        "class Redis:\n    def __init__(self, *a, **k): pass\n    def rpop(self, *a, **k): return None\n"
    )
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root / "src"), str(module_dir)])
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "dgm_kernel", "--once"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed = time.monotonic() - start
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert elapsed < 2

def test_package_cli_loop(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dgm_kernel"])
    async def noop():
        return None
    dummy_mod = ModuleType("redis")
    dummy_mod.Redis = lambda *a, **k: SimpleNamespace(rpop=lambda *_, **__: None)
    monkeypatch.setitem(sys.modules, "redis", dummy_mod)
    monkeypatch.setattr("dgm_kernel.meta_loop.meta_loop", noop)
    from dgm_kernel.__main__ import main
    main()
