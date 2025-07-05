import importlib.util
import tempfile
from pathlib import Path

def _load_db_module():
    path = Path(__file__).resolve().parents[2] / "src" / "llm_sidecar" / "db.py"
    spec = importlib.util.spec_from_file_location("llm_sidecar.db", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_connect_respects_patched_env(monkeypatch):
    db = _load_db_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(db, "DB_ROOT", Path(tmpdir))
        monkeypatch.setattr(db, "_db", None)
        monkeypatch.setattr(db, "_tables", {})

        captured = {}
        orig_connect = db.lancedb.connect

        def mock_connect(path, **kwargs):
            captured["path"] = str(path)
            return orig_connect(path, **kwargs)

        monkeypatch.setattr(db.lancedb, "connect", mock_connect)
        db.connect_db()

        assert captured["path"] == tmpdir
        assert db._db.uri == tmpdir
