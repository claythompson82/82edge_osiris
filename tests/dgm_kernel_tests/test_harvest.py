import datetime
import json
from pathlib import Path

from click.testing import CliRunner

from dgm_kernel import harvest_feedback


class DummyArrow:
    def __init__(self, rows):
        self._rows = rows

    def to_pylist(self):
        return list(self._rows)


class DummyTable:
    def __init__(self, rows):
        self._rows = rows

    def to_arrow(self):
        return DummyArrow(self._rows)


class DummyDB:
    def __init__(self, rows):
        self._rows = rows

    def table(self, name: str):
        return DummyTable(self._rows)

    def open_table(self, name: str):
        return DummyTable(self._rows)


def test_harvest_cli_filters_days(tmp_path, monkeypatch):
    now = datetime.datetime.now(datetime.timezone.utc)
    recent_ns = int(now.timestamp() * 1e9)
    old_ns = int((now - datetime.timedelta(days=10)).timestamp() * 1e9)
    rows = [
        {"when": recent_ns, "msg": "keep"},
        {"when": old_ns, "msg": "drop"},
    ]

    monkeypatch.setattr(
        harvest_feedback, "lancedb", type("M", (), {"connect": lambda *_: DummyDB(rows)})
    )

    out_path = tmp_path / "out.jsonl"
    runner = CliRunner()
    result = runner.invoke(harvest_feedback.main, ["--out", str(out_path), "--days", "5"])
    assert result.exit_code == 0, result.output

    data = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert data == [{"when": recent_ns, "msg": "keep"}]
