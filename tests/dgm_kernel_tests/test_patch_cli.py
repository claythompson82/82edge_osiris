import json
from pathlib import Path

from click.testing import CliRunner

from dgm_kernel import patch_cli


def _make_history(path: Path, n: int) -> list[dict[str, str]]:
    entries = []
    for i in range(n):
        entries.append({"patch_id": f"p{i}", "diff": f"diff{i}"})
    path.write_text(json.dumps(entries))
    return entries


def test_list_limits_output(tmp_path, monkeypatch):
    history_file = tmp_path / "history.json"
    entries = _make_history(history_file, 10)
    monkeypatch.setattr(patch_cli, "PATCH_HISTORY_FILE", history_file)

    runner = CliRunner()
    result = runner.invoke(patch_cli.main, ["list", "--limit", "5"])
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines == [e["patch_id"] for e in entries][-5:]


def test_show_outputs_diff(tmp_path, monkeypatch):
    history_file = tmp_path / "history.json"
    _make_history(history_file, 3)
    monkeypatch.setattr(patch_cli, "PATCH_HISTORY_FILE", history_file)

    runner = CliRunner()
    result = runner.invoke(patch_cli.main, ["show", "--id", "p1"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "diff1"


def test_show_requires_id(monkeypatch):
    runner = CliRunner()
    result = runner.invoke(patch_cli.main, ["show"])
    assert result.exit_code != 0
    assert "Missing option '--id'" in result.output


def test_histogram(tmp_path, monkeypatch):
    history_file = tmp_path / "history.json"
    entries = [
        {"patch_id": "p0", "pylint_score": 9.5},
        {"patch_id": "p1", "pylint_score": 8.3},
        {"patch_id": "p2", "pylint_score": 5.1},
    ]
    history_file.write_text(json.dumps(entries))
    monkeypatch.setattr(patch_cli, "PATCH_HISTORY_FILE", history_file)

    runner = CliRunner()
    result = runner.invoke(patch_cli.main, ["hist"])
    assert result.exit_code == 0, result.output
    lines = set(result.output.strip().splitlines())
    assert {"8\u20139: 1", "9\u201310: 1", "5\u20136: 1"}.issubset(lines)
