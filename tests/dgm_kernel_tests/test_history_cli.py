import json
from pathlib import Path
from click.testing import CliRunner

from dgm_kernel import cli_history


def _make_history(path: Path, n: int) -> None:
    entries = []
    for i in range(n):
        entries.append(
            {
                "patch_id": f"p{i}",
                "timestamp": i,
                "reward": float(i),
                "lines_changed": i + 1,
            }
        )
    path.write_text(json.dumps(entries))


def test_history_table_header(tmp_path, monkeypatch):
    history_file = tmp_path / "history.json"
    _make_history(history_file, 3)
    monkeypatch.setattr(cli_history, "PATCH_HISTORY_FILE", history_file)

    runner = CliRunner()
    result = runner.invoke(cli_history.main, ["--last", "2"])
    assert result.exit_code == 0, result.output
    first = result.output.splitlines()[0].split()
    assert first == ["id", "ts", "reward", "lines_changed"]
