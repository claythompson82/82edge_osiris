from __future__ import annotations

import json

import click
from rich.console import Console
from rich.table import Table

from .patch_cli import PATCH_HISTORY_FILE


@click.command()
@click.option("--last", "last", default=10, show_default=True, type=int)
def main(last: int) -> None:
    """Display recent patch history."""
    if not PATCH_HISTORY_FILE.exists():
        click.echo("No patch history found", err=True)
        raise SystemExit(1)

    history = json.loads(PATCH_HISTORY_FILE.read_text())
    entries = history[-last:]

    table = Table(box=None)
    for col in ("id", "ts", "reward", "lines_changed"):
        table.add_column(col)

    for entry in entries:
        table.add_row(
            str(entry.get("patch_id", "")),
            str(entry.get("timestamp", "")),
            str(entry.get("reward", "")),
            str(entry.get("lines_changed", "")),
        )

    Console(color_system=None).print(table)


if __name__ == "__main__":  # pragma: no cover - manual CLI usage
    main()
