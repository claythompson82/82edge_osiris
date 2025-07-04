from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Iterable

import click
import lancedb


def _load_rows(db: object) -> Iterable[dict]:
    """Return all rows from the phi3_feedback table as dictionaries."""
    try:
        table = db.table("phi3_feedback")
    except Exception:
        table = db.open_table("phi3_feedback")
    return table.to_arrow().to_pylist()


def _filter_recent(rows: Iterable[dict], cutoff_ns: int) -> Iterable[dict]:
    for row in rows:
        when = row.get("when", 0)
        if when >= cutoff_ns:
            yield row


@click.command()
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path))
@click.option("--days", type=int, help="Only include rows from the last N days")
def main(out_path: Path, days: int | None) -> None:
    """Dump recent phi3_feedback rows to JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(out_path.parent)
    rows = list(_load_rows(db))
    cutoff_ns = 0
    if days is not None:
        cutoff_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
        cutoff_ns = int(cutoff_dt.timestamp() * 1e9)
        rows = list(_filter_recent(rows, cutoff_ns))
    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")


if __name__ == "__main__":  # pragma: no cover - manual CLI usage
    main()
