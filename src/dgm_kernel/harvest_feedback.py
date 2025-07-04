from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Iterable, TypedDict, Iterator

import click
import lancedb  # type: ignore

# Cached connection for internal helpers
DB: lancedb.DBConnection | None = None  # type: ignore[attr-defined]


class FeedbackRow(TypedDict):
    """Defines the structure for a row of feedback data."""

    when: int
    text: str
    # Add other fields from your table as needed
    # example_field: str | None


def _load_rows(db: lancedb.DBConnection) -> Iterable[FeedbackRow]:
    """Return all rows from the phi3_feedback table as dictionaries."""
    table: lancedb.LanceTable  # type: ignore[attr-defined]
    try:
        table = db.table("phi3_feedback")  # type: ignore[attr-defined]
    except Exception:
        try:
            table = db.open_table("phi3_feedback")
        except Exception:
            return []
    return table.to_arrow().to_pylist()


def _filter_recent(rows: Iterable[FeedbackRow], cutoff_ns: int) -> Iterable[FeedbackRow]:
    for row in rows:
        if row.get("when", 0) >= cutoff_ns:
            yield row


def get_existing_ids() -> Iterator[dict[str, Any]]:
    """Yield existing feedback rows from the cached DB connection."""
    if DB is None:
        return iter([])
    try:
        table = DB.open_table("phi3_feedback")  # type: ignore[attr-defined]
    except Exception:
        return iter([])
    return iter(table.to_arrow().to_pylist())


@click.command()
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path))
@click.option("--days", type=int, help="Only include rows from the last N days")
def main(out_path: Path, days: int | None) -> None:
    """Dump recent phi3_feedback rows to JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(out_path.parent))
    rows = list(_load_rows(db))
    if days is not None:
        cutoff_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=days
        )
        cutoff_ns = int(cutoff_dt.timestamp() * 1e9)
        rows = list(_filter_recent(rows, cutoff_ns))
    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")


if __name__ == "__main__":  # pragma: no cover - manual CLI usage
    main()
