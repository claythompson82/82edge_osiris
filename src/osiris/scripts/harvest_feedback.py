"""
Harvest feedback rows from LanceDB ➜ JSONL.
"""

import argparse, json, time
from datetime import timedelta, timezone, datetime as _dt
from pathlib import Path

import lancedb


def _ns(dt: timedelta) -> int:
    "timedelta ➜ epoch nanoseconds"
    return int(dt.total_seconds() * 1e9)


def record_matches_filter(rec: dict, *, cutoff_ns: int, schema_version: str) -> bool:
    return (
        rec.get("schema_version", "").startswith(schema_version)
        and rec.get("feedback_type") == "correction"
        and int(rec.get("when", 0)) >= cutoff_ns
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="harvest_feedback")
    p.add_argument("--db-path", default=":memory:")
    p.add_argument("--days-back", type=int, default=1)
    p.add_argument("--schema-version", default="1.0")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    conn = lancedb.connect(args.db_path)
    tbl = conn.open_table("phi3_feedback")  # open_table works even if create_table override missing
    cutoff_ns = time.time_ns() - _ns(timedelta(days=args.days_back))

    rows = [
        r for r in tbl.to_pandas().to_dict("records")
        if record_matches_filter(r, cutoff_ns=cutoff_ns, schema_version=args.schema_version)
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
