import json
import lancedb


DB_PATH = "/app/lancedb_data"
OLD_TABLE = "osiris_runs"
NEW_TABLE = "orchestrator_runs"


def migrate():
    db = lancedb.connect(DB_PATH)
    tables = db.table_names()

    if NEW_TABLE in tables:
        print(f"{NEW_TABLE} table already exists. Nothing to migrate.")
        return

    if OLD_TABLE not in tables:
        print(f"{OLD_TABLE} table not found. Nothing to migrate.")
        return

    old_table = db.open_table(OLD_TABLE)
    records = old_table.search().to_list()
    processed = []
    for rec in records:
        row = dict(rec)
        if "run_id_override" in row and "run_id" not in row:
            row["run_id"] = row.pop("run_id_override")
        fo = row.get("final_output")
        if fo is not None and not isinstance(fo, str):
            try:
                row["final_output"] = json.dumps(fo)
            except Exception:
                row["final_output"] = str(fo)
        processed.append(row)

    try:
        from llm_sidecar.db import OrchestratorRunSchema
        new_table = db.create_table(NEW_TABLE, schema=OrchestratorRunSchema)
    except Exception:
        new_table = db.create_table(NEW_TABLE)

    if processed:
        new_table.add(processed)

    db.drop_table(OLD_TABLE)
    print(f"Migrated {len(processed)} records from {OLD_TABLE} to {NEW_TABLE}.")


if __name__ == "__main__":
    migrate()
