import os, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
TMP  = ROOT / ".tmp" / "lancedb_data"
TMP.mkdir(parents=True, exist_ok=True)

os.environ["DB_ROOT"] = str(TMP)     # <- llm_sidecar.db falls back to this
