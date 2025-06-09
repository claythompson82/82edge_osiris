import os
from pathlib import Path
db_path_str = os.environ.get("LANCEDB_DATA_PATH", ".lancedb_data")
DB_ROOT = Path(db_path_str)
DB_ROOT.mkdir(parents=True, exist_ok=True)

