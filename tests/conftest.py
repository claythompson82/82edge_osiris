# ---------- TEST DATA DIRS ----------
import os, tempfile, pathlib

# point LanceDB to a temp dir the user can write to
TEMP_DATA_DIR = pathlib.Path(tempfile.gettempdir()) / "osiris_lancedb_test"
os.environ.setdefault("DB_ROOT", str(TEMP_DATA_DIR))
os.environ.setdefault("LANCEDB_DATA_PATH", str(TEMP_DATA_DIR))  # fallback name

TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
# ------------------------------------
"""
pytest bootstrap — adds the project’s src/ folder to sys.path
without clobbering site-packages.

Why it’s safe:
• Uses insert(0) so stdlib & site-packages stay visible.
• Only inserts if the path isn’t already present (idempotent).
"""

import sys
from pathlib import Path

SRC_DIR = (Path(__file__).resolve().parent.parent / "src").resolve()
SRC_STR = str(SRC_DIR)

# Remove any prior occurrences, then add to index 0
try:
    sys.path.remove(SRC_STR)
except ValueError:
    pass

sys.path.insert(0, SRC_STR)
