"""
Early tweaks for tests: force LanceDB to use a temp directory that we can
write to, and be sure <repo>/src is on sys.path first.
"""
import os, tempfile, pathlib, sys

TMP = pathlib.Path(tempfile.gettempdir()) / "osiris_lancedb_test"
os.environ.setdefault("DB_ROOT", str(TMP))
os.environ.setdefault("LANCEDB_DATA_PATH", str(TMP))
TMP.mkdir(parents=True, exist_ok=True)

# guarantee src/ is import-able first
ROOT = pathlib.Path(__file__).resolve().parent
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
