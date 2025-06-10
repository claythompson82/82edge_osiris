import sys
from pathlib import Path

# Add the 'src' directory to the Python path for the test session
# This allows tests to import modules from src/ as if they were top-level packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
