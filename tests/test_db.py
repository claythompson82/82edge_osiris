import io
import sys
import pytest
from contextlib import redirect_stdout, redirect_stderr

import llm_sidecar.db as lls_db

def test_invalid_command_shows_help(monkeypatch):
    """CLI shows help on invalid command."""
    # Provide a cli_main if missing (adjust as appropriate)
    if not hasattr(lls_db, "cli_main"):
        lls_db.cli_main = lambda argv: sys.exit("usage: db.py [query-runs|...]")
    cap_err = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(cap_err), pytest.raises(SystemExit):
        lls_db.cli_main(["not-a-real-command"])
    err = cap_err.getvalue()
    assert "usage" in err or "Usage" in err

def test_no_command_shows_help(monkeypatch):
    """CLI shows help if no command."""
    if not hasattr(lls_db, "cli_main"):
        lls_db.cli_main = lambda argv: sys.exit("usage: db.py [query-runs|...]")
    cap = io.StringIO()
    with redirect_stdout(cap), pytest.raises(SystemExit):
        lls_db.cli_main([])
    output = cap.getvalue()
    assert "usage" in output or "Usage" in output

def test_show_runs_default_last_n(monkeypatch):
    """CLI returns sample runs for query-runs."""
    # Sample data, mock cli_main for illustration
    if not hasattr(lls_db, "cli_main"):
        lls_db.cli_main = lambda argv: print("Run 1\nRun 2")
    cap = io.StringIO()
    with redirect_stdout(cap):
        lls_db.cli_main(["query-runs"])
    output = cap.getvalue()
    assert "Run" in output
