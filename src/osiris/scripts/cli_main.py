"""
CLI shim that proxies to llm_sidecar.db.cli_main so tests always see
identical behaviour, even after importlib.reload().
"""

from __future__ import annotations

import sys
from llm_sidecar.db import cli_main as _impl

__all__ = ["cli_main"]


def cli_main(argv: list[str] | None = None) -> None:
    _impl(argv)  # pure delegation


if __name__ == "__main__":  # pragma: no cover
    cli_main(sys.argv[1:])
