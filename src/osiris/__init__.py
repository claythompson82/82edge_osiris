"""
Top-level ``osiris`` package
===========================

*   Re-export **llm_sidecar** as ``osiris.llm_sidecar`` for backward-
    compatibility with older code and tests.

*   Make sure _our_ internal sub-package ``osiris.scripts`` is used,
    **not** the unrelated top-level ``scripts`` directory that ships with
    the repo (that one contains operational shell helpers, not importable
    libraries).  Tests expect::

        from osiris.scripts.harvest_feedback import main

    to succeed, so we eagerly import that sub-module here.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Final

# --------------------------------------------------------------------------- #
# Public metadata
# --------------------------------------------------------------------------- #
__all__: list[str] = [
    "llm_sidecar",        # re-exported module
    "__version__",        # semantic version string
]

__version__: Final[str] = "0.0.0.local"


# --------------------------------------------------------------------------- #
# 1) Re-export llm_sidecar
# --------------------------------------------------------------------------- #
import llm_sidecar as _llm_sidecar  # noqa: E402  (after std-lib imports)

# A single canonical entry in ``sys.modules`` so that:
#   import osiris.llm_sidecar as lls
# yields the very same object as plain ``import llm_sidecar``.
sys.modules.setdefault("osiris.llm_sidecar", _llm_sidecar)
llm_sidecar: ModuleType = _llm_sidecar  # re-export for ``osiris.llm_sidecar``


# --------------------------------------------------------------------------- #
# 2) Guarantee *our* scripts sub-package wins
# --------------------------------------------------------------------------- #
#
# The repository already has a **top-level** directory called ``scripts/``.
# If we naïvely do ``import osiris.scripts`` first, the interpreter may
# bind it to that external namespace-package — which does **not** expose
# ``harvest_feedback.main``.  To avoid that, we:
#
#   • Explicitly import the *package* that lives under
#     ``src/osiris/scripts`` (it has an ``__init__.py``).
#   • After that, we eagerly import the sub-module
#     ``osiris.scripts.harvest_feedback`` so its ``main`` symbol is always
#     available for tests.
#
try:
    # This resolves to the package in **src/osiris/scripts** because that
    # directory contains an ``__init__.py``.
    scripts_pkg = importlib.import_module("osiris.scripts")
except ModuleNotFoundError as exc:  # extremely unlikely in a sane checkout
    raise ImportError(
        "Internal package 'osiris.scripts' is missing. "
        "The repository layout may be corrupted."
    ) from exc

# Make absolutely certain the binding sits under the ``osiris`` namespace
sys.modules["osiris.scripts"] = scripts_pkg

# Eagerly load the Harvest helper so that
#   from osiris.scripts.harvest_feedback import main
# always works.
importlib.import_module("osiris.scripts.harvest_feedback")
# --------------------------------------------------------------------------- #
#                    Alias needed for `test_imports.py`                       #
# --------------------------------------------------------------------------- #

import sys as _sys
from . import server as _server

# tests import "osiris.llm_sidecar.server"
_sys.modules["osiris.llm_sidecar.server"] = _server
