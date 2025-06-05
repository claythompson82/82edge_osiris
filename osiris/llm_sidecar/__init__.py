"""Wrapper package exposing project modules under the ``osiris.llm_sidecar`` namespace."""

import importlib
import sys
import types

import pathlib
_base = importlib.import_module("llm_sidecar")
__all__ = list(getattr(_base, "__all__", dir(_base)))
__path__ = list(getattr(_base, "__path__", []))
current_dir = pathlib.Path(__file__).parent
if str(current_dir) not in __path__:
    __path__.append(str(current_dir))

from .. import server  # re-export for legacy tests

__all__ = ["server", *globals().get("__all__", [])]


def __getattr__(name):
    return getattr(_base, name)


# Lazily expose ``server`` module to avoid heavy imports during package load
try:
    _server_module = importlib.import_module("osiris.llm_sidecar.server")
except Exception:
    _server_module = types.ModuleType(__name__ + ".server")
sys.modules[__name__ + ".server"] = _server_module

__version__ = "0.2.1"

from .server import SidecarServer

__all__.append("SidecarServer")

from . import *
