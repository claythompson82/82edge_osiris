"""Wrapper package exposing project modules under the ``osiris.llm_sidecar`` namespace."""

import importlib
import sys
import types

_base = importlib.import_module("llm_sidecar")
__all__ = getattr(_base, "__all__", dir(_base))
__path__ = getattr(_base, "__path__", [])

def __getattr__(name):
    return getattr(_base, name)

# Lazily expose ``server`` module to avoid heavy imports during package load
_server_module = types.ModuleType(__name__ + ".server")

def _server_getattr(name):
    server_mod = importlib.import_module("server")
    return getattr(server_mod, name)

_server_module.__getattr__ = _server_getattr
sys.modules[__name__ + ".server"] = _server_module
