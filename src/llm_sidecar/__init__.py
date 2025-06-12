"""Minimal shim so `import llm_sidecar.client` works in legacy tests."""
from importlib import import_module as _m
try:
    client = _m(".client", __name__)
except ModuleNotFoundError:
    pass
