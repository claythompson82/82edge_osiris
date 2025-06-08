"""osiris package root."""

__version__ = "0.2.1"

# Avoid importing any heavy submodules at package import time so that tests that
# merely import ``osiris`` don't pull in optional runtime dependencies.  The
# ``llm_sidecar`` and ``server`` submodules can still be imported explicitly via
# ``from osiris import llm_sidecar`` or ``from osiris import server`` thanks to
# Python's package import mechanics.

__all__ = ["llm_sidecar", "server"]
