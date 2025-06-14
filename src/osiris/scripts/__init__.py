"""
osiris.scripts
==============

Make the *scripts* directory a normal Python package so callers can do::

    from osiris.scripts.harvest_feedback import main

We also *eager-import* the ``harvest_feedback`` module once on package import.
That guarantees its attributes (e.g. ``main``) are present even when test
runners muck with `sys.modules` or perform partial imports.

Nothing here is heavy-weight, so the extra import has negligible impact.
"""

from importlib import import_module as _import_module

# Public re-exports for IDEs / wildcard imports
__all__: list[str] = [
    "harvest_feedback",
]

# --------------------------------------------------------------------------- #
# Ensure the sub-module is fully loaded
# --------------------------------------------------------------------------- #
# This populates sys.modules["osiris.scripts.harvest_feedback"] and, crucially,
# binds its globals (including `main`) so that:
#
#     from osiris.scripts.harvest_feedback import main
#
# always works – regardless of import order tricks during testing.
_import_module(__name__ + ".harvest_feedback")
