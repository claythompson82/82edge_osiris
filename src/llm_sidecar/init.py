# ... existing contents ...

import sys
import types

# --------------------------------------------------------------------------- #
# The *public* test-suite imports through ``osiris.llm_sidecar``.  We expose
# our real sub-packages there so that patch-paths resolve correctly.
# --------------------------------------------------------------------------- #
_pkg = types.ModuleType("osiris.llm_sidecar")
sys.modules.setdefault("osiris.llm_sidecar", _pkg)
sys.modules.setdefault("osiris.llm_sidecar.loader", loader)
sys.modules.setdefault("osiris.llm_sidecar.db", db)
