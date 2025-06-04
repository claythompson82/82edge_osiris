import importlib

# Basic smoke test to ensure main packages import without error

def test_import_osiris_packages():
    importlib.import_module('osiris.llm_sidecar')
    importlib.import_module('osiris.llm_sidecar.db')
    importlib.import_module('osiris_policy')
    importlib.import_module('dgm_kernel')
