import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import os
import importlib
import logging # For caplog

# Assuming osiris.server and llm_sidecar.loader are accessible
import osiris.server
import llm_sidecar.loader as loader_module

# Fixture to manage OSIRIS_TEST environment variable for this test module
@pytest.fixture(scope="function", autouse=True) # Changed scope to function
def manage_osiris_test_env_var_for_adapter_swap_module(monkeypatch):
    original_value = os.environ.get("OSIRIS_TEST")
    monkeypatch.setenv("OSIRIS_TEST", "1")

    importlib.reload(osiris.server)
    # Make sure loader_module globals are reset for each module run if they cache state
    importlib.reload(loader_module)

    # Update the app instance for the test client
    global app_instance_for_adapter_swap
    app_instance_for_adapter_swap = osiris.server.app

    yield

    if original_value is None:
        monkeypatch.delenv("OSIRIS_TEST", raising=False)
    else:
        monkeypatch.setenv("OSIRIS_TEST", original_value)

    importlib.reload(osiris.server)
    importlib.reload(loader_module)
    app_instance_for_adapter_swap = osiris.server.app

app_instance_for_adapter_swap = osiris.server.app # Initial app instance

@pytest.fixture
def test_client_adapter_swap() -> TestClient:
    return TestClient(app_instance_for_adapter_swap)


def test_bad_adapter_fallback(
    test_client_adapter_swap: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog
) -> None:
    """
    Tests adapter hot-swap fallback logic:
    - A bad 'latest' adapter exists.
    - A '.last_good' adapter exists and is valid.
    - System should log error for bad adapter, then successfully load and use the good fallback.
    - Health check's 'latest_adapter' should still report the latest directory by name.
    - Subsequent attempts to load the bad adapter should log a different message (log once behavior).
    - If '.last_good' points to a bad/missing adapter, loading should ultimately fail.
    """
    caplog.set_level(logging.INFO) # Capture info and error logs from loader module

    test_adapter_root = tmp_path / "adapters_swap_test"
    test_adapter_root.mkdir()

    last_good_file_path = test_adapter_root / ".last_good"

    monkeypatch.setattr(loader_module, "ADAPTER_ROOT", test_adapter_root)
    monkeypatch.setattr(loader_module, "LAST_GOOD_ADAPTER_FILE", last_good_file_path)

    # Reset loader's cached state
    monkeypatch.setattr(loader_module, "_current_adapter_components", None)
    monkeypatch.setattr(loader_module, "_current_adapter_path", None)
    # Accessing the global via getattr for safety if it was somehow not initialized yet by a direct import in loader
    _logged_bad_adapter_paths_ref = getattr(loader_module, "_logged_bad_adapter_paths", set())
    _logged_bad_adapter_paths_ref.clear()


    # 1. Create a "last good" adapter
    good_adapter_dir = test_adapter_root / "2023-01-01_good_adapter"
    good_adapter_dir.mkdir()
    # (No 'simulate_load_failure.txt' means it's good for the placeholder loader)
    with open(last_good_file_path, "w") as f:
        f.write(str(good_adapter_dir))

    # 2. Create a "latest" but "bad" adapter directory (chronologically later)
    bad_adapter_dir = test_adapter_root / "2023-01-02_bad_adapter"
    bad_adapter_dir.mkdir()
    with open(bad_adapter_dir / "simulate_load_failure.txt", "w") as f:
        f.write("This adapter is intentionally broken.")

    # 3. Trigger adapter loading (e.g., by calling a model getter)
    loaded_components = loader_module.get_phi3_model_and_tokenizer()

    # Assertions for successful fallback:
    assert loaded_components is not None, "Fallback should have loaded the good adapter"
    model, tokenizer = loaded_components
    assert model == f"model_from_{good_adapter_dir.name}", "Model not from good fallback adapter"
    assert tokenizer == f"tokenizer_from_{good_adapter_dir.name}", "Tokenizer not from good fallback adapter"

    assert any(
        f"Failed to load latest adapter '{bad_adapter_dir}'" in record.message and record.levelname == "ERROR"
        for record in caplog.records
    ), "Error log for the bad 'latest' adapter is missing"

    assert any(
        f"Found last good adapter path: {good_adapter_dir}" in record.message and record.levelname == "INFO"
        for record in caplog.records
    ), "Log for finding and attempting '.last_good' path is missing"
    assert any(
        f"Successfully loaded adapter from: {good_adapter_dir}" in record.message and record.levelname == "INFO"
        for record in caplog.records
    ), "Log for successful load of the good fallback adapter is missing"

    # 4. Health check: 'latest_adapter' field should show the name of the latest *directory* found by get_latest_adapter_dir()
    # even if loading it failed and fallback occurred.
    health_response = test_client_adapter_swap.get("/health?adapter_date=true")
    assert health_response.status_code == 200
    health_data = health_response.json()
    assert health_data.get("latest_adapter") == bad_adapter_dir.name # get_latest_adapter_dir finds the newest dir name

    # 5. Test "log once" behavior for the bad adapter path
    _logged_bad_adapter_paths_ref.clear() # Clear log history for this specific check
    caplog.clear() # Clear previous log records

    loader_module.get_phi3_model_and_tokenizer() # First attempt after clear, should log ERROR for bad_adapter_dir

    count_error_log_bad_adapter = sum(
        1 for rec in caplog.records if f"Failed to load latest adapter '{bad_adapter_dir}'" in rec.message and rec.levelname == "ERROR"
    )
    assert count_error_log_bad_adapter == 1, "Bad adapter error should be logged on the first attempt"

    caplog.clear() # Clear again to check next call
    loader_module.get_phi3_model_and_tokenizer() # Second attempt

    count_error_log_bad_adapter_second = sum(
        1 for rec in caplog.records if f"Failed to load latest adapter '{bad_adapter_dir}'" in rec.message and rec.levelname == "ERROR"
    )
    assert count_error_log_bad_adapter_second == 0, "Bad adapter error should NOT be logged again for the same path"

    assert any(
        f"Still failing to load adapter '{bad_adapter_dir}'" in record.message and record.levelname == "WARNING"
        for record in caplog.records
    ), "Warning for repeated failure on bad adapter path was not logged"


    # 6. Test scenario: .last_good file points to a non-existent/bad directory
    _logged_bad_adapter_paths_ref.clear()
    caplog.clear()
    monkeypatch.setattr(loader_module, "_current_adapter_components", None)
    monkeypatch.setattr(loader_module, "_current_adapter_path", None)

    non_existent_good_path = test_adapter_root / "2022-12-31_non_existent_good_adapter"
    # (This directory non_existent_good_path is not created)
    with open(last_good_file_path, "w") as f:
        f.write(str(non_existent_good_path))

    # Latest is still the bad_adapter_dir. Fallback will now also fail.
    final_components = loader_module.get_phi3_model_and_tokenizer()
    assert final_components is None, "Loading should ultimately fail and return None"

    assert any(
        f"Failed to load latest adapter '{bad_adapter_dir}'" in record.message and record.levelname == "ERROR"
        for record in caplog.records
    ), "Error for bad 'latest' adapter missing when fallback also fails"
    assert any(
        f"Failed to load fallback adapter" in record.message and str(non_existent_good_path) in record.message and record.levelname == "ERROR"
        for record in caplog.records
    ), "Error for non-existent/bad fallback adapter path missing"
    assert any(
        "No adapter could be loaded" in record.message and record.levelname == "ERROR"
        for record in caplog.records
    ), "Final 'No adapter could be loaded' error log missing"
