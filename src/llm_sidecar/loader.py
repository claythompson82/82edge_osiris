"""
Stub loader: the real implementation would return actual models.

In the test-suite the two `get_*_model_and_tokenizer` functions are **always
monkey-patched**, so the bodies below never run.  They only exist so that
importing this module doesn’t crash.

`get_latest_adapter_dir()` is called by the /health endpoint.
"""

from __future__ import annotations

import os
import logging # Added
from pathlib import Path
from typing import Any, Tuple, Optional, Set # Added Optional, Set

logger = logging.getLogger(__name__) # Added logger
_logged_bad_adapter_paths: Set[Path] = set() # Added for "log once"

# Default root – tests will monkey-patch this to a tmp directory.
ADAPTER_ROOT = Path(os.getenv("ADAPTER_ROOT", "/adapters"))
LAST_GOOD_ADAPTER_FILE = ADAPTER_ROOT / ".last_good" # Define .last_good file path


# Placeholder for what an adapter might represent or return upon loading
class LoadedAdapterComponents(Tuple[Any, Any]): # Simple tuple for now
    pass

def _load_adapter_dynamically(adapter_path: Path) -> Optional[LoadedAdapterComponents]: # Placeholder
    """
    Placeholder for dynamically loading adapter components from a given path.
    In a real scenario, this would involve importing modules or loading model files.
    Returns some representation of the loaded adapter, or raises Exception on failure.
    For this stub, we'll just check if the path is a directory.
    """
    logger.info(f"Attempting to load adapter from: {adapter_path}")
    if not adapter_path.exists() or not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter path {adapter_path} does not exist or is not a directory.")

    # Simulate loading success by returning some dummy components
    # In a real implementation, this would load models, tokenizers, etc.
    # For testing the guard, the important part is whether this function
    # returns successfully or raises an exception.

    # Simulate a "bad" adapter if a specific file exists (for testing fallback)
    if (adapter_path / "simulate_load_failure.txt").exists():
        raise ValueError(f"Simulated failure loading adapter: {adapter_path}")

    # Simulate successful load
    # These would be actual model and tokenizer components
    dummy_model = f"model_from_{adapter_path.name}"
    dummy_tokenizer = f"tokenizer_from_{adapter_path.name}"
    logger.info(f"Successfully loaded adapter from: {adapter_path}")
    return LoadedAdapterComponents((dummy_model, dummy_tokenizer))


_current_adapter_components: Optional[LoadedAdapterComponents] = None
_current_adapter_path: Optional[Path] = None


def get_active_adapter_components() -> Optional[LoadedAdapterComponents]:
    """
    Gets the active adapter components, trying to load the latest,
    then falling back to the last known good adapter.
    Implements the hot-swap guard logic.
    """
    global _current_adapter_components, _current_adapter_path, _logged_bad_adapter_paths

    latest_adapter_dir = get_latest_adapter_dir()
    adapter_to_load_path: Optional[Path] = None
    is_fallback_attempt = False

    if latest_adapter_dir:
        adapter_to_load_path = latest_adapter_dir
        logger.info(f"Attempting to load latest adapter: {adapter_to_load_path}")
        try:
            loaded_components = _load_adapter_dynamically(adapter_to_load_path)
            if loaded_components:
                _current_adapter_components = loaded_components
                _current_adapter_path = adapter_to_load_path
                # Successfully loaded the latest, update .last_good
                try:
                    LAST_GOOD_ADAPTER_FILE.parent.mkdir(parents=True, exist_ok=True)
                    LAST_GOOD_ADAPTER_FILE.write_text(str(adapter_to_load_path))
                    logger.info(f"Updated .last_good_adapter_path to {adapter_to_load_path}")
                except IOError as e_write:
                    logger.error(f"Failed to write .last_good_adapter_path file: {e_write}")
                _logged_bad_adapter_paths.clear() # Clear previous bad paths as we have a new good one
                return _current_adapter_components
        except Exception as e:
            log_key = adapter_to_load_path
            if log_key not in _logged_bad_adapter_paths:
                logger.error(f"Failed to load latest adapter '{adapter_to_load_path}': {e}. Attempting fallback.")
                _logged_bad_adapter_paths.add(log_key) # Log once for this path
            else:
                logger.warning(f"Still failing to load adapter '{adapter_to_load_path}'. Fallback already attempted or in progress.")
            # Proceed to fallback logic
            adapter_to_load_path = None # Reset to ensure fallback logic is triggered correctly

    # Fallback logic (if latest_adapter_dir was None or loading it failed)
    if adapter_to_load_path is None: # Indicates we need to try fallback
        is_fallback_attempt = True
        logger.info("Attempting to load from .last_good_adapter_path")
        try:
            if LAST_GOOD_ADAPTER_FILE.exists():
                last_good_path_str = LAST_GOOD_ADAPTER_FILE.read_text().strip()
                last_good_path = Path(last_good_path_str)
                logger.info(f"Found last good adapter path: {last_good_path}")

                # Avoid reloading if it's the same as current and current is already good
                if _current_adapter_path == last_good_path and _current_adapter_components is not None:
                    logger.info(f"Already using the last good adapter: {last_good_path}. No reload needed.")
                    return _current_adapter_components

                loaded_components = _load_adapter_dynamically(last_good_path)
                if loaded_components:
                    _current_adapter_components = loaded_components
                    _current_adapter_path = last_good_path
                    # Do not clear _logged_bad_adapter_paths here, as the latest might still be bad
                    return _current_adapter_components
            else:
                logger.warning(".last_good_adapter_path file not found. No fallback available.")
        except Exception as e_fallback:
            logger.error(f"Failed to load fallback adapter: {e_fallback}")

    if _current_adapter_components:
        logger.info(f"Continuing to use previously loaded/fallback adapter: {_current_adapter_path}")
        return _current_adapter_components

    logger.error("No adapter could be loaded (latest or fallback).")
    return None


def get_latest_adapter_dir() -> Path | None:
    """
    Return the most-recent adapter subdir – or **None** if none exist.

    We suppress permissions errors because the sandbox can’t create /adapters.
    """
    try:
        if not ADAPTER_ROOT.exists():
            return None
        subdirs = [p for p in ADAPTER_ROOT.iterdir() if p.is_dir()]
        return max(subdirs, default=None)
    except PermissionError:
        return None


# --- Model helpers – these will now use the guarded adapter loader -----------

# TODO(AZR-07): The actual structure of LoadedAdapterComponents and how specific models
# (phi3 vs hermes) are selected/returned needs to be defined.
# For now, assuming get_active_adapter_components might return components for a specific model type,
# or a generic one that these functions can adapt.
# This simplified version assumes get_active_adapter_components provides THE model and tokenizer.

def get_phi3_model_and_tokenizer() -> Optional[LoadedAdapterComponents]:
    # In a real system, this might involve logic to ensure the loaded adapter
    # is specifically for phi3, or get_active_adapter_components would take a model name.
    # For now, just return whatever the active adapter provides.
    components = get_active_adapter_components()
    if components:
        # Here we'd ideally check if components[0] is a phi3 model, components[1] its tokenizer
        logger.info(f"PHI3: Returning components from adapter: {_current_adapter_path}")
        return components
    logger.error("PHI3: No active adapter components available.")
    return None


def get_hermes_model_and_tokenizer() -> Optional[LoadedAdapterComponents]:
    # Similar to phi3, this would need logic for selecting/validating hermes components.
    components = get_active_adapter_components()
    if components:
        logger.info(f"Hermes: Returning components from adapter: {_current_adapter_path}")
        return components
    logger.error("Hermes: No active adapter components available.")
    return None
