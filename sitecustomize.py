"""
Sitecustomize script to unconditionally modify Python's import path.
It ensures that the project's `src` directory is prioritized in `sys.path`.
"""
import sys
import os # Though os is not strictly needed for this version, it's common.
import pathlib

# It's good practice to have a log, even if simple, for diagnostics.
# Printing to stderr is conventional for such messages.
def _log_sitecustomize(message: str) -> None:
    print(f"LOG_SITECUSTOMIZE: {message}", file=sys.stderr)

_log_sitecustomize("Script start. Attempting to unconditionally prepend 'src' directory to sys.path.")

try:
    # Determine the repository root assuming sitecustomize.py is at the root.
    repo_root = pathlib.Path(__file__).resolve().parent
    src_dir = repo_root / "src"

    _log_sitecustomize(f"Determined repository root: {repo_root}")
    _log_sitecustomize(f"Target 'src' directory: {src_dir}")

    if src_dir.is_dir():
        src_dir_abs_str = str(src_dir) # sys.path uses strings

        # Check if already present to avoid duplicates or reordering if already first.
        # More robustly, remove all existing instances then prepend.
        # This ensures it's exactly at the front and only once.

        # Create a new list excluding any existing resolved paths to src_dir
        new_sys_path = [p for p in sys.path if pathlib.Path(p).resolve() != src_dir.resolve()]
        sys.path[:] = new_sys_path # Modify sys.path in place

        sys.path.insert(0, src_dir_abs_str)
        _log_sitecustomize(f"Successfully prepended '{src_dir_abs_str}' to sys.path.")
    else:
        _log_sitecustomize(f"WARNING: 'src' directory '{src_dir}' not found. sys.path not modified by this logic.")

except Exception as e:
    _log_sitecustomize(f"ERROR during execution: {e}")

_log_sitecustomize(f"Final sys.path: {sys.path}")

# Clean up globals from this script to avoid polluting user's namespace
del sys, os, pathlib, _log_sitecustomize # type: ignore[name-defined] # Assuming _log_sitecustomize is defined
# Also delete repo_root, src_dir, src_dir_abs_str, new_sys_path, e if they were in global scope and an error occurred
# For simplicity, only deleting the ones definitely created.
# Python functions clean their own local scopes.
# `e` would only be global if the try block itself was not in a function.
# Since this whole script is global, `repo_root` etc. are global.
try:
    del repo_root, src_dir, src_dir_abs_str, new_sys_path # type: ignore[name-defined]
except NameError: # In case they weren't all defined (e.g., src_dir not a dir)
    pass
try:
    del e # type: ignore[name-defined] # if an exception occurred
except NameError:
    pass
