import subprocess
import os
import sys
from pathlib import Path


def test_driver_marketforge_task_set():
    """
    Tests that running driver.py with --task-set marketforge logs "trainer loop stub".
    """
    # Attempt to locate driver.py by walking up the directory tree until the
    # nightly_trainer directory is found. This is more robust than assuming a fixed depth.
    search_start = Path(__file__).resolve()
    driver_script_path = None
    for parent in [search_start] + list(search_start.parents):
        # Check for <root>/nightly_trainer/driver.py
        candidate = parent / "nightly_trainer" / "driver.py"
        if candidate.exists():
            driver_script_path = candidate
            break
        # Also check for <root>/scripts/nightly_trainer/driver.py
        alt_candidate = parent / "scripts" / "nightly_trainer" / "driver.py"
        if alt_candidate.exists():
            driver_script_path = alt_candidate
            break

    # Ensure the script path was found
    if not driver_script_path or not driver_script_path.exists():
        raise FileNotFoundError(f"Could not find driver.py. Searched from {search_start}.")

    try:
        # Run the script using the same Python interpreter that's running the tests
        process = subprocess.run(
            [sys.executable, driver_script_path, "--task-set", "marketforge"],
            capture_output=True,
            text=True,
            check=False,  # Check manually to provide better error message
            timeout=10,  # Add a timeout to prevent tests from hanging
        )

        # Combine stdout and stderr for checking the log message
        output = process.stdout + process.stderr

        if process.returncode != 0:
            print(f"Error running script. Exit code: {process.returncode}")
            print(f"Stdout:\n{process.stdout}")
            print(f"Stderr:\n{process.stderr}")

        assert (
            process.returncode == 0
        ), f"Script exited with {process.returncode}, output: {output}"
        assert (
            "trainer loop stub" in output
        ), f"'trainer loop stub' not found in output: {output}"

    except FileNotFoundError:
        # This might happen if sys.executable or driver_script_path is incorrect
        assert (
            False
        ), f"Could not find Python interpreter or script. Searched for script at {driver_script_path}"
    except subprocess.TimeoutExpired:
        assert False, "Running the driver script timed out."
    except Exception as e:
        assert False, f"An unexpected error occurred: {e}"


if __name__ == "__main__":
    # This allows running the test directly e.g., python tests/test_nightly_trainer.py
    test_driver_marketforge_task_set()
    print("Test passed.")