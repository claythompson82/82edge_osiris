import subprocess
import os
import sys
from pathlib import Path


def test_driver_marketforge_task_set():
    """
    Tests that running driver.py with --task-set marketforge logs "trainer loop stub".
    """
    # Resolve the repository root based on this test file's location. ``tests``
    # lives at the project root, so ``parents[1]`` should point to the root
    # directory when the code is checked out.  This avoids walking the entire
    # filesystem while still working when the tests are executed from an
    # installed location.
    repo_root = Path(__file__).resolve().parents[1]

    # Prefer ``nightly_trainer/driver.py`` at the project root.  Fall back to a
    # scripts directory if present.
    candidate_paths = [
        repo_root / "nightly_trainer" / "driver.py",
        repo_root / "scripts" / "nightly_trainer" / "driver.py",
    ]

    driver_script_path = next((p for p in candidate_paths if p.exists()), None)

    # Ensure the script path was found
    if not driver_script_path:
        searched = ", ".join(str(p) for p in candidate_paths)
        raise FileNotFoundError(f"Could not find driver.py. Looked in: {searched}")

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
