import subprocess
import os
import sys

def test_driver_marketforge_task_set():
    """
    Tests that running driver.py with --task-set marketforge logs "trainer loop stub".
    """
    # Construct the path to the driver.py script relative to this test file.
    # This assumes the tests directory is at the root of the project,
    # and nightly_trainer is also at the root.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    driver_script_path = os.path.join(project_root, "nightly_trainer", "driver.py")

    # Ensure the script path is correct and exists
    if not os.path.exists(driver_script_path):
        raise FileNotFoundError(f"Could not find driver script at {driver_script_path}")

    try:
        # Run the script using the same Python interpreter that's running the tests
        process = subprocess.run(
            [sys.executable, driver_script_path, "--task-set", "marketforge"],
            capture_output=True,
            text=True,
            check=False,  # Check manually to provide better error message
            timeout=10 # Add a timeout to prevent tests from hanging
        )

        # Combine stdout and stderr for checking the log message
        output = process.stdout + process.stderr

        if process.returncode != 0:
            print(f"Error running script. Exit code: {process.returncode}")
            print(f"Stdout:\n{process.stdout}")
            print(f"Stderr:\n{process.stderr}")

        assert process.returncode == 0, f"Script exited with {process.returncode}, output: {output}"
        assert "trainer loop stub" in output, f"'trainer loop stub' not found in output: {output}"

    except FileNotFoundError:
        # This might happen if sys.executable or driver_script_path is incorrect
        assert False, f"Could not find Python interpreter or script. Searched for script at {driver_script_path}"
    except subprocess.TimeoutExpired:
        assert False, "Running the driver script timed out."
    except Exception as e:
        assert False, f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    # This allows running the test directly e.g., python tests/test_nightly_trainer.py
    # For more comprehensive test runs, a test runner like pytest is recommended.
    test_driver_marketforge_task_set()
    print("Test passed.")
