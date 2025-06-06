#!/usr/bin/env python3

import os
import subprocess
import time
import random
from datetime import datetime

# --- Configuration ---
SERVICES = ["llm-sidecar", "orchestrator"]
ORCHESTRATOR_SCRIPT_PATH = "osiris_policy/orchestrator.py"  # Relative to repo root
MIN_SLEEP_SECONDS = 90
MAX_SLEEP_SECONDS = 180
# REPO_ROOT needs to be set if the script is not run from the root.
# For now, we assume it's run from the root.
REPO_ROOT = os.getcwd()  # Or use a fixed path if necessary, e.g., "/app"

# --- Helper Functions ---


def log_message(message):
    """Prints a message with a timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def check_chaos_mode():
    """Checks if CHAOS_MODE is enabled."""
    log_message("Checking CHAOS_MODE environment variable...")
    if os.environ.get("CHAOS_MODE") != "1":
        log_message("CHAOS_MODE is not set to '1'. Exiting.")
        exit(0)
    log_message("CHAOS_MODE is enabled. Proceeding with chaos.")


def get_orchestrator_cmd_and_pid():
    """
    Finds the PID and the full command line of the orchestrator script.
    Returns (command, pid) or (None, None) if not found.
    """
    try:
        # Find PIDs of processes matching the orchestrator script
        # Using pgrep to find PIDs more reliably
        pgrep_cmd = ["pgrep", "-f", f"python.*{ORCHESTRATOR_SCRIPT_PATH}"]
        result_pid = subprocess.run(
            pgrep_cmd, capture_output=True, text=True, check=False
        )

        pids = result_pid.stdout.strip().split("\n")
        if not pids or not pids[0]:
            log_message(f"No PID found for {ORCHESTRATOR_SCRIPT_PATH} using pgrep.")
            return None, None

        # Iterate through PIDs to find the exact command (pgrep can be too broad)
        for pid in pids:
            if not pid.strip():
                continue
            try:
                # Get the command line for the PID
                ps_cmd = ["ps", "-o", "cmd=", "-p", pid.strip()]
                result_cmd = subprocess.run(
                    ps_cmd, capture_output=True, text=True, check=True
                )
                command = result_cmd.stdout.strip()

                # Verify if the command indeed runs the orchestrator script
                if (
                    ORCHESTRATOR_SCRIPT_PATH in command
                    and "python" in command.split()[0].lower()
                ):
                    log_message(
                        f"Found orchestrator PID: {pid} with command: {command}"
                    )
                    return command, pid.strip()
            except subprocess.CalledProcessError as e:
                log_message(
                    f"Error checking command for PID {pid}: {e}. Output: {e.stderr}"
                )
            except Exception as e:
                log_message(
                    f"An unexpected error occurred while checking command for PID {pid}: {e}"
                )

        log_message(
            f"Could not confirm exact command for PIDs {pids} matching {ORCHESTRATOR_SCRIPT_PATH}."
        )
        return None, None

    except subprocess.CalledProcessError as e:
        log_message(
            f"Error finding orchestrator PID with pgrep: {e}. Stderr: {e.stderr}"
        )
        return None, None
    except Exception as e:
        log_message(
            f"An unexpected error occurred in get_orchestrator_cmd_and_pid: {e}"
        )
        return None, None


def restart_llm_sidecar():
    """Restarts the llm-sidecar service using docker-compose."""
    log_message("Attempting to restart llm-sidecar...")
    try:
        # Assuming docker-compose.yaml is in REPO_ROOT or docker/compose.yaml
        # For now, let's assume 'docker-compose restart' works directly.
        # A more robust solution might involve specifying the compose file path.
        # e.g., ['docker-compose', '-f', os.path.join(REPO_ROOT, 'docker-compose.yaml'), 'restart', 'llm-sidecar']
        # or ['docker-compose', '-f', os.path.join(REPO_ROOT, 'docker/compose.yaml'), 'restart', 'llm-sidecar']

        # First, check if the service exists and get project name
        ps_command = ["docker-compose", "ps", "-q", "llm-sidecar"]
        log_message(f"Running: {' '.join(ps_command)}")
        service_id = subprocess.run(
            ps_command, capture_output=True, text=True, cwd=REPO_ROOT, check=False
        )

        if service_id.returncode != 0 or not service_id.stdout.strip():
            log_message(
                f"llm-sidecar service not found or 'docker-compose ps' failed. Error: {service_id.stderr}"
            )
            # Try to find compose file in common locations if first try fails
            common_compose_files = [
                os.path.join(REPO_ROOT, "docker-compose.yaml"),
                os.path.join(REPO_ROOT, "docker/compose.yaml"),
                os.path.join(REPO_ROOT, "compose.yaml"),
            ]
            compose_file_to_use = None
            for cf in common_compose_files:
                if os.path.exists(cf):
                    ps_command_with_file = [
                        "docker-compose",
                        "-f",
                        cf,
                        "ps",
                        "-q",
                        "llm-sidecar",
                    ]
                    log_message(
                        f"Trying with compose file: {' '.join(ps_command_with_file)}"
                    )
                    service_id = subprocess.run(
                        ps_command_with_file,
                        capture_output=True,
                        text=True,
                        cwd=REPO_ROOT,
                        check=False,
                    )
                    if service_id.returncode == 0 and service_id.stdout.strip():
                        compose_file_to_use = cf
                        break

            if not compose_file_to_use:
                log_message(
                    "Could not find a valid docker-compose file for llm-sidecar. Cannot restart."
                )
                return

            restart_command = [
                "docker-compose",
                "-f",
                compose_file_to_use,
                "restart",
                "llm-sidecar",
            ]
        else:
            restart_command = ["docker-compose", "restart", "llm-sidecar"]

        log_message(f"Executing: {' '.join(restart_command)}")
        result = subprocess.run(
            restart_command, capture_output=True, text=True, cwd=REPO_ROOT, check=True
        )
        log_message(f"llm-sidecar restart initiated. Output: {result.stdout.strip()}")
        if result.stderr.strip():
            log_message(f"llm-sidecar restart stderr: {result.stderr.strip()}")
    except subprocess.CalledProcessError as e:
        log_message(f"Failed to restart llm-sidecar. Return code: {e.returncode}")
        log_message(f"Command: {' '.join(e.cmd)}")
        log_message(f"Stdout: {e.stdout.strip()}")
        log_message(f"Stderr: {e.stderr.strip()}")
    except FileNotFoundError:
        log_message(
            "Error: docker-compose command not found. Is it installed and in PATH?"
        )
    except Exception as e:
        log_message(f"An unexpected error occurred while restarting llm-sidecar: {e}")


def restart_orchestrator():
    """Kills and restarts the orchestrator script."""
    log_message("Attempting to restart orchestrator...")

    original_command, pid = get_orchestrator_cmd_and_pid()

    if pid:
        try:
            log_message(f"Killing orchestrator (PID: {pid})...")
            kill_cmd = ["kill", pid]
            subprocess.run(kill_cmd, check=True)
            log_message(f"Process {pid} killed.")
            time.sleep(3)  # Give it a moment to release resources
        except subprocess.CalledProcessError as e:
            log_message(
                f"Failed to kill orchestrator (PID: {pid}): {e}. It might have already exited."
            )
        except Exception as e:
            log_message(
                f"An unexpected error occurred while killing orchestrator PID {pid}: {e}"
            )
    else:
        log_message(
            "Orchestrator PID not found. Will attempt to start if command is known (e.g. from a previous run or default)."
        )

    if original_command:
        log_message(f"Restarting orchestrator with command: {original_command}")
        try:
            # Ensure the command is run from the repository root
            # Split the command string into a list for Popen
            # Example: "python osiris_policy/orchestrator.py --arg1 value1"
            # We need to make sure this is run in the background and output is redirected
            # to avoid filling up console or hanging the chaos script.
            # Using nohup is a common way for this, but Popen with detached=True
            # and redirection is more platform-agnostic if we avoid shell=True.
            # For simplicity and typical Linux environments:

            # Ensure relative paths in the command are based on REPO_ROOT
            # If original_command is like "python somescript.py" and somescript.py is relative,
            # it needs to be found from REPO_ROOT.
            # Popen's `cwd` argument handles this.

            # Ensure the command is split correctly, especially if it contains spaces in arguments.
            # For now, shlex.split might be safer if arguments can have spaces, but
            # basic command.split() works if args are simple.
            # Given `ps` output, it's usually space-separated.
            cmd_parts = original_command.split()

            # Start the process in the background
            # Redirect stdout and stderr to /dev/null to avoid clutter and potential blocking
            with open(os.devnull, "wb") as devnull:
                subprocess.Popen(
                    cmd_parts,
                    cwd=REPO_ROOT,
                    stdout=devnull,
                    stderr=devnull,
                    start_new_session=True,
                )
            log_message(
                f"Orchestrator restart initiated with command: {' '.join(cmd_parts)}"
            )

        except Exception as e:
            log_message(
                f"Failed to restart orchestrator with command '{original_command}': {e}"
            )
    else:
        log_message("Original command for orchestrator not found. Cannot restart.")
        log_message(
            f"Consider starting it manually if needed, e.g., 'cd {REPO_ROOT} && python {ORCHESTRATOR_SCRIPT_PATH} --your-args &'"
        )


# --- Main Loop ---


def main():
    """Main loop for the chaos script."""
    check_chaos_mode()

    log_message(f"Chaos script started. Will randomly restart: {', '.join(SERVICES)}.")
    log_message(f"Processes will be restarted from: {REPO_ROOT}")
    log_message(
        f"Sleep interval between restarts: {MIN_SLEEP_SECONDS} to {MAX_SLEEP_SECONDS} seconds."
    )

    if ORCHESTRATOR_SCRIPT_PATH.startswith("/"):
        log_message(
            f"Warning: ORCHESTRATOR_SCRIPT_PATH ('{ORCHESTRATOR_SCRIPT_PATH}') is absolute. "
            "Ensure this is intended and correct for the environment."
        )

    # Attempt to find initial orchestrator command once, in case it's not running but we want to start it.
    # This behavior is debatable for a "chaos" script (should it start services not running?)
    # For now, the restart_orchestrator function will only restart if it was found running.
    # If we want to start it even if not running, we'd need a default command.
    # _, initial_orchestrator_pid = get_orchestrator_cmd_and_pid()
    # if not initial_orchestrator_pid:
    #    log_message("Orchestrator does not seem to be running at script start.")

    try:
        while True:
            selected_service = random.choice(SERVICES)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            log_message(
                f"Selected service to restart: {selected_service} at {current_time}"
            )

            if selected_service == "llm-sidecar":
                restart_llm_sidecar()
            elif selected_service == "orchestrator":
                restart_orchestrator()
            else:
                log_message(f"Unknown service: {selected_service}. Skipping.")

            sleep_duration = random.randint(MIN_SLEEP_SECONDS, MAX_SLEEP_SECONDS)
            log_message(f"Sleeping for {sleep_duration} seconds...")
            time.sleep(sleep_duration)

    except KeyboardInterrupt:
        log_message("Chaos script interrupted by user. Exiting.")
    except Exception as e:
        log_message(f"An unexpected error occurred in the main loop: {e}")
        log_message("Exiting due to error.")
    finally:
        log_message("Chaos script finished.")


if __name__ == "__main__":
    main()
