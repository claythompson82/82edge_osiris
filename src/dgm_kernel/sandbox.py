import subprocess
import tempfile
import shutil
from pathlib import Path
import logging

log = logging.getLogger(__name__)


def run_patch_in_sandbox(
    patch: dict,
    repo_root: Path | None = None,
    image: str = "python:3.11-slim",
    memory: str = "512m",
    cpus: str | int = "1",
) -> tuple[bool, str, int]:
    """Apply a patch to a fresh repo copy and execute it in an isolated Docker container.

    Returns (passed, logs, exit_code).
    """
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_repo = Path(tmpdir) / "repo"
            shutil.copytree(repo_root, tmp_repo, dirs_exist_ok=True)

            rel_target = Path(patch["target"])
            if rel_target.is_absolute():
                rel_target = rel_target.relative_to(repo_root)
            target_path = tmp_repo / rel_target
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(patch["after"])

            module_name = str(rel_target.with_suffix("")).replace("/", ".")
            cmd = [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--memory",
                str(memory),
                "--cpus",
                str(cpus),
                "-v",
                f"{tmp_repo}:/app",
                image,
                "python",
                "-c",
                f"import {module_name}",
            ]
            log.info("Sandbox command: %s", " ".join(cmd))
            process = subprocess.run(cmd, capture_output=True, text=True)
            logs = process.stdout + process.stderr
            return process.returncode == 0, logs, process.returncode
    except FileNotFoundError:
        log.error("Docker not found. Ensure Docker is installed for sandbox testing.")
        return False, "docker missing", -1
    except Exception as e:
        log.error(f"Sandbox execution failed: {e}")
        return False, str(e), -1
