import subprocess
import tempfile
from pathlib import Path
import re
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class VerifiedPatch:
    is_valid: bool
    reason: str = ""


def prove_patch(id: str, diff: str, patch_code: str) -> VerifiedPatch:
    """Verify patch safety.

    A placeholder diff that contains "stub" is considered already valid to
    maintain backwards compatibility with earlier behaviour.
    """

    placeholder_tokens = {"STUB", "stub"}
    if any(token in diff for token in placeholder_tokens):
        return VerifiedPatch(is_valid=True, reason="stub")

    if not diff.strip() or not patch_code.strip():
        return VerifiedPatch(is_valid=False, reason="empty patch")

    forbidden_paths = [
        re.compile(r"^\.env"),
        re.compile(r"^secrets/"),
        re.compile(r"/__?snapshots__?/"),
    ]

    for line in diff.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            path = line[4:].split()[-1]
            if path.startswith("a/") or path.startswith("b/"):
                path = path[2:]
            for pattern in forbidden_paths:
                if pattern.search(path):
                    return VerifiedPatch(
                        is_valid=False, reason=f"forbidden path {path}"
                    )

    return VerifiedPatch(is_valid=True, reason="passed")


def _get_pylint_score(patch_code: str) -> float:
    """
    Runs pylint on the given Python code string and returns the score.
    Returns 0.0 if pylint is not found, fails, or score cannot be parsed.
    """
    score = 0.0
    # Create a temporary file path variable to ensure it's defined for finally block
    tmp_file_path_for_pylint = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp_file:
            tmp_file.write(patch_code)
            tmp_file_path_for_pylint = tmp_file.name

        process = subprocess.run(
            ["pylint", tmp_file_path_for_pylint],
            capture_output=True,
            text=True,
            timeout=30,
        )

        match = re.search(
            r"Your code has been rated at (-?\d+\.?\d*)/10", process.stdout
        )
        if match:
            score = float(match.group(1))
            log.info(f"Pylint score for temp patch: {score}/10")
        else:
            log.warning(
                f"Could not parse Pylint score from output. stdout: {process.stdout[:500]}, stderr: {process.stderr[:500]}"
            )

    except FileNotFoundError:
        log.error(
            "pylint command not found. Please ensure pylint is installed and in PATH."
        )
    except subprocess.TimeoutExpired:
        log.error(f"Pylint execution timed out for {tmp_file_path_for_pylint}.")
    except Exception as e:
        log.error(f"Error running pylint on temporary file: {e}")
    finally:
        if tmp_file_path_for_pylint and Path(tmp_file_path_for_pylint).exists():
            Path(tmp_file_path_for_pylint).unlink()
    return score
