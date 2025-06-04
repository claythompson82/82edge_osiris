import subprocess
import tempfile
from pathlib import Path
import re
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

@dataclass
class VerifiedPatch:
    id: str
    diff: str
    score: float
    status: str


def prove_patch(id: str, diff: str, patch_code: str) -> VerifiedPatch:
    """
    Verifies the patch using pylint score.
    """
    pylint_score = _get_pylint_score(patch_code)
    status = "APPROVED" if pylint_score >= 9.0 else "REJECTED"
    return VerifiedPatch(id=id, diff=diff, score=pylint_score, status=status)

def _get_pylint_score(patch_code: str) -> float:
    """
    Runs pylint on the given Python code string and returns the score.
    Returns 0.0 if pylint is not found, fails, or score cannot be parsed.
    """
    score = 0.0
    # Create a temporary file path variable to ensure it's defined for finally block
    tmp_file_path_for_pylint = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
            tmp_file.write(patch_code)
            tmp_file_path_for_pylint = tmp_file.name

        process = subprocess.run(
            ["pylint", tmp_file_path_for_pylint],
            capture_output=True, text=True, timeout=30
        )

        match = re.search(r"Your code has been rated at (-?\d+\.?\d*)/10", process.stdout)
        if match:
            score = float(match.group(1))
            log.info(f"Pylint score for temp patch: {score}/10")
        else:
            log.warning(f"Could not parse Pylint score from output. stdout: {process.stdout[:500]}, stderr: {process.stderr[:500]}")

    except FileNotFoundError:
        log.error("pylint command not found. Please ensure pylint is installed and in PATH.")
    except subprocess.TimeoutExpired:
        log.error(f"Pylint execution timed out for {tmp_file_path_for_pylint}.")
    except Exception as e:
        log.error(f"Error running pylint on temporary file: {e}")
    finally:
        if tmp_file_path_for_pylint and Path(tmp_file_path_for_pylint).exists():
            Path(tmp_file_path_for_pylint).unlink()
    return score
