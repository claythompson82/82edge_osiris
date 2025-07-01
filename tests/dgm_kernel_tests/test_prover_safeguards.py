import difflib
from pathlib import Path

from dgm_kernel.prover import prove_patch


def test_bad_patch_does_not_modify_source(tmp_path) -> None:
    """
    A patch that introduces a syntax error should

    • receive a **low prove-score** (≤ 0.4), and
    • never touch the on-disk source file.
    """
    target = Path("src/dgm_kernel/__init__.py")
    original = target.read_text()

    # create an obviously broken variant
    broken = original + "\nsyntax_error(\n"

    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            broken.splitlines(keepends=True),
            fromfile=str(target),
            tofile=str(target),
        )
    )

    # prove_patch now returns a float confidence ∈ [0, 1]
    assert prove_patch(diff) <= 0.4

    # file on disk must stay pristine
    assert target.read_text() == original
