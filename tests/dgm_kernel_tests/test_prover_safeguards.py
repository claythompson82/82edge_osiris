import difflib
from pathlib import Path

from dgm_kernel.prover import prove_patch


def test_bad_patch_does_not_modify_source(tmp_path):
    target = Path("src/dgm_kernel/__init__.py")
    original = target.read_text()

    broken = original + "\nsyntax_error(\n"
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            broken.splitlines(keepends=True),
            fromfile=str(target),
            tofile=str(target),
        )
    )

    assert prove_patch(diff) <= 0.4
    assert target.read_text() == original
