import sys
sys.path.append('.')

import pytest
from hypothesis import given, strategies as st

from dgm_kernel.prover import prove_patch


@given(st.text().filter(lambda s: s.strip() == ""))
def test_empty_diffs_invalid(empty_diff):
    result = prove_patch(id="1", diff=empty_diff, patch_code="")
    assert not result.is_valid


@given(st.text(min_size=1).filter(lambda s: s.strip() != ""))
def test_valid_diff_passes(code):
    diff = "--- a/file.py\n+++ b/file.py\n@@\n-old\n+new"
    result = prove_patch(id="1", diff=diff, patch_code=code)
    assert result.is_valid


@given(
    st.sampled_from(["secrets/token.txt", ".env", "tests/unit/__snapshots__/out.snap"])
)
def test_forbidden_paths_fail(path):
    diff = f"--- a/{path}\n+++ b/{path}\n@@\n-old\n+new"
    result = prove_patch(id="1", diff=diff, patch_code="print('x')")
    assert not result.is_valid
