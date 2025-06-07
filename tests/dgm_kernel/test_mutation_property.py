import string
import keyword
from pathlib import Path
import importlib

from hypothesis import given, strategies as st, settings, HealthCheck

from dgm_kernel.meta_loop import _apply_patch, _rollback


ident = st.text(alphabet=string.ascii_lowercase, min_size=1).filter(lambda s: s not in keyword.kwlist)
code_strategy = st.builds(lambda name, val: f"{name} = {val}", ident, st.integers())


@given(before_code=code_strategy, after_code=code_strategy)
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_apply_and_rollback_roundtrip(tmp_path, before_code, after_code):
    target = tmp_path / "module.py"
    patch = {"target": str(target), "before": before_code, "after": after_code}
    target.write_text(before_code)

    assert _apply_patch(patch) is True
    assert target.read_text() == after_code

    _rollback(patch)
    assert target.read_text() == before_code


@given(after_code=code_strategy)
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_apply_creates_file(tmp_path, after_code):
    target = tmp_path / "new_module.py"
    patch = {"target": str(target), "before": "", "after": after_code}
    assert _apply_patch(patch) is True
    assert target.exists()
    assert target.read_text() == after_code
