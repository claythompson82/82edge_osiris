from hypothesis import given, strategies as st, settings, HealthCheck
from dgm_kernel.sandbox import Sandbox


def test_sandbox_success(tmp_path) -> None:
    patch = {"target": str(tmp_path / "m.py"), "after": "print('ok')"}
    passed, logs, code = Sandbox().run(patch)
    assert passed is True
    assert code == 0


def test_sandbox_failure(tmp_path) -> None:
    patch = {"target": str(tmp_path / "m.py"), "after": "raise ValueError('boom')"}
    passed, logs, code = Sandbox().run(patch)
    assert passed is False
    assert "SandboxError" in logs


@given(st.text(min_size=1))
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_sandbox_property(tmp_path, text: str) -> None:
    patch = {"target": str(tmp_path / "m.py"), "after": text}
    passed, logs, code = Sandbox().run(patch)
    assert isinstance(passed, bool)
    assert isinstance(logs, str)
    assert isinstance(code, int)
