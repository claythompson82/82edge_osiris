from hypothesis import given, strategies as st, settings, HealthCheck
from dgm_kernel.sandbox import Sandbox

@given(
    host=st.text(min_size=1, max_size=5, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
    port=st.integers(min_value=1, max_value=65535),
)
@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_sandbox_egress_blocked(tmp_path, host: str, port: int) -> None:
    code = (
        "import asyncio\n"
        "async def r():\n"
        f"    await asyncio.open_connection('{host}', {port})\n"
        "asyncio.run(r())"
    )
    patch = {"target": str(tmp_path / "m.py"), "after": code}
    passed, logs, _ = Sandbox().run(patch)
    assert passed is False
    assert "egress blocked" in logs.lower()
