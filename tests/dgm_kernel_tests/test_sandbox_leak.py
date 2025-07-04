import sys
import psutil
import pytest


def test_sandbox_leak():
    if sys.platform.startswith("win"):
        pytest.xfail("resource module missing")
    from dgm_kernel.sandbox import Sandbox

    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    patch = {"after": "print('ok')"}
    for _ in range(50):
        Sandbox().run(patch)
    rss_after = proc.memory_info().rss
    assert rss_after - rss_before < 5 * 1024 * 1024

