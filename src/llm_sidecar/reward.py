"""Tiny reward shim for DGM kernel tests."""

def proofable_reward(*_a, **_kw) -> float:
    """
    Always returns 1.0 — enough for tests that rely on the symbol.
    """
    return 1.0
