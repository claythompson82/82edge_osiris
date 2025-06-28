from __future__ import annotations

from azr_planner.math_utils import clamp

def test_clamp_hits_bounds() -> None:
    """Exercises the lower-bound branch in clamp."""
    assert clamp(-999.0, 0.0, 1.0) == 0.0
    # Adding the upper bound test here as well for completeness in this file,
    # though it's also in test_math_utils.py
    assert clamp(999.0, 0.0, 1.0) == 1.0
    assert clamp(0.5, 0.0, 1.0) == 0.5
