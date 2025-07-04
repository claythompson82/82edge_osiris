from __future__ import annotations

import pytest
from pydantic import ValidationError

import dgm_kernel  # noqa:F401 - triggers global config
from dgm_kernel.trace_schema import Trace


def test_trace_extra_field_raises():
    with pytest.raises(ValidationError):
        Trace(id="t1", timestamp=0, pnl=0.0, extra_field="x")

