"""Tests for AZR Planner math_utils."""

import pytest
import math
from azr_planner.math_utils import (
    example_math_function,
    calculate_mean,
    calculate_std_dev,
)

def test_example_math_function() -> None:
    """Test the example_math_function."""
    assert example_math_function(2.0, 3.0) == 13.0
    assert example_math_function(0.0, 0.0) == 0.0
    assert example_math_function(-1.0, 1.0) == 2.0
    assert example_math_function(1.5, 2.5) == 8.5

def test_calculate_mean() -> None:
    """Test calculate_mean function."""
    assert calculate_mean([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0
    assert calculate_mean([10.0]) == 10.0
    assert calculate_mean([-1.0, 1.0]) == 0.0
    assert calculate_mean([1.5, 2.5, 3.5]) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="Input list cannot be empty"):
        calculate_mean([])

def test_calculate_std_dev() -> None:
    """Test calculate_std_dev function (sample standard deviation)."""
    assert calculate_std_dev([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(math.sqrt(2.5))
    assert calculate_std_dev([1.0, 3.0]) == pytest.approx(math.sqrt(2.0))
    assert calculate_std_dev([5.0, 5.0, 5.0, 5.0]) == 0.0
    assert calculate_std_dev([-1.0, -2.0, -3.0]) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="Standard deviation requires at least two data points."):
        calculate_std_dev([1.0])
    with pytest.raises(ValueError, match="Standard deviation requires at least two data points."):
        calculate_std_dev([])

# Docstring checks (visual reminder, not programmatic)
assert calculate_mean.__doc__ is not None
assert calculate_std_dev.__doc__ is not None
assert example_math_function.__doc__ is not None
