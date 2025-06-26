"""Pure-math utility functions for AZR Planner."""

import math
from typing import List


def example_math_function(x: float, y: float) -> float:
    """
    This is an example function.
    It computes the sum of squares of x and y.
    """
    return x**2 + y**2

def calculate_mean(data: List[float]) -> float:
    """
    Calculates the mean of a list of numbers.

    Args:
        data: A list of floating-point numbers.

    Returns:
        The mean of the numbers in the list.

    Raises:
        ValueError: If the input list is empty.
    """
    if not data:
        raise ValueError("Input list cannot be empty when calculating mean.")
    return sum(data) / len(data)

def calculate_std_dev(data: List[float]) -> float:
    """
    Calculates the standard deviation of a list of numbers.

    Args:
        data: A list of floating-point numbers.

    Returns:
        The standard deviation of the numbers in the list.

    Raises:
        ValueError: If the input list has fewer than two elements.
    """
    n = len(data)
    if n < 2:
        raise ValueError("Standard deviation requires at least two data points.")
    mean = calculate_mean(data)
    variance = sum([(x - mean) ** 2 for x in data]) / (n -1) # Sample standard deviation
    return math.sqrt(variance)
