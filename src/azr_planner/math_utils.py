"""Pure-math utility functions for AZR Planner."""

import math
from typing import List, Dict # Added Dict
import pandas as pd # For EMA and rolling calculations

# Existing helper functions (calculate_mean, calculate_std_dev) can be kept if needed,
# or removed if pandas equivalents are preferred for new functions.
# For now, keeping them as they might be used by tests or other parts not yet seen.

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
    Calculates the sample standard deviation of a list of numbers.
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
    variance = sum([(x - mean) ** 2 for x in data]) / (n - 1)
    return math.sqrt(variance)


def ema(series: List[float], span: int) -> float:
    """
    Calculates the Exponential Moving Average (EMA) for the given series.
    Returns the latest EMA value.
    Args:
        series: List of float values.
        span: The span for the EMA calculation.
    Returns:
        The latest EMA value in the series. Returns NaN if series is too short.
    Raises:
        ValueError: if span is not positive or series is empty.
    """
    if not series:
        raise ValueError("Series cannot be empty for EMA calculation.")
    if span <= 0:
        raise ValueError("Span must be positive for EMA calculation.")
    if len(series) == 0: # Should be caught by "not series" but good to be explicit
        return float('nan') # Or raise error, current pandas behavior is NaN for empty result

    s = pd.Series(series)
    ema_series = s.ewm(span=span, adjust=False).mean() # adjust=False is common for financial EMAs
    if ema_series.empty or pd.isna(ema_series.iloc[-1]): # Check for NaN before returning
        return float('nan')
    return float(ema_series.iloc[-1]) # Cast to float


def rolling_volatility(series: List[float], window: int) -> float:
    """
    Calculates the rolling standard deviation (volatility) for the given series.
    Returns the latest volatility value. Assumes daily data and annualizes.
    Args:
        series: List of float values (e.g., daily price changes or returns).
        window: The rolling window size.
    Returns:
        The latest annualized rolling volatility. Returns NaN if series is too short.
    Raises:
        ValueError: if window is not positive or series is too short for a full window.
    """
    if window <= 0:
        raise ValueError("Window must be positive for rolling volatility calculation.")
    if not series:
        raise ValueError("Series cannot be empty for rolling volatility.")
    if window <= 0: # This check was already there, good.
        raise ValueError("Window must be positive for rolling volatility calculation.")
    if len(series) < window:
        return float('nan')

    s = pd.Series(series)
    # Calculate rolling standard deviation of *returns*. If series is prices, first calculate returns.
    # Assuming `series` are log returns for simplicity here. If they are prices, this needs change.
    # For example, if prices: s_returns = np.log(s / s.shift(1))
    # For now, assume `series` are already suitable for direct std dev calculation (e.g. returns)
    rolling_std = s.rolling(window=window).std()
    if rolling_std.empty or pd.isna(rolling_std.iloc[-1]):
        return float('nan')

    # Annualize: assuming daily data, sqrt(252) is common for trading days.
    # This part of the spec might be detailed in the PDF. Using 252 as a placeholder.
    annualized_vol = float(rolling_std.iloc[-1]) * math.sqrt(252) # Cast to float
    return annualized_vol


import numpy as np
from scipy.stats import entropy as shannon_entropy

def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamps a value between a minimum and maximum."""
    return max(min_val, min(value, max_val))

def _calculate_shannon_entropy(data: List[float], base: float = math.e) -> float:
    """
    Calculates Shannon entropy for a list of data.
    Helper in case scipy.stats.entropy is not available or for specific base.
    """
    if not data:
        return 0.0

    counts = pd.Series(data).value_counts()
    probabilities = counts / len(data)

    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]

    if probabilities.empty:
        return 0.0

    entropy_val = -np.sum(probabilities * np.log(probabilities) / np.log(base))
    return float(entropy_val)


def latent_risk(equity_curve: List[float]) -> float:
    """
    Calculates the latent risk score based on the equity curve.
    Formula from AZR Planner Design PDF §3.2.

    Args:
        equity_curve: List of equity values. The formula implies at least 30 data points
                      for some calculations (e.g., rolling vol, drawdown).
                      The function will handle shorter series by returning max risk or NaN equivalent.

    Returns:
        A risk score clamped between 0 and 1.
    """
    num_points = len(equity_curve)

    # Ensure we only operate on the last 30 bars for relevant calculations as per PDF
    # or the full curve if shorter than 30.
    # The PDF implies calculations like dd and vol are over the last 30 bars.
    # For consistency, we'll use the tail for all components if longer than 30.
    # If shorter, some metrics might not be meaningful or directly calculable as per spec.

    # The problem states "rolling 30-period tail of series" for latent_risk call.
    # This implies the input `equity_curve` to this function might already be the tail.
    # However, for robustness, let's ensure calculations are on at most 30 points
    # if the input is longer, or handle shorter series gracefully.

    if num_points == 0:
        return 1.0 # Max risk for empty curve

    # Use the full equity_curve as provided to this function.
    # The caller (engine) should provide the relevant segment (e.g., last 30 points).

    # σ_a: rolling_volatility(equity_curve, window=30)
    # The existing rolling_volatility annualizes. window=30 is specified.
    # If num_points < 30, rolling_volatility will return NaN.
    sigma_a = rolling_volatility(equity_curve, window=30)
    sigma_a = np.nan_to_num(sigma_a, nan=1.0) # Treat NaN vol as high risk component

    sigma_tgt = 0.25

    # dd: max(1 – equity_curve[i] / equity_curve[:i+1].max()) over last 30 bars
    # If equity_curve has < 30 bars, this calculation is over available bars.
    max_dd = 0.0
    if num_points > 0:
        historical_max = pd.Series(equity_curve).expanding(min_periods=1).max()
        drawdowns = 1 - (pd.Series(equity_curve) / historical_max)
        # Ensure we only consider the last 30 data points for this specific dd calculation,
        # or fewer if the series is shorter than 30.
        relevant_drawdowns = drawdowns.iloc[-min(num_points, 30):]
        if not relevant_drawdowns.empty:
            max_dd = relevant_drawdowns.max()
        max_dd = np.nan_to_num(float(max_dd), nan=1.0) # Treat NaN DD as high risk

    # H: Shannon entropy of 5-day equity returns (base-e)
    # This requires at least 6 data points in equity_curve to get one 5-day return.
    # To get a series of 5-day returns, we need more.
    # Let's assume if we can't calculate it, H is maximal (contributing to higher risk).
    H = 0.0 # Default to 0 if not calculable, to avoid undue penalty if data too short
    if num_points >= 6: # Need at least 1 5-day return (6 points for ec[5]/ec[0] - 1)
        # Calculate 5-day returns: (price_t / price_{t-5}) - 1
        # Using pct_change(periods=5) is equivalent to (val / val.shift(5)) - 1
        returns_5day = pd.Series(equity_curve).pct_change(periods=5).dropna()
        if not returns_5day.empty:
            # Discretize returns for entropy calculation if they are continuous.
            # The PDF doesn't specify bins. Using a reasonable number of bins.
            # Or, if the returns are already somewhat discrete, can use them directly.
            # Scipy's entropy handles continuous data by discretizing or via method for continuous vars.
            # Here, we use value_counts which implies discretization.
            # Let's try with direct values first, then consider binning if results are off.
            # Using scipy.stats.entropy which can take pk (probabilities) or vk (values).
            # If vk, it calculates counts and then probabilities.

            # The problem statement implies equity_curve is a list of floats (prices/values).
            # returns_5day will be a series of floats.
            # scipy.stats.entropy(pk=probabilities_of_returns)

            # To get pk, we need to count occurrences of unique return values or bin them.
            # Let's use the helper _calculate_shannon_entropy which does value_counts.
            # Limit to the last 30 5-day returns if available
            relevant_returns_5day = returns_5day.iloc[-min(len(returns_5day), 30):].tolist()
            if relevant_returns_5day:
                H = _calculate_shannon_entropy(relevant_returns_5day, base=math.e)
                # Alternative using scipy directly if data is prepared as counts/probabilities
                # counts = pd.Series(relevant_returns_5day).value_counts()
                # H = shannon_entropy(counts, base=math.e)

    H_max = 3.0 # Pre-computed constant

    # raw = 0.5*(σ_a/σ_tgt) + 0.3*dd + 0.2*(H / H_max)
    # Handle potential division by zero for H_max or sigma_tgt if they could be zero.
    # sigma_tgt is constant 0.25. H_max is constant 3.0. So no division by zero there.

    term_vol = 0.5 * (sigma_a / sigma_tgt) if sigma_tgt > 0 else 0.5 # Max penalty if sigma_tgt is 0
    term_dd = 0.3 * max_dd
    term_entropy = 0.2 * (H / H_max) if H_max > 0 else 0.2 # Max penalty if H_max is 0

    raw_risk = term_vol + term_dd + term_entropy

    final_risk_score = _clamp(raw_risk, 0.0, 1.0)

    return final_risk_score
