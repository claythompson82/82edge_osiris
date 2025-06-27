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


def latent_risk(equity_curve: List[float], vol_surface: Dict[str, float], risk_free_rate: float) -> float:
    """
    Calculates the latent risk score based on the equity curve, volatility surface,
    and risk-free rate.
    This is a placeholder implementation. The full formula is per §3.2–3.4 of the PDF.
    The actual implementation will require more detailed components (drawdowns, time under water, etc.).

    Args:
        equity_curve: List of equity values.
        vol_surface: Dictionary of instrument volatilities.
        risk_free_rate: The risk-free rate.

    Returns:
        A risk score clamped between 0 and 1.
    """
    if not equity_curve or len(equity_curve) < 30: # Min length from PlanningContext
        # Or handle as per full spec for latent risk with insufficient data
        return 1.0 # Max risk if data is insufficient

    # Placeholder logic:
    # 1. Calculate some metric from equity_curve (e.g., Sharpe-like ratio or drawdown measure)
    # 2. Incorporate vol_surface (e.g., average vol, or vol of a reference instrument)
    # 3. Use risk_free_rate in calculations.

    # Simplified example:
    # Use rolling volatility of equity curve percentage changes as a risk indicator.
    # This is NOT the full formula from the PDF.
    if len(equity_curve) < 2:
        simple_vol_metric = 1.0 # Max risk
    else:
        equity_returns = [
            (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            if equity_curve[i-1] != 0
            else 0.0
            for i in range(1, len(equity_curve))
        ]
        # Corrected indentation for this block:
        if not equity_returns or len(equity_returns) < 2:
            simple_vol_metric = 1.0
        else:
            std_dev_returns = calculate_std_dev(equity_returns)
            simple_vol_metric = min(1.0, max(0.0, (std_dev_returns / 0.05) * 0.8 )) if std_dev_returns > 0 else 0.0

    # Incorporate vol_surface (e.g., average of provided surface vols)
    avg_surface_vol = 0.0
    if vol_surface:
        avg_surface_vol = sum(vol_surface.values()) / len(vol_surface)
    # Crude combination:
    # Higher avg_surface_vol means higher risk. Normalize it (e.g. if avg 0.5 is high risk)
    surface_vol_metric = min(1.0, max(0.0, (avg_surface_vol / 0.5) * 0.5))

    # Combine metrics (very naively)
    # This is a placeholder. The actual formula from PDF §3.2-3.4 is needed.
    # For example, one might use Sharpe, Sortino, max drawdown, time under water, etc.
    # and combine them, possibly with weights.

    # Placeholder risk score: average of the two crude metrics
    # Adjust based on risk_free_rate: higher rfr might imply lower risk tolerance for some strats
    # For simplicity, let's say if rfr > 0.05, we are more risk averse (higher score).
    risk_adjustment = 0.0
    if risk_free_rate > 0.05:
        risk_adjustment = 0.1

    combined_risk = (simple_vol_metric + surface_vol_metric) / 2.0 + risk_adjustment

    # Clamp result between 0 and 1
    final_risk_score = min(1.0, max(0.0, combined_risk))

    return final_risk_score
