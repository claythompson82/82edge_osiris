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


def atr(hlc_data: List[tuple[float, float, float]], window: int = 14) -> float:
    """
    Calculates the Average True Range (ATR).

    Args:
        hlc_data: A list of tuples, where each tuple contains (high, low, close)
                  for a period. Data should be in chronological order.
        window: The period for the ATR calculation.

    Returns:
        The Average True Range value. Returns float('nan') if data is insufficient.

    Raises:
        ValueError: If window is not positive or hlc_data is empty.
    """
    if window <= 0:
        raise ValueError("Window must be positive for ATR calculation.")
    if not hlc_data:
        raise ValueError("Input data list cannot be empty for ATR calculation.")
    if len(hlc_data) < window: # Need at least 'window' periods to calculate initial SMA of TR
        # More precisely, we need 'window' TR values. A single HLC tuple gives one TR value
        # if we have the *previous* close. So, we need len(hlc_data) >= window + 1
        # to form 'window' TR values for the initial ATR (which is an SMA of TRs).
        # However, a common library approach (like pandas_ta) might use Wilder's smoothing
        # from the start, which can begin with fewer values.
        # For a simple SMA of TRs for the first ATR value:
        # We need `window` TR values. The first TR value requires `hlc_data[0]` and `hlc_data[-1]` (previous close).
        # So, to get `window` TRs, we need `window + 1` HLC data points.
        # Example: hlc_data[0] (no prev_close for TR), hlc_data[1] (uses hlc_data[0].close for TR1), ...
        # Let's adjust: we need at least `window` TRs. The first TR is calculated using the first two HLC points.
        # No, standard ATR: first TR uses current H, L and *previous* C.
        # So, to get N TR values, you need N HLC tuples and 1 *prior* close.
        # If hlc_data[0] is the first period, we need a close before that, or start TR calculation from the 2nd period.
        # Let's assume hlc_data[0] is period 1, hlc_data[1] is period 2.
        # True Range for period `i`:
        #   tr1 = high[i] - low[i]
        #   tr2 = abs(high[i] - close[i-1])
        #   tr3 = abs(low[i] - close[i-1])
        #   TR[i] = max(tr1, tr2, tr3)
        # This means we need at least 2 data points in hlc_data to calculate the first TR.
        # For `window` TRs, we need `window + 1` data points in hlc_data.
        return float('nan')


    true_ranges = []
    if not hlc_data or len(hlc_data) < 2: # Need at least 2 days for the first TR
        return float('nan')

    # Calculate True Ranges
    for i in range(1, len(hlc_data)):
        high, low, _ = hlc_data[i]
        prev_close = hlc_data[i-1][2] # close of the previous period

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_range = max(tr1, tr2, tr3)
        true_ranges.append(true_range)

    if not true_ranges or len(true_ranges) < window:
        # Not enough TR values to calculate ATR over the specified window
        return float('nan')

    # Calculate ATR:
    # First ATR is the simple average of the first 'window' TRs.
    # Subsequent ATRs use Wilder's smoothing: ATR = (Previous ATR * (window - 1) + Current TR) / window

    # For simplicity and alignment with many libraries, we can use pandas ewm for Wilder's smoothing
    # Wilder's smoothing is equivalent to an EMA with alpha = 1/N.
    # Span for EMA is 2*N - 1. So for ATR, span = 2*window - 1.
    # However, the first value of ATR is typically an SMA.
    # Let's implement manually to be clear.

    if not true_ranges: # Should be caught above, but as a safeguard
        return float('nan')

    tr_series = pd.Series(true_ranges)

    # Calculate initial ATR as SMA of the first 'window' TRs
    # Then apply Wilder's smoothing for subsequent values.
    # Pandas ewm with com = window - 1 (alpha = 1/window) and adjust=True
    # is commonly used for Wilder's smoothing.
    # Or, more directly, alpha = 1/window, so span = 2*window -1 for adjust=False EMA
    # Or, alpha = 1/window, com = window -1 for adjust=True EMA (closer to Wilder's)

    # For ATR, typically the first value is an SMA, then smoothing.
    # A common way: SMA for first `window` TRs, then RMA/SMMA/Wilder's for the rest.
    # pd.Series.ewm(alpha=1/window, adjust=False).mean() gives the SMMA / Wilder's.
    # Let's use this for the whole series of TRs and take the last value.
    # This is a common simplification.
    if len(tr_series) == 0: # If true_ranges was empty
        return float('nan')

    atr_values = tr_series.ewm(alpha=1/window, adjust=False, min_periods=window).mean()

    if atr_values.empty or pd.isna(atr_values.iloc[-1]):
        return float('nan')

    return float(atr_values.iloc[-1])


def kelly_fraction(mu: float, sigma: float) -> float:
    """
    Calculates the Kelly fraction.

    Args:
        mu: The expected excess return (e.g., mean of returns over risk-free rate).
        sigma: The standard deviation of returns.

    Returns:
        The Kelly fraction (K = mu / sigma^2). Returns 0.0 if sigma is zero
        or if the resulting fraction is negative (implying no bet).
    """
    if sigma <= 0: # sigma is std dev, must be positive. sigma^2 cannot be zero unless sigma is zero.
        return 0.0  # Or raise error, but returning 0 implies no allocation.

    # Standard Kelly for simple bets: f = p - (1-p)/b where p is win prob, b is odds.
    # For continuous returns: f = mu / sigma^2
    # Assuming 'mu' is already the expected *excess* return.
    # If mu is not excess return (e.g. just mean return), and r is risk-free rate,
    # then it would be (mu - r) / sigma^2. The problem says "mu, sigma", implying mu is appropriate.

    fraction = mu / (sigma ** 2)

    # Kelly fraction should typically be between 0 and 1 for long-only, full Kelly.
    # If mu is negative, fraction is negative, meaning bet against or don't bet.
    # We can cap it at 0 if negative, as negative fraction isn't directly usable for sizing a long position.
    if fraction < 0:
        return 0.0

    # The problem mentions "Kelly fraction w/ cap 3 × ATR".
    # This function should return the raw Kelly fraction. The capping logic
    # will be applied in the position sizing step in the engine.
    # Some sources also cap Kelly fraction at 1 (i.e., no leverage from Kelly itself).
    # For now, returning the raw computed positive fraction.
    # If the task means "full Kelly" (not fractional Kelly), then this is it.
    # If it implies fractional Kelly (e.g. half-Kelly), that would be an adjustment elsewhere.
    return fraction
