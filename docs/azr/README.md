# AZR Module Documentation

This document provides an overview of the AZR modules, including the AZR Planner.

## AZR Planner Micro-Service

The AZR Planner service provides trade proposals based on market context.

### Endpoint: Propose Trade

**Path**: `/azr_api/internal/azr/planner/propose_trade`

**Method**: `POST`

**Description**: Submits a planning context to the AZR Planner and returns a trade proposal. This endpoint is intended for internal use and is available when the `OSIRIS_TEST` environment variable is set.

**Request Body**:

The request body must be a JSON object conforming to the `PlanningContext` schema (as of AZR-06).

**`PlanningContext` Schema Details (Key Fields):**
*   `timestamp` (string, required): ISO 8601 date-time string. e.g., `"2024-07-31T10:00:00Z"`
*   `equityCurve` (array of numbers, required): List of at least 30 equity values. e.g., `[100.0, 101.0, ...]`
*   `dailyHistoryHLC` (array of [High, Low, Close] tuples, required): List of HLC tuples, minimum length typically 15-30+ depending on calculation needs (e.g., `LR_V2_MIN_POINTS`). e.g., `[[101.0, 99.0, 100.0], ...]`
*   `volSurface` (object, required): Key-value pairs for instrument volatilities. e.g., `{"MES": 0.18, "M2K": 0.22}`
*   `riskFreeRate` (number, required): The risk-free rate. e.g., `0.025`
*   `nSuccesses` (integer, required, default: 0): Number of historical successes for confidence calibration. e.g., `10`
*   `nFailures` (integer, required, default: 0): Number of historical failures for confidence calibration. e.g., `5`
*   `dailyVolume` (array of numbers, optional): Trading volumes per period.
*   `currentPositions` (array of Leg objects, optional): Currently held positions.

**Example Request Body JSON:**
```json
{
  "timestamp": "2024-07-31T10:00:00Z",
  "equityCurve": [100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5, 105.0, 104.5, 106.0, 105.5, 107.0, 106.5, 108.0, 107.5, 109.0, 108.5, 110.0, 109.5, 111.0, 110.5, 112.0, 111.5, 113.0, 112.5, 114.0, 113.5, 115.0],
  "dailyHistoryHLC": [
    [100.5, 99.5, 100.0], [101.5, 100.0, 101.0], [101.0, 99.0, 100.5], [102.5, 100.0, 102.0],
    [102.0, 100.5, 101.5], [103.5, 101.0, 103.0], [103.0, 101.5, 102.5], [104.5, 102.0, 104.0],
    [104.0, 102.5, 103.5], [105.5, 103.0, 105.0], [105.0, 103.5, 104.5], [106.5, 104.0, 106.0],
    [106.0, 104.5, 105.5], [107.5, 105.0, 107.0], [107.0, 105.5, 106.5], [108.5, 106.0, 108.0],
    [108.0, 106.5, 107.5], [109.5, 107.0, 109.0], [109.0, 107.5, 108.5], [110.5, 108.0, 110.0],
    [110.0, 108.5, 109.5], [111.5, 109.0, 111.0], [111.0, 109.5, 110.5], [112.5, 110.0, 112.0],
    [112.0, 110.5, 111.5], [113.5, 111.0, 113.0], [113.0, 111.5, 112.5], [114.5, 112.0, 114.0],
    [114.0, 112.5, 113.5], [115.5, 113.0, 115.0]
  ],
  "volSurface": {
    "MES": 0.18
  },
  "riskFreeRate": 0.025,
  "nSuccesses": 10,
  "nFailures": 2
}
```

**Example `curl` command:**

```bash
# Ensure OSIRIS_TEST environment variable was set when the server started.
# Example: export OSIRIS_TEST=1; uvicorn src.osiris.server:app --host 0.0.0.0 --port 8000

curl -X POST http://localhost:8000/azr_api/internal/azr/planner/propose_trade \\
-H "Content-Type: application/json" \\
-d '{
  "timestamp": "2024-07-31T10:00:00Z",
  "equityCurve": [100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130],
  "dailyHistoryHLC": [[100,99,100],[101,100,101],[102,101,102],[103,102,103],[104,103,104],[105,104,105],[106,105,106],[107,106,107],[108,107,108],[109,108,109],[110,109,110],[111,110,111],[112,111,112],[113,112,113],[114,113,114],[115,114,115],[116,115,116],[117,116,117],[118,117,118],[119,118,119],[120,119,120],[121,120,121],[122,121,122],[123,122,123],[124,123,124],[125,124,125],[126,125,126],[127,126,127],[128,127,128],[129,128,129],[130,129,130]],
  "volSurface": {"MES": 0.18},
  "riskFreeRate": 0.025,
  "nSuccesses": 10,
  "nFailures": 2
}'
```

**Example using `httpx` (Python):**

```python
# Ensure OSIRIS_TEST environment variable was set when the server started.
import httpx
import datetime # For generating current timestamp

# Assuming the FastAPI server is running on localhost:8000
# and was started with OSIRIS_TEST=1
client = httpx.Client()

# Generate dummy HLC data for the example
dummy_hlc_data = [[100.0 + j*0.1 + i*0.01, 99.0 + j*0.1 + i*0.01, 99.5 + j*0.1 + i*0.01] for j in range(35)]


planning_context = {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "equityCurve": [100.0 + i * 0.5 for i in range(35)],
    "dailyHistoryHLC": dummy_hlc_data,
    "volSurface": {"MES": 0.19}, # volSurface is now required
    "riskFreeRate": 0.022,      # riskFreeRate is now required
    "nSuccesses": 15,
    "nFailures": 3
    # dailyVolume and currentPositions are optional and omitted in this example
}

try:
    response = client.post(
        "http://localhost:8000/azr_api/internal/azr/planner/propose_trade",
        json=planning_context
    )
    response.raise_for_status()
    trade_proposal = response.json()
    print("Trade Proposal Received:")
    print(trade_proposal)
except httpx.HTTPStatusError as exc:
    print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
    print(f"Response content: {exc.response.text}")
except httpx.RequestError as exc:
    print(f"An error occurred while requesting {exc.request.url!r}: {exc}")
finally:
    client.close()
```

**Success Response (`200 OK`):**

The endpoint will return a `TradeProposal` JSON object, reflecting the AZR-06 engine logic:
```json
{
  "action": "HOLD", // Example, could be ENTER or EXIT
  "rationale": "Neutral conditions: Latent Risk (0.50), Confidence (0.60). Holding positions.", // Example
  "latent_risk": 0.501, // Example value from latent_risk_v2
  "confidence": 0.600,  // Example value from bayesian_confidence
  "legs": null // Or a list of legs if action is ENTER/EXIT
  // AZR-05 specific fields like signal_value, atr_value, etc., will be null or absent
}
```
The `latent_risk`, `confidence`, `action`, `rationale`, and `legs` will be populated based on the new engine logic.

---

## Public Endpoint: Propose Trade (v1)

**Path**: `/azr_api/v1/propose_trade`

**Method**: `POST`

**Description**: Submits a `PlanningContext` to the AZR Planner and returns a `TradeProposal`. This is the primary public interface for obtaining trade proposals.

**Request Body**:
The request body must be a JSON object conforming to the `PlanningContext` schema (as detailed above for the internal endpoint, including `timestamp`, `equityCurve`, `dailyHistoryHLC`, `volSurface`, `riskFreeRate`, `nSuccesses`, `nFailures`, and optional `dailyVolume`, `currentPositions`).

**Example `curl` command:**
```bash
curl -X POST http://localhost:8000/azr_api/v1/propose_trade \\
-H "Content-Type: application/json" \\
-d '{
  "timestamp": "2024-07-31T11:00:00Z",
  "equityCurve": [100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130],
  "dailyHistoryHLC": [[100,99,100],[101,100,101],[102,101,102],[103,102,103],[104,103,104],[105,104,105],[106,105,106],[107,106,107],[108,107,108],[109,108,109],[110,109,110],[111,110,111],[112,111,112],[113,112,113],[114,113,114],[115,114,115],[116,115,116],[117,116,117],[118,117,118],[119,118,119],[120,119,120],[121,120,121],[122,121,122],[123,122,123],[124,123,124],[125,124,125],[126,125,126],[127,126,127],[128,127,128],[129,128,129],[130,129,130]],
  "volSurface": {"MES": 0.19},
  "riskFreeRate": 0.023,
  "nSuccesses": 50,
  "nFailures": 10
}'
```

**Success Response (`200 OK`):**
The endpoint will return a `TradeProposal` JSON object, identical in structure to the internal endpoint's response, reflecting the AZR-06 engine logic. Example:
```json
{
  "action": "ENTER",
  "rationale": "Favorable conditions: Latent Risk (0.15) < 0.25 and Confidence (0.85) > 0.70.",
  "latent_risk": 0.150,
  "confidence": 0.850,
  "legs": [{"instrument": "MES", "direction": "LONG", "size": 1.0, "limit_price": null}]
}
```

---

## AZR Planner Backtesting

The AZR Planner includes a backtesting harness to evaluate strategy performance on historical data.

### How to run a local back-test

The backtester can be invoked via the Osiris CLI:

```bash
python -m osiris.scripts.cli_main planner backtest --dataset sp500_sample --out backtest_report.json
```

**Arguments:**
*   `--dataset`: Specifies the dataset to use. Currently, `sp500_sample` is supported, which uses data from `src/azr_planner/datasets/sp500_sample.csv`.
*   `--out`: The file path where the JSON backtest report will be saved. Defaults to `backtest_report.json`.

The output report (`backtest_report.json`) will contain detailed daily results, overall performance metrics, and the equity curve.

### Backtest Metrics Definitions

The backtest report includes the following key performance metrics:

*   **CAGR (Compound Annual Growth Rate)**: The year-over-year growth rate of an investment over a specified period. Calculated as `(End Value / Start Value)^(1 / Num Years) - 1`.
*   **Maximum Drawdown (MDD)**: The largest peak-to-trough decline during a specific period, expressed as a percentage of the peak. Indicates downside risk.
*   **Sharpe Ratio**: Measures the risk-adjusted return. Calculated as `(Mean Excess Return) / (Std Dev of Excess Return)`, annualized by multiplying by `sqrt(Num Trading Days Per Year)`. Excess return is typically over the risk-free rate.
*   **Sortino Ratio**: Similar to Sharpe Ratio, but only penalizes for downside volatility. Calculated as `(Mean Excess Return) / (Downside Deviation)`, annualized. Downside deviation is calculated using returns below a Minimum Acceptable Return (MAR), often the risk-free rate.
*   **Win Rate**: The percentage of total trades that were profitable. `(Number of Winning Trades / Total Number of Trades)`.
*   **Total Trades**: The total number of executed trades (entries or exits that resulted in a logged P&L).
*   **Winning Trades**: Number of trades with P&L > 0.
*   **Losing Trades**: Number of trades with P&L < 0.
*   **Average Win P&L**: The average profit from winning trades.
*   **Average Loss P&L**: The average loss from losing trades (will be a negative value).
*   **Average Trade P&L**: The average profit or loss across all trades.
*   **Profit Factor**: Gross Profit (sum of all winning P&Ls) divided by Gross Loss (absolute sum of all losing P&Ls). A value greater than 1 indicates profitability. Can be infinite if there are no losses.

---

## 💻 Local / Codestral Dev Loop

This section outlines the steps to set up a minimal environment for developing and testing the AZR Planner components within the Codestral sandbox or a similar local, resource-constrained environment.

**Objective**: Achieve a green run for `pytest` and `mypy` specifically for the `azr_planner` modules, without pulling in heavy dependencies like `torch` or `whisper`.

**Sandbox Setup and Testing Steps:**

1.  **Editable Install of the Project**:
    This step is crucial for ensuring that Python can find your local `src/azr_planner` modules. It symlinks the project into your Python environment's `site-packages`.
    ```bash
    python -m pip install -e .
    ```

2.  **Install Minimal Test Dependencies**:
    A specific requirements file, `requirements-azr-tests.txt`, contains only the lightweight dependencies needed for AZR planner tests.
    ```bash
    pip install -r requirements-azr-tests.txt
    ```
    The contents of `requirements-azr-tests.txt` should be:
    ```txt
    fastapi>=0.110
    pydantic>=2.7
    pytest
    pytest-mock
    hypothesis
    # Add other minimal dependencies like mypy if you intend to run it here
    # mypy
    # pytest-cov
    # pytest-asyncio
    # httpx
    # uvicorn
    # anyio
    ```
    *(Note: The user provided a very minimal list. The commented lines represent packages that were in the previous iteration of `requirements-azr-tests.txt` and might be needed for full `pytest` and `mypy` runs as per original broader plan. Adjust as necessary for the exact scope of "AZR tests pass".)*

---

## Walk-forward Methodology

Walk-forward analysis is a method of testing a trading strategy that aims to simulate how the strategy would have performed in real-time, thereby reducing the risk of overfitting to historical data.

*   **Concept**: Instead of optimizing a strategy on an entire historical dataset at once, walk-forward analysis divides the data into several contiguous "windows".
*   **Process**:
    *   The strategy is optimized (or its parameters are determined) on an initial segment of historical data (the "in-sample" or "training" window).
    *   It is then tested on the immediately following segment of data (the "out-of-sample" or "testing" window) using the parameters derived from the training window.
    *   This process is repeated by rolling both windows forward in time. For example, the first training window might be days 1-60, tested on days 61-90. The next could be days 31-90 (training) tested on days 91-120 (if step is 30 days).
*   **Windowing in AZR-09**: The `run_walk_forward` function implements a rolling window approach. Each "window" (e.g., a 30-day slice as per the task) is effectively a period over which a backtest (`run_backtest`) is performed. The strategy logic within `generate_plan` uses its own lookback (e.g., 30 days of equity curve for `latent_risk_v2`) from the data *within* that window.
*   **Window Size & Step**:
    *   `window_days`: Defines the length of each historical data slice passed to an individual backtest run. This is the period the strategy is simulated over in one iteration of the walk-forward process.
    *   `step_days`: Determines how many days the window is moved forward for the next iteration. A smaller step means more overlapping windows and more tests.
*   **Purpose**: The primary goal is to assess the strategy's robustness and stability over different market conditions and to see if parameters optimized in one period hold up in a subsequent, unseen period. This helps guard against curve-fitting.
*   **Aggregated Metrics**: Performance metrics (like Sharpe ratio, max drawdown) are calculated for each out-of-sample window. These are then aggregated (e.g., `mean_sharpe`, `worst_drawdown` over all windows) to give a more realistic expectation of future performance. The `total_return` is calculated over the entire period using a single continuous backtest for a holistic view.
*   **Benefits**: Provides a more conservative and potentially more reliable estimate of a strategy's future performance compared to a single backtest on all data. Highlights parameter stability.
*   **Limitations**: Assumes strategy parameters are recalibrated/re-evaluated at the end of each training window. The choice of window lengths and step size can influence results. Past performance, even with walk-forward, is not a guarantee of future results.

---

## Generating Walk-Forward Reports via CLI

The AZR Planner provides a command-line interface (CLI) to run a walk-forward backtest on custom equity data and generate an HTML performance report.

**CLI Usage:**

```bash
python -m osiris.scripts.backtest_cli \\
       --equity-curve path/to/your/data.csv \\
       --out path/to/your/report.html
```

*   `--equity-curve`: Path to a CSV file containing historical price data. The CSV must have 'timestamp' and 'price' columns.
*   `--out`: Filepath where the generated HTML report will be saved.
*   `--window-days` (optional): Number of days in each rolling window slice for the walk-forward analysis. Defaults to a suitable value (e.g., 31).

**Report Contents:**

The HTML report includes:
*   A table of aggregated performance metrics (Mean Sharpe, Worst Drawdown, Total Return, etc.).
*   An equity curve line chart comparing the strategy's performance against a buy-and-hold benchmark.
*   A histogram showing the distribution of individual trade Profit & Loss.

**Example Screenshot (placeholder):**

![Sample Walk-Forward Report](img/sample_report.png)

*(Note: The actual image `docs/azr/img/sample_report.png` is not generated by this tool; a placeholder link is included as per requirements.)*

---

## Daily P&L Workflow (AZR-14)

The AZR system includes a daily Profit & Loss (P&L) simulation and reporting workflow. At the end of each UTC trading day (rollover at 21:00 UTC), this process ingests all trade fills for that day, updates the portfolio's positions, and calculates key P&L metrics. These metrics include realized P&L from closed trades, unrealized P&L on open positions based on EOD market prices, net position values, cash balance, total equity, gross and net exposures, and tracking of the portfolio's equity curve and drawdown. The resulting `DailyPNLReport` is persisted to a database (LanceDB) and a Prometheus counter (`azr_pnl_reports_total`) is incremented. A read-only API endpoint (`GET /azr_api/v1/pnl/daily?last_n=<int>`) allows fetching the latest P&L reports.

**Sample `DailyPNLReport` JSON Payload:**
```json
{
  "date": "2024-03-15",
  "realized_pnl": 150.75,
  "unrealized_pnl": -35.20,
  "net_position_value": 12340.50,
  "cash": 87659.50,
  "total_equity": 99964.80,
  "gross_exposure": 12340.50,
  "net_exposure": 12340.50,
  "cumulative_max_equity": 100500.00,
  "current_drawdown": 0.005325373134328358, // (100500 - 99964.80) / 100500
  "equity_curve_points": [
    99800.00,
    100100.25,
    100500.00,
    99964.80
  ]
}
```

---

## 💻 Local / Codestral Dev Loop

This section outlines the steps to set up a minimal environment for developing and testing the AZR Planner components within the Codestral sandbox or a similar local, resource-constrained environment.

**Objective**: Achieve a green run for `pytest` and `mypy` specifically for the `azr_planner` modules, without pulling in heavy dependencies like `torch` or `whisper`.

**Sandbox Setup and Testing Steps:**

1.  **Editable Install of the Project**:
    This step is crucial for ensuring that Python can find your local `src/azr_planner` modules. It symlinks the project into your Python environment's `site-packages`.
    ```bash
    python -m pip install -e .
    ```

2.  **Install Minimal Test Dependencies**:
    A specific requirements file, `requirements-azr-tests.txt`, contains only the lightweight dependencies needed for AZR planner tests.
    ```bash
    pip install -r requirements-azr-tests.txt
    ```
    The contents of `requirements-azr-tests.txt` should be:
    ```txt
    fastapi>=0.110
    pydantic>=2.7
    pytest
    pytest-mock
    hypothesis
    # Add other minimal dependencies like mypy if you intend to run it here
    # mypy
    # pytest-cov
    # pytest-asyncio
    # httpx
    # uvicorn
    # anyio
    ```
    *(Note: The user provided a very minimal list. The commented lines represent packages that were in the previous iteration of `requirements-azr-tests.txt` and might be needed for full `pytest` and `mypy` runs as per original broader plan. Adjust as necessary for the exact scope of "AZR tests pass".)*


3.  **Set Environment Variables**:
    The AZR Planner endpoint is conditionally mounted based on `OSIRIS_TEST`.
    ```bash
    export OSIRIS_TEST=1
    ```
    *(The `sitecustomize.py` script at the repository root now unconditionally attempts to add `src/` to `sys.path` to aid module discovery in environments like the sandbox.)*

4.  **Run Scoped Pytest Checks**:
    Execute tests only for the `azr_planner` directory and specific AZR-related server tests.
    ```bash
    pytest tests/azr_planner -q
    pytest tests/test_server.py::test_azr_planner_smoke_endpoint_exists -q
    # Add other specific AZR server tests if needed:
    # pytest tests/test_server.py::test_azr_planner_invalid_input_equity_curve_too_short -q
    # pytest tests/test_server.py::test_azr_planner_invalid_input_missing_required_field -q
    # pytest tests/test_server.py::test_azr_planner_prefix_and_tags_available_when_osiris_test_set -q
    ```
    These should all pass.

5.  **Run Scoped MyPy Checks**:
    Use `mypy.ini` to configure `mypy` to only check the `azr_planner` files and ensure `namespace_packages = True` is set.
    The `mypy.ini` should contain:
    ```ini
    [mypy]
    strict = True
    namespace_packages = True
    files = src/azr_planner,tests/azr_planner
    ```
    Then run:
    ```bash
    mypy --strict
    ```
    This should report 0 errors for the scoped files.

**Troubleshooting `ModuleNotFoundError` in Sandbox:**

If `pytest` or `mypy` still report `ModuleNotFoundError` after these steps:
 *   **Verify `sitecustomize.py` execution**: The `sitecustomize.py` script at the repository root unconditionally attempts to add `src/` to `sys.path`. Ensure it's actually running for `pytest`/`mypy` by adding temporary print statements to its beginning and checking if they appear in the tool output.
*   **Check `sys.path` directly**: From within a `pytest` test (e.g., using `import sys; print(sys.path)`) or a `mypy` run (less straightforward), try to inspect the actual `sys.path` being used by the tool.
*   **Python invocation**: Ensure `python`, `pip`, `pytest`, `mypy` are all being invoked from the same virtual environment or Python installation where packages are expected to be.

These steps aim to provide a reproducible green run for the AZR Planner's core components in a resource-limited environment. Full repository-wide checks with all dependencies are typically deferred to a more robust CI environment.
