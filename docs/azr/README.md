# AZR Module Documentation

This document provides an overview of the AZR modules, including the AZR Planner.

## AZR Planner Micro-Service

The AZR Planner service provides trade proposals based on market context.

### Endpoint: Propose Trade

**Path**: `/azr_api/internal/azr/planner/propose_trade`

**Method**: `POST`

**Description**: Submits a planning context to the AZR Planner and returns a trade proposal. This endpoint is intended for internal use and is available when the `OSIRIS_TEST` environment variable is set.

**Request Body**:

The request body must be a JSON object conforming to the `PlanningContext` schema:

```json
{
  "timestamp": "2024-07-31T10:00:00Z",
  "equityCurve": [100.0, 101.0, 100.5, ...], // Array of at least 30 numbers
  "volSurface": { // Optional
    "MES": 0.18,
    "M2K": 0.22
  },
  "riskFreeRate": 0.025 // Optional
}
```

**`PlanningContext` Schema Details:**
*   `timestamp` (string, required): ISO 8601 date-time string.
*   `equityCurve` (array of numbers, required): List of at least 30 equity values.
*   `volSurface` (object, optional): Key-value pairs where keys are instrument tickers (e.g., "MES", "M2K") and values are their volatilities.
*   `riskFreeRate` (number, optional): The risk-free rate.


**Example `curl` command:**

```bash
# Ensure OSIRIS_TEST environment variable was set when the server started.
# Example: export OSIRIS_TEST=1; uvicorn src.osiris.server:app --host 0.0.0.0 --port 8000

curl -X POST http://localhost:8000/azr_api/internal/azr/planner/propose_trade \\
-H "Content-Type: application/json" \\
-d '{
  "timestamp": "2024-01-15T10:30:00Z",
  "equityCurve": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130],
  "volSurface": {"MES": 0.2, "M2K": 0.25},
  "riskFreeRate": 0.015
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

planning_context = {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "equityCurve": [100.0 + i * 0.5 for i in range(35)], # Example equity curve
    "volSurface": {"MES": 0.19, "ETH_OPT": 0.95},
    "riskFreeRate": 0.022,
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

The endpoint will return a `TradeProposal` JSON object. For the current stub implementation, this will be:
```json
{
  "latentRisk": 0.0,
  "legs": []
}
```
Future implementations will populate `latentRisk` and `legs` with calculated values.

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
