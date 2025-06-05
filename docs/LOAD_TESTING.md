# Load Testing Osiris with k6

This guide explains how to run a basic load test against the `llm_sidecar` service using [k6](https://k6.io/).

## Prerequisites
- Docker and Docker Compose installed
- The Osiris repository cloned locally
- k6 installed locally (`brew install k6` on macOS, `choco install k6` on Windows) or run via Docker

## Starting the stack
Launch the sidecar and its dependencies:

```bash
cd docker
docker compose up redis llm-sidecar
```

Wait until the API is available at `http://localhost:8000`.

## Running the load test
The default script `k6_sidecar_load.js` exercises the `/generate` and `/metrics` endpoints. From the repository root, execute:

```bash
k6 run scripts/load/k6_sidecar_load.js
```

If k6 is not installed, you can run it via Docker:

```bash
docker run --rm -i --network host -v $(pwd)/scripts/load/k6_sidecar_load.js:/script.js grafana/k6 run /script.js
```

The script sends requests to `/generate` with a sample prompt and checks the `/metrics` endpoint. It reports request rates, P95/P99 latencies and error rates in the output summary.

Set `OSIRIS_URL` to target a different base URL:

```bash
OSIRIS_URL=http://localhost:8000 k6 run scripts/load/k6_sidecar_load.js
```

### Additional scenarios

Two more scripts cover read‑heavy and write‑heavy patterns:

* `k6_read_heavy.js` performs repeated `GET /health` and `GET /metrics` requests.
* `k6_propose_trade.js` posts prompts from `prompts.json` to `/propose_trade_adjustments/`.

Run them in the same way, for example:

```bash
k6 run scripts/load/k6_read_heavy.js
k6 run scripts/load/k6_propose_trade.js
```

The `prompts.json` file can be edited to supply custom test inputs.

## Stopping services
Press `Ctrl+C` to stop k6 when done. Shut down the stack with:

```bash
docker compose down
```

## Service Level Objectives

The k6 scripts define Service Level Objectives (SLOs) for the most
important endpoints. Each script sets [k6 thresholds](https://k6.io/docs/using-k6/thresholds/)
so the summary output clearly shows whether the targets are met.

| Endpoint | Success rate | Latency |
| -------- | ------------ | ------- |
| `POST /generate` | 99.9% | p95 < 800&nbsp;ms, p99 < 1200&nbsp;ms |
| `GET /health` and `GET /metrics` | 99.9% | p95 < 400&nbsp;ms |
| `POST /propose_trade_adjustments/` | 99.5% | p95 < 1500&nbsp;ms |

When a test run finishes, k6 prints a pass/fail summary for each of
these thresholds so you can quickly see if the SLOs were honored.
