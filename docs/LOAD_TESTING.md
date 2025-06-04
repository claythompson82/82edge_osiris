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
From the repository root, execute:

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

## Stopping services
Press `Ctrl+C` to stop k6 when done. Shut down the stack with:

```bash
docker compose down
```
