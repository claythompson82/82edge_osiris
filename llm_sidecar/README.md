# llm_sidecar

The `llm_sidecar` package contains the code that powers the LLM sidecar service used in Osiris. It provides model loading utilities, a small event bus, database helpers and other support modules that are imported by `osiris.server`.

## Key features

- **Model loader** (`loader.py`) – pulls Hermes and Phi‑3 models into memory and exposes helper getters.
- **Event bus** (`event_bus.py`) – asynchronous Redis based pub/sub for components.
- **LanceDB integration** (`db/`) – stores feedback logs and orchestrator run data.
- **Hermes scoring plugin** (`hermes_plugin.py`) – evaluates trade proposals.
- **Text‑to‑speech** (`tts.py`) – wraps Chatterbox and publishes audio on Redis.
- **Reward stub** (`reward.py`) – placeholder proofable reward function.

## Running locally

The sidecar API lives in `osiris/server.py`. When developing outside Docker you can run it directly with Uvicorn:

```bash
export OSIRIS_SIDECAR_URL=http://localhost:8000
# set paths to your models if they differ from the defaults
export MICRO_LLM_MODEL_PATH=/path/to/phi3.onnx
uvicorn osiris.server:app --reload
```

This starts the FastAPI application with hot reload enabled on <http://localhost:8000>.

## Configuration

Common environment variables consumed by the sidecar:

| Variable | Description |
| --- | --- |
| `MICRO_LLM_MODEL_PATH` | Path to the Phi‑3 ONNX model file. |
| `ENABLE_METRICS` | Enable Prometheus metrics collection (`true`/`false`). |
| `ENABLE_PROFILING` | Expose `/debug/prof` with request profiling information. |
| `OSIRIS_SIDECAR_URL` | Base URL used by tests and other services. |
| `SENTRY_DSN` / `SENTRY_ENV` / `SENTRY_TRACES_SAMPLE_RATE` | Optional Sentry telemetry. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Endpoint for OpenTelemetry traces. |
| `OTEL_SERVICE_NAME` | Service name for OTEL (defaults to `osiris_llm_sidecar`). |

Redis connection defaults to `redis://localhost:6379/0` but can be changed by instantiating `EventBus` with a different URL in the server.

## Architecture notes

- `loader.py` lazily loads models and tokenizers and exposes them via getter functions.
- `db/` initializes a LanceDB database under `/app/lancedb_data` and provides a small CLI (`python -m llm_sidecar.db query-runs`).
- `event_bus.py` offers simple publish/subscribe helpers used by the orchestrator and TTS components.
- `hermes_plugin.py` scores trade ideas with the Hermes model.
- `tts.py` streams WAV bytes over Redis for real‑time audio feedback.

## Running tests

Install test dependencies with `pip install -r requirements-tests.txt` and run:

```bash
pytest tests/test_event_bus.py tests/test_db.py tests/test_db_bootstrap.py
```

Tests use [fakeredis](https://github.com/cunla/fakeredis-py) for the event bus and a temporary directory for LanceDB.
