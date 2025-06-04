# Osiris Policy Orchestrator

The `osiris_policy.orchestrator` module implements the event‑driven policy loop for Osiris. It listens for market ticks published to Redis, builds a LangGraph workflow around those ticks and interacts with the `llm_sidecar` to generate and validate trade ideas. Execution details are logged to LanceDB so that decisions can later be audited or used for model fine‑tuning.

## Overview

1. **Tick ingestion** – `market_tick_listener` subscribes to the configured Redis channel. Ticks are buffered until the `--ticks_per_proposal` threshold is reached.
2. **Graph execution** – Once triggered the workflow goes through the nodes defined in `build_graph()`:
   - `query_market` – basic preprocessing of the tick buffer.
   - `generate_proposal` – calls `/generate?model_id=phi3` on the sidecar to obtain a JSON proposal.
   - `risk_management` – validates the proposal with `advisor.risk_gate` and stores the advice in LanceDB.
   - `evaluate_proposal` – optional Hermes assessment through `/propose_trade_adjustments`.
   - `publish_events` – publishes results and optional TTS events to Redis and composes the final output.
3. **Run logging** – `process_workflow_run` writes the final state to the `orchestrator_runs.lance` table via `log_run`.

The orchestrator can run continuously as part of the full simulator setup or standalone from the command line.

## Dependencies

- **Redis** – message bus for ticks and event publishing. The `EventBus` helper handles connections and Pub/Sub logic.
- **LanceDB** – stores orchestrator runs and risk advice. The default path is `./lancedb_data`.
- **llm_sidecar** – HTTP service providing generation and evaluation endpoints used by the nodes.
- **LangGraph** – orchestrates the workflow graph.

OpenTelemetry tracing is enabled when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.

## Configuration

The CLI entry point `run_orchestrator()` accepts:

```bash
python -m osiris_policy.orchestrator \
  --redis_url redis://localhost:6379/0 \
  --market_channel market.ticks \
  --ticks_per_proposal 10
```

- `--redis_url` – Redis instance containing the tick stream and used by `EventBus`.
- `--market_channel` – channel name where market ticks are published.
- `--ticks_per_proposal` – number of ticks to buffer before starting a new workflow.

Risk limits are defined in `RISK_GATE_CONFIG` inside the module and can be adjusted as needed.

## Interactions

- **Simulator → Redis** – publishes `market.ticks` which the orchestrator consumes.
- **Orchestrator → llm_sidecar** – requests proposal generation and evaluation via HTTP.
- **Orchestrator ↔ Redis** – publishes `advice.generated`, `phi3.proposal.created`, `phi3.proposal.assessed` and optional TTS acknowledgement events.
- **Orchestrator → LanceDB** – logs each run and risk gate decision.

The main LangGraph flow is linear with a conditional branch after `risk_management` deciding whether to evaluate the proposal further or publish results immediately.

## Development and Testing

To iterate locally run the orchestrator inside the repo environment:

```bash
make dev-shell
python -m osiris_policy.orchestrator --redis_url redis://localhost:6379/0
```

A lightweight integration test exists under `tests/mock_orchestrator.py`. It patches heavy dependencies and exercises `main_async`:

```bash
pytest tests/mock_orchestrator.py
```

For a dockerised trace run, use the compose file provided in `tests/docker-compose.traces.yaml`.

