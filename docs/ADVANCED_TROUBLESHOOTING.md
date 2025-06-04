# Advanced Troubleshooting

This page lists techniques for diagnosing issues in a production Osiris deployment.

## Common Runtime Issues
- **Orchestrator not processing ticks**: ensure `market.ticks` events are published to Redis and that the orchestrator container is subscribed. Use `redis-cli monitor` to watch events.
- **LLM errors**: check the `llm_sidecar` logs for stack traces. Many issues come from missing models or insufficient GPU memory. Lower `MAX_TOKENS` or run on CPU with `DEVICE=cpu` if needed.
- **Performance degradation**: use the Grafana dashboard to monitor CPU, memory and GPU usage. High latency may indicate the sidecar is overloaded.

## Interpreting Logs
- Logs are JSON structured. Use `jq` when tailing files or streaming with `docker compose logs -f`.
- The orchestrator logs workflow IDs and decisions. LanceDB also records each run for later inspection via `scripts/harvest_feedback.py`.
- Set `LOG_LEVEL=debug` in the service environment to increase verbosity.

## Using the Observability Stack
- Prometheus scrapes metrics from each service. Import `ops/prometheus/osiris_alerts.yaml` to enable alerts.
- Traces emitted via OpenTelemetry can be collected with any OTLP-compatible collector. Set `OTEL_EXPORTER_OTLP_ENDPOINT` to your collector URL.
- The Grafana dashboard (`ops/grafana/osiris_observability.json`) displays latency, GPU usage, Redis depth and error rates. Use the Service/Environment filters at the top to narrow down issues.

## Common Failure Modes
- **Redis backlog too high**: the orchestrator may be unable to keep up. Scale the orchestrator or sidecar services and investigate slow tasks.
- **GPU memory exhaustion**: the VRAM watchdog container will restart the sidecar when usage stays above 90%. Consider reducing batch sizes or switching to CPU for low throughput environments.
- **Missing LanceDB tables**: ensure the `lancedb_data` volume is mounted and writable. The sidecar creates tables on startup if they do not exist.

If problems persist, compare your deployment against the reference `docker/compose.yaml` or Helm chart to ensure all environment variables and volumes are configured correctly.
