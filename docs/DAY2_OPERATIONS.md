# Day 2 Operations

This guide covers common tasks for maintaining a running Osiris deployment.

## Checking Service Health
- Use `docker compose ps` or `kubectl get pods` to verify containers are running and healthy.
- The `/health` endpoint on each service returns `200` when the component is ready.
- Stream logs with `docker compose logs -f <service>` or `kubectl logs -f <pod>` to diagnose issues.

## Monitoring Resource Usage
- Enable the provided Prometheus rules (see `ops/prometheus/osiris_alerts.yaml`).
- Import the Grafana dashboard at `ops/grafana/osiris_observability.json` to view CPU, memory and GPU metrics.
- For quick checks, run `docker stats` or use `kubectl top pods` if metrics-server is installed.

## Backup and Restore Procedures
### LanceDB
- Data is stored under `/app/lancedb_data` in the `llm-sidecar` container.
- Mount this path as a persistent volume to keep history between restarts.
- Periodically copy the directory to durable storage (e.g., S3) and restore it by mounting the saved files.

### Redis
- The default Docker setup runs Redis with ephemeral storage. For persistence use a volume or enable RDB/AOF in a custom configuration.
- To back up, copy the `dump.rdb` or `appendonly.aof` files from the Redis data directory.

### Fine‑tuned Adapters
- If you run `scripts/nightly_qlora.sh` or `scripts/run_qlora.py`, save the adapters written to the `--output_dir`.
- Store them in version control or external storage and mount them when starting the sidecar.

## Scaling Considerations
- The `llm_sidecar` service can be scaled horizontally. When using Docker Compose increase `--scale llm-sidecar=<N>`.
- For Kubernetes, adjust `replicaCount` in `helm/osiris/values.yaml` or enable the `autoscaling` section.
- Ensure Redis and LanceDB have sufficient resources before scaling the orchestrator or sidecars.
- When `autoscaling.enabled` is true, the chart creates a Horizontal Pod Autoscaler using the settings under each component's `autoscaling` block.
- Set `pdb.enabled` to `true` to create a PodDisruptionBudget that keeps at least `minAvailable` pods running during node maintenance.

## Upgrading Osiris
1. Pull the latest container images or build them from the updated repository.
2. Apply database migrations if any are included.
3. Restart the services one by one, checking `/health` after each restart.
4. Re-import the Grafana dashboard if metrics or panels changed.
