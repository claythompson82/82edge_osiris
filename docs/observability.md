---
## OpenTelemetry Configuration

The services emit traces using [OpenTelemetry](https://opentelemetry.io/). Set the following environment variables to enable tracing locally:

```
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_SERVICE_NAME=osiris_llm_sidecar
OTEL_TRACES_SAMPLER=parentbased_always_on
```

Adjust `OTEL_SERVICE_NAME` for other components (e.g. `osiris_orchestrator`). Traces can be collected with any OTLP compatible collector.

## Prometheus Alert Rules

Osiris includes a set of predefined Prometheus alert rules to help monitor key aspects of the system. These rules are defined in `ops/prometheus/osiris_alerts.yaml`.

When deploying Osiris via the Helm chart, these rules will be automatically installed as a `PrometheusRule` custom resource in your Kubernetes cluster, provided the `prometheus.rules.create` value is set to `true` in your `values.yaml` file (it is `true` by default). This requires a Prometheus Operator (commonly deployed as part of `kube-prometheus-stack`) in your cluster that can discover and process `PrometheusRule` resources.

The following alerts are defined:

*   **HighGpuVramUsage**
    *   **Description:** Monitors the GPU VRAM usage. Triggers if the average VRAM usage exceeds 90% for 2 minutes.
    *   **Metric Used (example):** `avg_over_time(dcgm_fb_used_bytes[2m]) / avg_over_time(dcgm_fb_total_bytes[2m]) * 100` (or `dcgm_fb_used_percent`). Ensure your GPU monitoring exposes these DCGM metrics.

*   **HighLLMErrorRate**
    *   **Description:** Monitors the error rate of LLM requests. Triggers if the error rate exceeds 5% over a 5-minute period.
    *   **Metric Used (placeholder):** `sum(rate(llm_requests_total{status="error"}[5m])) / sum(rate(llm_requests_total[5m])) * 100`
    *   **ACTION REQUIRED:** This alert uses placeholder metrics. You **must** review and update the `expr` in `ops/prometheus/osiris_alerts.yaml` to use the actual metrics your LLM services expose for tracking request counts and errors (e.g., specific counter names, labels for status, model ID, etc.).

*   **RedisBacklogTooHigh**
    *   **Description:** Monitors the length of a specific Redis list (queue). Triggers if the queue length exceeds 5000 items for 10 minutes.
    *   **Metric Used (placeholder):** `redis_list_length{list_name="your_main_job_queue"}`
    *   **ACTION REQUIRED:** This alert uses a placeholder Redis list name (`your_main_job_queue`). You **must** review and update the `list_name` label in the `expr` in `ops/prometheus/osiris_alerts.yaml` to match the actual Redis list name used by your Osiris application for its main job queue.

### Integrating Alerts with Notification Systems (PagerDuty, SNS, etc.)

Prometheus itself generates alerts, but it relies on the **Alertmanager** component to route these alerts to notification systems like PagerDuty, Slack, email, Opsgenie, SNS, etc.

If you have `kube-prometheus-stack` or a similar Prometheus setup, an Alertmanager instance is likely already running. To receive notifications for the Osiris alerts:

1.  **Configure Receivers:** Define one or more "receivers" in your Alertmanager configuration. A receiver specifies how to connect to a notification service (e.g., PagerDuty integration key, Slack API URL, SNS topic ARN).
2.  **Configure Routing:** Set up routing rules in Alertmanager to determine which alerts (based on their labels, like `severity`, `app`, etc.) should be sent to which receivers. For example, you might route `severity: critical` alerts to PagerDuty and `severity: warning` alerts to Slack.

Refer to the [official Alertmanager documentation](https://prometheus.io/docs/alerting/latest/configuration/) for detailed instructions on configuring receivers and routing. The Osiris alerts include labels like `severity` and `service` (commented out, but can be enabled) that you can use in your Alertmanager routing rules.

## Importing Grafana Dashboard

The Osiris Observability dashboard provides a centralized view of key metrics for the Osiris system, including service health, GPU performance, Redis queue depths, LLM token rates, and Sentry error summaries.

The dashboard definition is stored as a JSON file in the repository at `ops/grafana/osiris_observability.json`.

### Importing to Grafana Cloud (or any Grafana instance)

1.  **Navigate to Dashboards:**
    In your Grafana instance, go to the "Dashboards" section (usually found in the left-hand sidebar).
    *   `[SCREENSHOT PLACEHOLDER: Grafana sidebar with "Dashboards" highlighted]`

2.  **Open Import Dialog:**
    On the Dashboards page, look for an "Import" button (often near the top right or under a "New" dropdown) and click it.
    *   `[SCREENSHOT PLACEHOLDER: Grafana Dashboards page with "Import" button highlighted]`

3.  **Upload JSON File:**
    You'll be presented with options to import. Choose "Upload JSON file".
    *   `[SCREENSHOT PLACEHOLDER: Grafana import dialog showing "Upload JSON file" option]`
    Click the "Upload JSON file" button and select the `osiris_observability.json` file from your local checkout of the repository (`ops/grafana/osiris_observability.json`).

4.  **Configure Dashboard Options:**
    After uploading, Grafana will display options for the new dashboard:
    *   **Name:** You can keep the default "Osiris Observability" or change it.
    *   **Folder:** Choose a folder to save the dashboard in.
    *   **UID:** It's recommended to keep the existing UID (`osiris-observability-dashboard`) if you plan to update the dashboard via re-imports. If you want a completely new instance, you can clear this or let Grafana generate a new one.
    *   **Data Sources:** Crucially, you will need to map the placeholder data sources defined in the dashboard to your actual Grafana data sources.
        *   `DS_PROMETHEUS`: Select your Prometheus data source.
        *   `DS_SENTRY`: Select your Sentry data source.
    *   `[SCREENSHOT PLACEHOLDER: Grafana import options page, highlighting Name, Folder, UID, and especially the Data Source selection dropdowns for Prometheus and Sentry]`

5.  **Import:**
    Click the "Import" button. The dashboard should now be available in your Grafana instance.

### Dashboard Overview and Configuration

The dashboard is organized into several rows:

*   **General Service Health:**
    *   **Request Latency (p95/avg):** Shows the 95th percentile and average latency for HTTP requests to services like `llm-sidecar` and `orchestrator`. Uses Prometheus. Assumes metrics like `http_server_duration_seconds_bucket`.
*   **GPU Performance:**
    *   **GPU VRAM Usage:** Displays GPU memory usage as a percentage. Uses Prometheus. Assumes metrics like `dcgm_fb_used_bytes` and `dcgm_fb_total_bytes` from a DCGM exporter.
*   **Redis Queue:**
    *   **Redis Queue Depth:** Shows the length of important Redis lists (queues). Uses Prometheus. Assumes metrics like `redis_list_length` from a Redis exporter. You may need to adjust the `list_name` pattern in the panel query for your specific queue names.
*   **LLM Performance:**
    *   **LLM Tokens per Second:** Tracks the rate of tokens processed by Hermes and Phi3 models. Uses Prometheus. Assumes a counter metric like `llm_tokens_processed_total` with a `model_id` label.
*   **Error Tracking:**
    *   **Error Rate (Sentry):** Summarizes error events from Sentry. Uses the Sentry data source. You will likely need to configure the panel's "Project ID(s)" in the Sentry query options to match your Sentry project. The default query shows `event.type:error` grouped by `level`.

Remember to adjust template variable selections (Environment, Service) at the top of the dashboard to filter the data as needed.
---
