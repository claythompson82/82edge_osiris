---
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
