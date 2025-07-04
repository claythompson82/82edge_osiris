# DGM Kernel Dashboard

The dashboard definition lives at `ops/grafana/dgm_dashboard.json`. To use it:

1. Open your Grafana instance and choose **Dashboards** → **Import**.
2. Upload `dgm_dashboard.json` and leave the UID as `dgm-kernel` so future updates can overwrite it.
3. Map `DS_PROMETHEUS` to your Prometheus data source and finish the import.

See the Grafana documentation for more details on [importing dashboards](https://grafana.com/docs/grafana/latest/dashboards/manage-dashboards/).
