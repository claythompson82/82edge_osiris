# DGM Kernel Observability

To enable Prometheus scraping and install the Grafana dashboard, deploy the chart with monitoring enabled:

```bash
helm install dgm-kernel ./deploy/dgm-kernel-chart --set monitoring.enabled=true
```

After installation, a dashboard titled **DGM Kernel Metrics** will appear in Grafana with a panel showing the 5‑minute rate of `dgm_patch_apply_total`.

> _Screenshot of the dashboard goes here_
