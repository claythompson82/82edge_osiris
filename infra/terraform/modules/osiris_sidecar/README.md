# Osiris Helm Chart Terraform Module

This module deploys the Osiris application using its Helm chart. It exposes variables for common configuration options.

## Usage

```hcl
module "osiris" {
  source = "../modules/osiris_sidecar"

  namespace               = "osiris-dev"
  image_tag               = "1.1.0"
  llm_sidecar_image_tag   = "1.1.0"
  otel_collector_endpoint = "http://otel-collector:4317"
}
```

Run `terraform init` and `terraform apply` in an environment directory to deploy the chart.

## Variables

- `release_name` – Helm release name.
- `namespace` – Namespace for deployment.
- `chart_path` – Path to the Helm chart.
- `chart_version` – Version of the chart (optional).
- `image_tag` – Orchestrator image tag.
- `replica_count` – Orchestrator replica count.
- `llm_sidecar_image_tag` – LLM sidecar image tag.
- `llm_sidecar_replica_count` – LLM sidecar replica count.
- `llm_sidecar_gpu_node_selector` – Map of node selector labels for GPU scheduling.
- `llm_sidecar_resources` – Resource requests and limits for the sidecar.
- `otel_collector_endpoint` – OTEL collector endpoint.
- `additional_values` – Additional values map passed to the chart.

Outputs provide the release name, namespace, and deployment status.
