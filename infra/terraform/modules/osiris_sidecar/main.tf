resource "helm_release" "osiris" {
  name             = var.release_name
  namespace        = var.namespace
  chart            = var.chart_path
  create_namespace = true
  version          = var.chart_version

  values = [yamlencode({
    replicaCount = var.replica_count
    image = {
      tag = var.image_tag
    }
    config = {
      OTEL_EXPORTER_OTLP_ENDPOINT = var.otel_collector_endpoint
    }
    llmSidecar = {
      replicaCount = var.llm_sidecar_replica_count
      image = {
        tag = var.llm_sidecar_image_tag
      }
      nodeSelector = var.llm_sidecar_gpu_node_selector
      resources    = var.llm_sidecar_resources
      config = {
        OTEL_EXPORTER_OTLP_ENDPOINT = var.otel_collector_endpoint
      }
    }
  })]
}
