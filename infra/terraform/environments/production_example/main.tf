terraform {
  required_version = ">= 1.3"
  required_providers {
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.11"
    }
  }
}

provider "helm" {
  kubernetes {
    config_path = var.kubeconfig
  }
}

provider "kubernetes" {
  config_path = var.kubeconfig
}

module "osiris" {
  source = "../../modules/osiris_sidecar"

  namespace                     = var.namespace
  image_tag                     = var.image_tag
  replica_count                 = var.replica_count
  llm_sidecar_image_tag         = var.llm_sidecar_image_tag
  llm_sidecar_replica_count     = var.llm_sidecar_replica_count
  llm_sidecar_resources         = var.llm_sidecar_resources
  otel_collector_endpoint       = var.otel_collector_endpoint
  llm_sidecar_gpu_node_selector = var.llm_sidecar_gpu_node_selector
  additional_values             = merge({
    orchestrator = {
      resources = var.orchestrator_resources
    }
    redis = {
      enabled = false
    }
  }, var.additional_values)
}

resource "kubernetes_horizontal_pod_autoscaler_v2" "orchestrator" {
  metadata {
    name      = "${module.osiris.release_name}-orchestrator"
    namespace = module.osiris.namespace
  }

  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = "${module.osiris.release_name}-orchestrator"
    }

    min_replicas = 3
    max_replicas = 10

    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = 70
        }
      }
    }
  }
}


