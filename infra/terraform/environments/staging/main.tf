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
}
