variable "kubeconfig" {
  description = "Path to kubeconfig file"
  type        = string
  default     = "~/.kube/config"
}

variable "namespace" {
  description = "Namespace to deploy Osiris"
  type        = string
  default     = "osiris-prod"
}

variable "image_tag" {
  description = "Orchestrator image tag"
  type        = string
  default     = "v1.0.0"
}

variable "llm_sidecar_image_tag" {
  description = "LLM sidecar image tag"
  type        = string
  default     = "v1.0.0"
}

variable "otel_collector_endpoint" {
  description = "OTEL collector endpoint"
  type        = string
  default     = ""
}

variable "llm_sidecar_gpu_node_selector" {
  description = "Node selector labels for GPU nodes"
  type        = map(string)
  default     = {}
}

variable "replica_count" {
  description = "Number of orchestrator replicas"
  type        = number
  default     = 3
}

variable "llm_sidecar_replica_count" {
  description = "Number of LLM sidecar replicas"
  type        = number
  default     = 3
}

variable "llm_sidecar_resources" {
  description = "Resource requests and limits for the LLM sidecar"
  type        = any
  default = {
    limits = {
      cpu    = "2"
      memory = "4Gi"
    }
    requests = {
      cpu    = "1"
      memory = "2Gi"
    }
  }
}

variable "orchestrator_resources" {
  description = "Resource requests and limits for the orchestrator"
  type        = any
  default = {
    limits = {
      cpu    = "2"
      memory = "2Gi"
    }
    requests = {
      cpu    = "1"
      memory = "1Gi"
    }
  }
}

variable "additional_values" {
  description = "Additional Helm values for advanced configuration"
  type        = map(any)
  default     = {}
}
