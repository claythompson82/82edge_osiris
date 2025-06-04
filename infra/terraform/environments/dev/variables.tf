variable "kubeconfig" {
  description = "Path to kubeconfig file"
  type        = string
  default     = "~/.kube/config"
}

variable "namespace" {
  description = "Namespace to deploy Osiris"
  type        = string
  default     = "osiris-dev"
}

variable "image_tag" {
  description = "Orchestrator image tag"
  type        = string
  default     = "latest"
}

variable "llm_sidecar_image_tag" {
  description = "LLM sidecar image tag"
  type        = string
  default     = "latest"
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
