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

variable "replica_count" {
  description = "Number of orchestrator replicas"
  type        = number
  default     = 1
}

variable "llm_sidecar_replica_count" {
  description = "Number of LLM sidecar replicas"
  type        = number
  default     = 1
}

variable "llm_sidecar_resources" {
  description = "Resource requests and limits for the LLM sidecar"
  type        = any
  default     = {}
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
