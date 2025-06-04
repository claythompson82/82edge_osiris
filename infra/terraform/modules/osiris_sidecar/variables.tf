variable "release_name" {
  description = "Helm release name"
  type        = string
  default     = "osiris"
}

variable "namespace" {
  description = "Kubernetes namespace to deploy Osiris"
  type        = string
  default     = "osiris"
}

variable "chart_path" {
  description = "Path to the Osiris Helm chart"
  type        = string
  default     = "../../../helm/osiris"
}

variable "chart_version" {
  description = "Version of the chart to deploy. Set to null to use local path without version"
  type        = string
  default     = null
}

variable "image_tag" {
  description = "Osiris orchestrator image tag"
  type        = string
  default     = "latest"
}

variable "replica_count" {
  description = "Number of orchestrator replicas"
  type        = number
  default     = 1
}

variable "llm_sidecar_image_tag" {
  description = "LLM sidecar image tag"
  type        = string
  default     = "latest"
}

variable "llm_sidecar_replica_count" {
  description = "Number of LLM sidecar replicas"
  type        = number
  default     = 1
}

variable "llm_sidecar_gpu_node_selector" {
  description = "Node selector map for scheduling the LLM sidecar on GPU nodes"
  type        = map(string)
  default     = {}
}

variable "llm_sidecar_resources" {
  description = "Resource requests and limits for the LLM sidecar"
  type        = any
  default     = {}
}

variable "otel_collector_endpoint" {
  description = "OTEL collector endpoint for traces"
  type        = string
  default     = ""
}

variable "additional_values" {
  description = "Additional values to merge into the Helm release"
  type        = map(any)
  default     = {}
}
