output "release_name" {
  value       = helm_release.osiris.name
  description = "Name of the Helm release"
}

output "namespace" {
  value       = helm_release.osiris.namespace
  description = "Kubernetes namespace where Osiris is deployed"
}

output "status" {
  value       = helm_release.osiris.status
  description = "Deployment status"
}
