# Osiris "Production-like" Environment

This example demonstrates a more robust configuration using the Osiris Terraform module.
It showcases higher replica counts, explicit resource limits and the ability to
consume managed services such as cloud hosted Redis. Secrets are expected to be
populated by an external system, so none are hard coded here. A sample
Horizontal Pod Autoscaler is defined and ingress resources can be provided via
`additional_values`.

Run `terraform init` and `terraform plan` to validate the configuration.
