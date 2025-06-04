# Example Environments

This directory contains example Terraform configurations for deploying Osiris using the shared module.

Two sample environments are provided:

- `dev` – a lightweight setup intended for development and testing.
- `staging` – a configuration that more closely mirrors production settings.

To deploy an environment, change into the desired directory and run:

```bash
terraform init
terraform plan
# terraform apply  # Uncomment to perform an actual deployment
```

Ensure your `kubeconfig` variable points to the Kubernetes cluster you wish to manage.
