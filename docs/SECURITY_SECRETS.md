# Secrets Management Review

This document summarizes how secrets are handled in the Osiris project and lists
recommended improvements.

## Current State

* Local development uses environment files (`.env`) copied from `.env.template`.
* A sample environment file exists at `configs/env/dev.env` which previously
  included a real looking `POLYGON_API_KEY`.
* CI/CD workflows reference GitHub secrets for publishing images and docs.
* Docker and Kubernetes manifests rely on environment variables for injecting
  credentials. The `HF_TOKEN` is now provided at build time using a BuildKit
  secret (`hf_token_secret`) instead of a build argument. Compose files still
  contain placeholders in `docker/docker-compose.cloud.yaml`.

## Issues Identified

* `configs/env/dev.env` contained a hard coded API key. Even if the key was not
  active, keeping it in version control risks accidental exposure.
* The repository lacked a `.gitignore`, increasing the chance of committing
  local `.env` files containing credentials.

## Hardening Steps

1. Replaced the `POLYGON_API_KEY` value in `configs/env/dev.env` with a placeholder
   and documented that real values belong in untracked `.env` files.
2. Added a `.gitignore` that excludes common secret locations such as `.env`
   files and the `configs/env` directory.
3. Updated developer documentation to highlight best practices for handling
   secrets and to reference this guide.

## Recommendations

* Use a secrets manager (e.g. AWS Secrets Manager or HashiCorp Vault) for
  production deployments. Mount secrets as files or inject them as environment
  variables at runtime.
* Rotate credentials regularly and restrict access based on the principle of
  least privilege.
* Avoid storing sensitive data in Docker images or source control. Use build
  arguments or runtime environment variables instead.


