#!/bin/bash

# Configure branch protection for the main branch using the GitHub API.
# Requires the GitHub CLI (gh) to be authenticated with repo admin permissions.
set -euo pipefail

REPO="${GITHUB_REPOSITORY:-$(git config --get remote.origin.url | sed -E 's#.*github.com[:/]([^/]+/[^.]+)(\.git)?#\1#')}"
BRANCH="main"

read -r -d '' PAYLOAD <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "checks": [
      {"context": "CI"},
      {"context": "LLM Sidecar Smoke Test"},
      {"context": "E2E Adapter Swap"},
      {"context": "E2E Orchestrator Smoke Test"}
    ]
  },
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true
  },
  "required_linear_history": true,
  "enforce_admins": false
}
JSON

echo "Setting branch protection on ${REPO}:${BRANCH}..."

gh api \
    --method PATCH \
    repos/${REPO}/branches/${BRANCH}/protection \
    --input - <<<"${PAYLOAD}"

echo "Branch protection updated."
