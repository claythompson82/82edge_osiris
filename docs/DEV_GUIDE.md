# Development Guide

This document outlines the recommended steps for a local development setup.

## Setting up a dev env

1. Copy the environment template and adjust any settings:

   ```bash
   cp .env.template .env
   # Edit .env to match your environment
   ```

2. Install the repository in editable mode:

   ```bash
   pip install -e .
   pip install pre-commit
   pre-commit install
   pre-commit autoupdate && pre-commit run --all-files
   ```

- `pip install -r requirements-tests.txt` installs extras needed only for running tests.

## Testing

pip install -r requirements-tests.txt before running pytest.


## Branch protection

Run `scripts/ci/setup_branch_protection.sh` once to enable the required status checks on the `main` branch. The script uses the GitHub CLI and needs an authenticated session with permission to edit repository settings.

After running it, `gh api repos/:owner/:repo/branches/main/protection` should list the four checks.
