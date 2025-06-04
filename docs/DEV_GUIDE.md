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
   ```

- `pip install -r requirements-tests.txt` installs extras needed only for running tests.

## Testing

pip install -r requirements-tests.txt before running pytest.

