# Development Guide

This document outlines the recommended steps for a local development setup.

## Local Setup

1. Copy the environment template and adjust any settings:

   ```bash
   cp .env.template .env
   # Edit .env to match your environment
   ```

2. Install the repository in editable mode:

   ```bash
   pip install -e .
   ```

3. Run the tests to verify your environment:

   ```bash
   pytest
   ```
