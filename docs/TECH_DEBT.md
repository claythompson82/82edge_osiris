# Technical Debt Overview

This document captures areas of the codebase that may require future refactoring or cleanup.

## Orchestrator Configuration
- Some endpoints like the Phi-3 generation service and event bus Redis URL were previously hard-coded. They now fall back to environment variables (`PHI3_API_URL` and `EVENT_BUS_REDIS_URL`) but could benefit from a central configuration system.

## dgm_kernel Patch Generation
- `dgm_kernel/meta_loop.py` still relies on a placeholder patch generation function. See TODO at line 87 referencing [issue #99](https://github.com/82edge/osiris/issues/99).

## Broken Symlinks
- Files under `osiris/scripts/` are symbolic links ending with a trailing newline in the target path, making them unusable. This should be corrected by recreating the symlinks without newline characters.


