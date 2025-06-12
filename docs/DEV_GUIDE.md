# Development Guide

Below is a set of common tasks and environment notes for contributors.

## Table of Contents
- [Local development setup](#local-development-setup)
- [Pre-commit install](#pre-commit-install)
- [Dev environment check](#dev-environment-check)
- [Local harness](#local-harness)
- [Running sidecar and Redis](#running-sidecar-and-redis)
- [Pointing tests at local sidecar](#pointing-tests-at-local-sidecar)
- [Using the dev container](#using-the-dev-container)
- [Common workflows](#common-workflows)
- [CI cheatsheet](#ci-cheatsheet)
- [Testing](#testing)
- [Branch protection](#branch-protection)
- [Contribution workflow](#contribution-workflow)
- [Example environment](#example-environment)
- [pyenv with poetry](#pyenv-with-poetry)
- [venv with pip](#venv-with-pip)
- [Hot reload](#hot-reload)
- [Logging basics](#logging-basics)
- [Extra resources](#extra-resources)

## Local development setup
### Prerequisites
1. Docker and Docker Compose
2. Python 3.11
3. Git and `pre-commit`

Clone the repo and copy `.env.template` to `.env`. Adjust values as needed.
Use either `pyenv` with Poetry or the standard `venv` module. For a pyenv workflow, install Python 3.11 and run `poetry install`. The lock file pins all packages. For a lightweight approach, create a `venv` with `python -m venv .venv` and activate it. Then install with `pip install -e .` plus `pip install -r requirements-tests.txt` for the extras.

## Pre-commit install
Install the hooks with `pre-commit install` after installing the package. This ensures formatting and linting checks run automatically before each commit. Run `pre-commit autoupdate` to sync hook versions and `pre-commit run --all-files` one time to lint the entire repo.

## Running the Test Suite

You can now run all tests under your virtualenv’s Python:

```bash
# From the repo root, with .venv activated
make test


## Dev environment check
Run `python scripts/check_dev_env.py` to verify that your system meets the prerequisites.
The script reports missing tools, blocked ports, and Docker status so you can fix issues
before running the services.

## Local harness
The local harness spins up the policy orchestrator and simulator using your Python environment. Run `make dev-shell` to enter a Poetry shell with your environment variables loaded. From there run `python -m osiris_policy.orchestrator --redis_url redis://localhost:6379/0`. This setup lets you iterate on policies without building new Docker images.

## Running sidecar and Redis
Use the provided compose file to run the sidecar along with Redis. Execute:
```bash
docker compose -f docker/compose.yaml up redis llm-sidecar
```
The API becomes available at `http://localhost:8000` and Redis on `6379`. Stop services with `docker compose down`.

## Pointing tests at local sidecar
Tests expect the sidecar URL at `OSIRIS_SIDECAR_URL`. When running a local instance set:
```bash
export OSIRIS_SIDECAR_URL=http://localhost:8000
```
This directs the integration tests at your local sidecar instead of staging.

## Using the dev container
VS Code users can develop inside a container without installing Python locally.
Install the **Remote - Containers** extension, then choose **"Reopen in Container"**.
The configuration under `.devcontainer/` builds the image from `Dockerfile`,
installs the project in editable mode, and exposes ports `8000` and `6379`.
The container has all test dependencies so you can run `pytest` immediately.

## Common workflows
- Added a new reward function? Run:
  ```bash
  pytest -k reward
  ```
- Changed adapter logic? Execute:
  ```bash
  make adapters-smoke
  ```
These commands run the focused test suites quickly before opening a PR.

## CI cheatsheet
The main checks run on every PR are:
1. CI
2. LLM Sidecar Smoke Test
3. E2E Adapter Swap
4. E2E Orchestrator Smoke Test
Ensure these pass locally before pushing large changes. Review `.github/workflows` for the exact commands.

## Testing
Install extras for pytest with:
```bash
pip install -r requirements-tests.txt
```
Run `pytest` from the repo root. Use `-k` to filter tests. For coverage metrics run `pytest --cov`.

## Branch protection
Set up required checks with `scripts/ci/setup_branch_protection.sh`. This script uses the GitHub CLI and must be run by a repo admin. It configures the four checks listed in the CI cheatsheet and enforces a linear history on `main`.

## Contribution workflow
1. Create a feature branch from `main`.
2. Install dependencies and run `pre-commit install`.
3. Make your changes and run `pre-commit` followed by `pytest`.
4. Push the branch and open a pull request.
5. Address review comments and ensure CI passes before merging.

## Example environment
The `.env.template` file documents all configurable variables. Copy it to `.env` and edit values such as API keys or model paths. Keep secrets out of source control.

## pyenv with poetry
1. Install pyenv and a compatible Python:
   ```bash
   pyenv install 3.11.12
   pyenv local 3.11.12
   ```
2. Install Poetry and run `poetry install`.
3. Activate the shell with `poetry shell` when developing.

## venv with pip
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install the package and extras:
   ```bash
   pip install -e .
   pip install -r requirements-tests.txt
   ```

## Hot reload
For rapid iteration, run the sidecar directly with uvicorn:
```bash
OSIRIS_SIDECAR_URL=http://localhost:8000 uvicorn llm_sidecar.server:app --reload
```
Code changes are reloaded automatically so you can test API tweaks without rebuilding Docker images.

## Logging basics
Logs are printed to stdout. Set `LOG_LEVEL=debug` in your `.env` for verbose output. When running under Docker, use `docker compose logs -f llm-sidecar` to stream logs. Important events are also sent to Redis channels for visibility during orchestration.

## Extra resources
- `ARCH.md` describes system components.
- `PROJECT_OVERVIEW.md` explains repository layout.
- Review `README.md` for quick start instructions.

## Data ingestion
Market data CSVs live under `data/`. Use the simulator to feed this data into Redis. Example command:
```bash
python -m sim.engine data/spy_1h.csv --redis_url redis://localhost:6379/0 --speed 10x
```
Customize `--speed` and date range flags to test various scenarios.
New options allow simulating trading frictions:

```bash
python -m sim.engine data/spy_1h.csv \
  --redis_url redis://localhost:6379/0 \
  --order_channel market.orders \
  --commission_rate 0.001 \
  --slippage_impact 0.1
```

The engine listens for JSON order messages on `--order_channel` and applies the
configured commission and slippage models when updating the portfolio.

## GPU troubleshooting
If the container fails due to VRAM limits, lower `MAX_TOKENS` in the `.env` or switch to CPU mode by setting `DEVICE=cpu`. The `vram_watchdog` service can automatically restart the sidecar when usage exceeds a threshold.

## Build and deploy
To build the Docker image manually and supply a Hugging Face token use BuildKit secrets:
```bash
docker build --secret id=hf_token_secret,src=/path/to/token.txt -t osiris .
```
To rebuild all containers via Compose:
```bash
make rebuild
```
Images are tagged with the Git commit. Push to your registry if deploying remotely. The orchestrator and sidecar containers look for updated images on each start.

## Docker cleanup
Dangling images can consume disk space. Remove them with:
```bash
docker image prune -f
```
You can also prune unused volumes via `docker volume prune`. Be careful, this deletes data.

## Debugging tips
Set `DEBUG=true` in your `.env` to enable extra verbose traces. When diagnosing test failures, run pytest with `-vv` and check the `tests/logs` directory for captured output. Use `pdb.set_trace()` in the code to drop into an interactive debugger.

Common logs can be followed with Docker:
```bash
docker compose logs -f llm-sidecar        # LLM API
docker compose logs -f orchestrator       # Policy loop
docker compose logs -f redis              # Event bus
```
LanceDB stores data under `./lancedb_data`. Inspect it with `ls lancedb_data` or the LanceDB CLI.

## File layout
- `llm_sidecar/` – sidecar server code
- `osiris_policy/` – orchestrator logic and adapters
- `tests/` – pytest suites
- `docker/` – compose files and Dockerfiles
Refer to `PROJECT_OVERVIEW.md` for a full breakdown.

## Additional notes
Patience profits. Keep your branches small and push early drafts to trigger CI. Reviewers appreciate descriptive commit messages and minimal force pushes.

## Adapter guide
Adapters connect policy logic to broker interfaces. They live in `osiris_policy/adapters/`. Each adapter implements a common interface for submitting orders. Unit tests under `tests/adapters/` verify correctness. Use the `make adapters-smoke` target for quick validation when working on these modules.

## Reward evaluation
Reward functions score proposed trades. They are found in `llm_sidecar/reward/`. When adding a new function ensure it is pure so it can be unit tested. Run `pytest -k reward` to execute only these tests.

## Model updates
Models are downloaded at build time. To refresh them, delete the `models/` directory and rebuild the sidecar container. The compose file mounts `./models` for persistence between runs so large downloads are avoided on each start.

## Redis administration
Connect with `redis-cli` to inspect channels or flush data. Running `redis-cli monitor` is helpful when debugging event streams. Remember that a `redis-cli FLUSHALL` will clear any in-memory state the orchestrator relies on.

## Network configuration
Docker compose puts all services on a private network. Ports 8000 (sidecar) and 6379 (Redis) are published to localhost. Adjust the `ports` section in `docker/compose.yaml` if you need to expose the services to other machines on your LAN.

## Coding standards
The repo uses Black and Ruff. Avoid long import chains and prefer explicit relative imports within packages. Keep functions short and pure when possible. Run `pre-commit run --files <file>` before committing to catch lint errors early.

## Pull request process
Feature branches should be based on `main`. Push early and often to run CI. Provide a clear description in the PR body and link to any open issues. Once approved, squash merge to keep history tidy.

## Sim loop details
The simulator publishes ticks to `market.ticks` in Redis. The orchestrator subscribes and triggers a proposal every N ticks. Adjust `ticks_per_proposal` on the orchestrator command line to tune how frequently models are queried.

## Test data generation
Synthetic candles can be generated with `scripts/gen_fake_data.py`. Pass the desired number of rows and the script writes a CSV suitable for the simulator. Use this when adding edge case tests.

## IDE setup
The project works well with VS Code. Install the Python extension and point it at the `.venv` interpreter. Enable "format on save" to run Black automatically. For Poetry users, the `poetry env info -p` path can be supplied to the editor if needed.

## Continuous integration
All pushes trigger GitHub Actions defined under `.github/workflows/`. These workflows build the Docker images and run the smoke tests. Locally you can invoke the same commands via `make ci-local` if you want to replicate the pipeline.

## Tox usage
Some developers prefer using Tox to manage multi-Python testing. A sample `tox.ini` is included. Run `tox -e py311` to execute the default environment which mirrors the CI configuration.

## Local database
The default build stores data in `./lancedb_data`. If you need a fresh state, remove this directory before starting the compose stack. Data is preserved between runs so your test strategies have history to work with.

## Network proxies
If your environment requires a proxy, set `HTTP_PROXY` and `HTTPS_PROXY` in `.env` so Docker and Python tooling can reach external resources. These variables are automatically passed into the containers when you use `docker compose`.

## Clean shutdown
Always stop services with `docker compose down` to ensure volumes are unmounted cleanly. Leaving containers running while editing code can lead to stale dependencies if the file watchers miss changes.

## Terminal hotkeys
When running tests or the simulator interactively, press `Ctrl+C` to exit gracefully. For Docker logs, `Ctrl+C` will detach if you used `docker compose logs -f`.

## Unit test layout
Tests live under `tests/` with subfolders mirroring the code layout. Fixtures are in `tests/conftest.py`. Naming your test files `test_*.py` ensures pytest discovers them automatically.

## Git hooks
The repo's pre-commit config installs hooks under `.git/hooks/`. If you need to bypass them temporarily, use `git commit --no-verify`. This should be rare as the hooks catch most style issues.

## Automated formatting
Running `pre-commit run` formats Python files with Black and checks style via Ruff. CI enforces the same rules, so commit formatted code to avoid failing builds.

## Metadata
Documentation pages live in `docs/` and are served as static files from the sidecar. Keep diagrams in `docs/img/` and reference them with relative paths so they display correctly in GitHub.

## Changelog process
User facing changes should be summarized in `CHANGELOG.md`. Follow the existing sections for new features, bug fixes, and documentation updates. A concise entry helps maintainers track progress between releases.

## Static analysis
Run `ruff --select I` to automatically sort imports. The CI pipeline checks for unused variables and other common mistakes via Ruff.

## Linting YAML
CI runs `yamllint` over workflow files. You can run it locally with:
```bash
yamllint .github
```

## Formatting docs
Markdown files should wrap lines at 100 characters. Use fenced code blocks for terminal commands so they render properly in GitHub.

## Release process
Tags on the `main` branch trigger a Docker build and push to the registry. Ensure `CHANGELOG.md` is updated and version numbers are bumped in `pyproject.toml` before tagging.

## Issue triage
When new issues are opened, label them and add a short comment acknowledging the report. Bugs should reference failing tests or log output where possible.

## Documentation style
Write in a concise, active voice. Code snippets should be minimal but complete enough to copy and paste. Use headings liberally so readers can skim for context.

## Coding fonts
A monospaced font improves readability of console output. In VS Code select a font like Fira Code with ligatures disabled if you prefer a classic look.

## Third-party services
Some features rely on external APIs such as brokerage data. When developing without credentials, mock these calls in tests using the `tests/mocks/` helpers.

## Data retention
Large test logs and artifacts can accumulate. Periodically clear `tests/logs` and remove old Docker volumes to keep the repository lightweight.

## Time zones
Simulated data is in UTC. If your system timezone differs, set `TZ=UTC` in `.env` to avoid confusion when comparing logs and chart data.

## Common mistakes
- Forgetting to set `OSIRIS_SIDECAR_URL` when running tests
- Not activating the virtual environment before installing packages
- Running `docker compose` without copying `.env`

## Container logs
Use `make logs` or `docker compose logs -f llm-sidecar` to follow the sidecar output. The orchestrator service can be inspected similarly with `SVC=orchestrator`.

## Node dependencies
A few static assets are built with Node tools. Install `npm` and run `npm ci` in the `static/` directory if you modify the web console files. The pipeline caches these dependencies for faster builds.

## Text to speech
The sidecar exposes a `/speak` endpoint for generating WAV data. During tests, this call is mocked. To hear responses locally, point your browser at `static/audio_console.html` once the sidecar is running.

## Roadmap
Future work includes expanding the set of adapters and improving the feedback loop for Phi-3. Contributions are welcome. See `issues/` for open tasks.

## Branch naming
Use short, descriptive branch names like `feat/reward-fn` or `bugfix/sidecar-redis`. Avoid spaces or special characters.

## Nightly jobs
A GitHub Actions workflow triggers the feedback fine-tuning every night. Check `nightly_trainer/` for details on the training scripts. Logs from this run are available in the Actions tab for maintainers.

## Bug bash
Before each release we run a bug bash where all failing tests must be addressed. Use the `tests/expected_failures.txt` file to track known issues temporarily.

## Testing containers
You can run `docker compose -f docker/compose.yaml exec llm-sidecar pytest` to execute tests inside the container environment. This mirrors the CI build closely.

## Credentials
Sensitive keys should be loaded from the environment only. Avoid committing them
to git history. A `.gitignore` file now excludes `.env` and other secret files by
default. Copy `.env.template` to `.env` for local use and store real credentials
there. See [Secrets Management Review](SECURITY_SECRETS.md) for hardening tips.

## Long-running tasks
For lengthy simulations or training loops, consider running inside `tmux` or `screen` so the session survives SSH disconnects. Logs can be tailed in another window while the job runs.

## Getting help
If you hit a blocker, open a discussion in GitHub or ping the maintainers on the team chat. Screenshots of errors and full command logs make it easier to diagnose problems quickly.

## Contact
For repo admins, reach out via the `@osiris-maintainers` team on GitHub. We aim to respond within one business day to new issues or questions.

## Appendix
This guide is a living document. Feel free to submit pull requests to improve clarity or add missing topics as the project evolves.

## Performance tuning
Try reducing `max_length` in generation requests to lower GPU memory usage. Profiling scripts under `scripts/perf/` can help identify bottlenecks in your policy code.

## Monitoring
Prometheus metrics are exposed on `/metrics`. Scrape this endpoint to track latency and error counts during long test runs.

## Advanced orchestration
For experiments involving multiple policies, edit `configs/orchestrator.yaml` to list each policy module. The orchestrator will load them in sequence and aggregate their suggestions.

## Final words
Thanks for contributing to Osiris. Happy coding!
