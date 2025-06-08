# Makefile for LLM Sidecar

.PHONY: rebuild logs dev-shell openapi format-check lint-check test-coverage clean run-dev-sidecar docs-serve install

# Default service for logs if not specified
SVC ?= llm-sidecar

rebuild:
	docker compose build --no-cache && docker compose up -d --remove-orphans

logs:
	@docker compose logs -f $(SVC)

dev-shell:
	dotenv -f .env -- poetry shell

openapi:
	PYTHONPATH=. python scripts/generate_openapi.py

# Example usage:
# make rebuild
# make logs
# make logs SVC=orchestrator

# Check code formatting without modifying files
format-check:
	python -m black --check .

# Run static analysis linters
lint-check:
	python -m ruff check .
	@command -v actionlint >/dev/null && actionlint || echo "actionlint not installed"

# Execute tests with a coverage report
test-coverage:
	pytest --cov=.

# Remove caches and build artifacts
clean:
	find . -name '__pycache__' -type d -exec rm -rf {} +
	rm -rf .pytest_cache build dist

# Start the LLM sidecar with hot reloading
run-dev-sidecar:
	OSIRIS_SIDECAR_URL=http://localhost:8000 uvicorn llm_sidecar.server:app --reload

# Serve documentation locally
docs-serve:
	mkdocs serve -f mkdocs.yml

install:
       pip install -r requirements.txt
       pip install -r requirements-dev.txt

# Check for up-to-date lock files
lock:
	pip-compile --dry-run --quiet --generate-hashes --output-file=requirements.txt requirements.in -c constraints_cpu.txt
	pip-compile --dry-run --quiet --generate-hashes --output-file=requirements-dev.txt requirements-dev.in -c constraints_cpu.txt

# Ensure Docker is available
docker-check:
	@docker info > /dev/null 2>&1 || (echo "Docker is not running" && exit 1)

# CI job to verify locks when running in CI
ci-lock:
	@if [ -n "$(CI)" ]; then $(MAKE) lock; else echo "Skipping lock check"; fi
