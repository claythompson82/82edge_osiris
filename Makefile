# Makefile for LLM Sidecar

.PHONY: rebuild logs dev-shell openapi format-check lint-check test-coverage clean run-dev-sidecar docs-serve

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
	black --check .
	ruff format --check .

# Run static analysis linters
lint-check:
	ruff check .
	actionlint

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
