# Makefile for LLM Sidecar

.PHONY: rebuild logs ci-lock

# Default service for logs if not specified
SVC ?= llm-sidecar

rebuild:
	docker compose build --no-cache && docker compose up -d --remove-orphans

logs:
	@docker compose logs -f $(SVC)

ci-lock:
	poetry lock --check
	docker compose config --quiet

# Example usage:
# make rebuild
# make logs
# make logs SVC=orchestrator
