# Makefile for LLM Sidecar

.PHONY: rebuild logs dev-shell

# Default service for logs if not specified
SVC ?= llm-sidecar

rebuild:
	docker compose build --no-cache && docker compose up -d --remove-orphans

logs:
	@docker compose logs -f $(SVC)

dev-shell:
	dotenv -f .env -- poetry shell

# Example usage:
# make rebuild
# make logs
# make logs SVC=orchestrator
