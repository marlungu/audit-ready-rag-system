.PHONY: help install dev db-up db-down init-db ingest serve ask test evaluate lint clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

dev: ## Start full stack with docker-compose
	docker compose up --build -d

db-up: ## Start only PostgreSQL
	docker compose up -d postgres

db-down: ## Stop all services
	docker compose down

init-db: ## Initialize database schema and indexes
	python -m scripts.init_db

ingest: ## Run document ingestion with quality gates
	python -m scripts.embed_documents

serve: ## Start FastAPI server locally
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

ask: ## Start interactive CLI
	python -m scripts.ask

test: ## Run test suite
	python -m pytest tests/ -v

evaluate: ## Run evaluation harness
	python -m scripts.evaluate

lint: ## Run linter
	python -m ruff check app/ scripts/ tests/

clean: ## Remove generated files
	rm -f eval_results.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
