# Makefile for Unified Cloud Functions

.PHONY: help install test lint format deploy clean

PROJECT_ID ?= $(shell gcloud config get-value project)
REGION ?= us-central1
FUNCTION_NAME = unified-security-fetcher

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linting"
	@echo "  make format     - Format code"
	@echo "  make deploy     - Deploy to Cloud Functions"
	@echo "  make clean      - Clean build artifacts"
	@echo "  make local      - Run locally with uvicorn"

install:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio fastapi[test] black flake8

test:
	ENABLE_SAMPLE_DATA=true PROJECT_ID=test-project pytest tests/ -v

lint:
	flake8 . --max-line-length=100 --exclude=.venv,__pycache__,tests
	python -m py_compile main.py
	python -m py_compile app/main.py

format:
	black . --line-length=100 --exclude=.venv

deploy:
	@if [ -z "$(PROJECT_ID)" ]; then \
		echo "Error: PROJECT_ID not set"; \
		exit 1; \
	fi
	./deploy.sh $(PROJECT_ID) $(REGION)

local:
	ENABLE_SAMPLE_DATA=true uvicorn app.main:app --reload --port 8000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage