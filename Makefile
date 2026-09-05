# VoxFrontier developer shortcuts (GNU make; on Windows use `make` via
# Git Bash / WSL, or run the underlying commands directly).
.PHONY: help install install-dev lint format test test-cov run-all synth clean audit

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	python -m pip install .

install-dev: ## Install in editable mode with dev tools
	python -m pip install -e ".[dev]"
	pre-commit install

lint: ## Run ruff checks
	ruff check src tests

format: ## Auto-fix lint issues
	ruff check --fix src tests

test: ## Run the test suite
	python -m pytest

test-cov: ## Run tests with coverage report
	python -m pytest --cov=voxfrontier --cov-report=term-missing

synth: ## Generate the synthetic dataset
	vxf synth

run-all: ## Run the full analysis pipeline
	vxf run-all

clean: ## Remove build/runtime artefacts
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache \
		__pycache__ src/voxfrontier/__pycache__ output figures data/synthetic
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

audit: ## Verify run manifest against current output tables
	python -c "from voxfrontier.utils.manifest import verify_manifest; \
from pathlib import Path; \
import sys; \
v = verify_manifest(Path('output')); \
print(v); sys.exit(0 if all(v.values()) else 1)"
