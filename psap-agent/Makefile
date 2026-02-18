.PHONY: local dev test clean

local:
	@echo "Setting up and starting Template Agent locally..."
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment..."; \
		uv venv; \
	fi
	@echo "Activating virtual environment..."
	@source .venv/bin/activate && \
	echo "Installing dependencies..." && \
	uv pip install -e ".[dev]" && \
	echo "Starting with in-memory storage..." && \
	USE_INMEMORY_SAVER=true python -m template_agent.src.main

dev:
	@echo "Starting Template Agent with podman-compose..."
	@podman-compose up --build

test:
	@echo "Running tests..."
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment..."; \
		uv venv; \
	fi
	@source .venv/bin/activate && \
	echo "Installing dependencies..." && \
	uv pip install -e ".[dev]" && \
	echo "Running pytest..." && \
	pytest

clean:
	@echo "Cleaning up non-code artifacts..."
	@rm -rf .venv
	@rm -rf __pycache__
	@rm -rf .pytest_cache
	@rm -rf .coverage
	@rm -rf htmlcov
	@rm -rf .mypy_cache
	@rm -rf .ruff_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -f Current-IT-Root-CAs.pem
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "Cleanup complete"
