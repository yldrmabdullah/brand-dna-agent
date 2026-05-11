.PHONY: install install-dev test lint format type-check clean run-cos run-les-benjamins docker-build docker-run

# ─── Setup ────────────────────────────────────────────────────────────────
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	playwright install chromium

install-web:
	pip install -e ".[web]"

serve:
	uvicorn brand_dna.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

# ─── Quality ──────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=brand_dna --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

type-check:
	mypy src/brand_dna

quality: format lint type-check test

# ─── Runs ─────────────────────────────────────────────────────────────────
# WeasyPrint on macOS needs to find Homebrew's libpango/libcairo/libgobject.
# On Linux these are typically picked up via ldconfig; on macOS we export here.
HOMEBREW_LIB := /opt/homebrew/lib
export DYLD_FALLBACK_LIBRARY_PATH := $(HOMEBREW_LIB):$(DYLD_FALLBACK_LIBRARY_PATH)

run-cos:
	brand-dna run --config configs/brands/cos.yaml

run-arket:
	brand-dna run --config configs/brands/arket.yaml

run-les-benjamins:
	brand-dna run --config configs/brands/les_benjamins.yaml

run:
	@if [ -z "$(CONFIG)" ]; then echo "Usage: make run CONFIG=configs/brands/<brand>.yaml"; exit 1; fi
	brand-dna run --config $(CONFIG)

# ─── Docker ───────────────────────────────────────────────────────────────
docker-build:
	docker build -t brand-dna-agent:latest .

docker-run:
	@if [ -z "$(CONFIG)" ]; then echo "Usage: make docker-run CONFIG=configs/brands/<brand>.yaml"; exit 1; fi
	docker run --rm \
		--env-file .env \
		-v $(PWD)/outputs:/app/outputs \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/configs:/app/configs:ro \
		brand-dna-agent:latest run --config $(CONFIG)

# ─── Housekeeping ─────────────────────────────────────────────────────────
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

clean-outputs:
	rm -rf outputs/* data/cache/*
