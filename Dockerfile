# syntax=docker/dockerfile:1.6

# ───────────────────────────────────────────────────────────────────────────
# Stage 1: builder — install dependencies into a virtualenv
# ───────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build-time system deps. We need libjpeg/webp/png headers for Pillow wheels
# that don't ship binary wheels for slim, and rust-based wheels (tokenizers)
# need build-essential.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        libjpeg-dev \
        libwebp-dev \
        libpng-dev \
        libtiff-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated venv so the runtime stage gets a clean copy.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip setuptools wheel \
    && pip install .

# ───────────────────────────────────────────────────────────────────────────
# Stage 2: runtime — slim image with just what we need to run
# ───────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/data/hf_cache \
    BRAND_DNA_OUTPUT_DIR=/app/outputs \
    BRAND_DNA_CACHE_DIR=/app/data/cache

# Runtime system deps for WeasyPrint (Pango/Cairo/GDK-PixBuf) and image libs.
# These are smaller than the -dev variants used in the builder.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        libwebp7 \
        libpng16-16 \
        libtiff6 \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        libffi8 \
        fonts-dejavu-core \
        fonts-liberation \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv from the builder stage. This is the magic of multi-stage builds:
# we leave the build toolchain behind.
COPY --from=builder /opt/venv /opt/venv

# Non-root user. Production-grade: never run as root.
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

# Copy source + configs + prompts so the package can find prompt templates
# regardless of CWD.
COPY --chown=app:app src/ /app/src/
COPY --chown=app:app configs/ /app/configs/
COPY --chown=app:app pyproject.toml /app/pyproject.toml

# Re-install in editable mode so the venv references /app/src (lets us swap
# code via volume mounts for dev without rebuilding).
RUN pip install -e .

# Output / cache dirs (mountable volumes)
RUN mkdir -p /app/outputs /app/data/cache /app/data/hf_cache \
    && chown -R app:app /app

USER app

# Tini handles signals and zombie reaping — important for async runners.
ENTRYPOINT ["/usr/bin/tini", "--", "brand-dna"]
CMD ["--help"]
