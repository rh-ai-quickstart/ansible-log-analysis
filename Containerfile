# ============================================================================
# Builder Stage - Install dependencies
# ============================================================================
FROM registry.access.redhat.com/ubi8/python-312 AS builder
USER root

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (optimal layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies only (not the project itself yet)
# Set environment to prevent CUDA dependencies and increase timeout
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=600 \
    TORCH_CUDA_ARCH_LIST="" \
    uv sync --frozen --no-install-project --no-dev

# Copy source code and install project in production mode
COPY README.md ./
COPY src/ ./src/
COPY data/logs/failed/ ./data/logs/failed/

# Install the project itself (production mode, not editable)
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=600 \
    TORCH_CUDA_ARCH_LIST="" \
    uv sync --frozen --no-dev --no-editable

# ============================================================================
# Runtime Stage
# ============================================================================
FROM registry.access.redhat.com/ubi8/python-312
USER root

WORKDIR /app

# Copy the complete virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application files
COPY --from=builder /app/src /app/src
COPY --from=builder /app/README.md /app/
COPY --from=builder /app/pyproject.toml /app/

# Copy additional data files
COPY data/knowledge_base/ ./data/knowledge_base/
COPY data/logs/failed/ ./data/logs/failed/
COPY backend_init_pipeline.py ./

# Set environment variables
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/hf_cache

# Create necessary directories and set permissions for OpenShift (random UID, group 0)
RUN mkdir -p /app/data/logs/failed /hf_cache && \
    chgrp -R 0 /app /hf_cache && \
    chmod -R g=u /app /hf_cache

# Expose port
EXPOSE 8000

# Default command
ENTRYPOINT ["uvicorn", "alm.main_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]