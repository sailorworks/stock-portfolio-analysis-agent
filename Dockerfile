# Stock Portfolio Analysis Agent - Production Dockerfile
# Base image: Python 3.12 slim for minimal size and security
FROM python:3.12-slim AS base

# Environment variables for Python optimization and security
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
# - curl: useful for health checks and debugging
# - ca-certificates: required for HTTPS connections to OpenAI, Composio, yFinance
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files to container
COPY . /app

# Install Python dependencies via pip
# - Upgrade pip to latest version
# - Install project and all dependencies from pyproject.toml
# - Verify critical dependencies are importable
RUN python -m pip install --upgrade pip \
    && pip install . \
    && python -c "import agent, fastapi, yfinance, pandas, numpy, pydantic; print('Deps OK')"

# Runtime environment configuration
ENV HOST=0.0.0.0 \
    PORT=8000

# Expose port 8000 for FastAPI application
EXPOSE 8000

# Start uvicorn server using the main.py entry point
CMD ["python", "-m", "uvicorn", "main:create_configured_app", "--host", "0.0.0.0", "--port", "8000"]
