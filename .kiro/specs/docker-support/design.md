# Docker Support - Design Document

## 1. Design Overview

This design implements production-ready Docker containerization for the Stock Portfolio Analysis Agent. The solution provides a minimal, secure, and efficient Docker image that can be deployed to any container platform while maintaining compatibility with the existing application architecture.

### 1.1 Design Goals

1. **Minimal Image Size**: Use slim base image and efficient layering to keep image under 1GB
2. **Security First**: Follow Docker security best practices (non-root user consideration, minimal dependencies, no secrets in image)
3. **Production Ready**: Support environment-based configuration, proper logging, and graceful shutdown
4. **Developer Friendly**: Simple build and run commands with clear documentation
5. **Platform Agnostic**: Compatible with Docker, Kubernetes, and cloud platforms (Render, Railway, AWS ECS)

### 1.2 Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Use `python:3.12-slim` base | Balances size (~150MB) with functionality; includes necessary system libraries |
| Install via pip (not uv) | Simplifies container; uv adds complexity and size; pip works with pyproject.toml |
| Single-stage build | Simpler to maintain; multi-stage can be added later if size becomes critical |
| Use existing main.py entry point | Leverages existing startup logic; no custom entrypoint script needed |
| Environment variable configuration | Standard Docker pattern; allows runtime configuration without rebuilding |

## 2. Component Design

### 2.1 Dockerfile Structure

The Dockerfile follows a layered approach optimized for caching and security:

```dockerfile
# Layer 1: Base image and environment setup
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Layer 2: System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Layer 3: Application setup
WORKDIR /app
COPY . /app

# Layer 4: Python dependencies
RUN python -m pip install --upgrade pip \
    && pip install . \
    && python -c "import agent, fastapi, yfinance, pandas, numpy, pydantic; print('Deps OK')"

# Layer 5: Runtime configuration
ENV HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000

# Layer 6: Entry point
CMD ["python", "-m", "uvicorn", "main:create_configured_app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2.1.1 Layer Breakdown

**Layer 1 - Base Image & Python Environment**
- Base: `python:3.12-slim` (Debian-based, ~150MB)
- `PYTHONDONTWRITEBYTECODE=1`: Prevents .pyc file creation (reduces size, improves security)
- `PYTHONUNBUFFERED=1`: Ensures logs appear in real-time (critical for container orchestration)
- `PIP_NO_CACHE_DIR=1`: Prevents pip cache (reduces image size by ~50-100MB)

**Layer 2 - System Dependencies**
- `curl`: Useful for health checks and debugging
- `ca-certificates`: Required for HTTPS connections to OpenAI, Composio, yFinance
- `--no-install-recommends`: Minimizes installed packages
- `rm -rf /var/lib/apt/lists/*`: Cleans up apt cache (saves ~20-30MB)

**Layer 3 - Application Files**
- `WORKDIR /app`: Sets working directory
- `COPY . /app`: Copies entire project (filtered by .dockerignore)

**Layer 4 - Python Dependencies**
- Upgrades pip to latest version
- Installs project via `pip install .` (uses pyproject.toml + hatchling)
- Verification import: Ensures critical dependencies are importable

**Layer 5 - Runtime Configuration**
- Sets default HOST and PORT
- Exposes port 8000 for documentation (actual binding happens at runtime)

**Layer 6 - Entry Point**
- Uses uvicorn to run the FastAPI app
- Calls `main:create_configured_app` factory function
- Binds to 0.0.0.0:8000 (overridable via environment variables)

### 2.2 .dockerignore Design

The `.dockerignore` file prevents unnecessary files from being copied into the image:

```
# Version Control
.git
.gitignore

# Python Virtual Environments
.venv
venv/
__pycache__/
*.pyc
*.pyo
*.pyd

# Testing
tests/
.tests/
.pytest_cache/
.hypothesis/
*.coverage
.coverage.*
htmlcov/

# OS Files
.DS_Store
Thumbs.db
*.swp
*.swo

# Local Configuration
.env
.env.local
.env.*.local

# Documentation (optional - include if needed)
# README.md
# docs/

# Build artifacts
dist/
build/
*.egg-info/

# IDE
.vscode/
.idea/
*.sublime-*
```

#### 2.2.1 Exclusion Categories

1. **Version Control**: `.git` directory (can be 10-50MB+)
2. **Virtual Environments**: `.venv`, `__pycache__` (100-500MB+)
3. **Test Files**: `tests/`, `.pytest_cache/`, `.hypothesis/` (not needed in production)
4. **OS Files**: `.DS_Store`, `Thumbs.db` (metadata files)
5. **Local Config**: `.env` files (secrets should never be in image)
6. **Build Artifacts**: `dist/`, `*.egg-info/` (regenerated during build)

### 2.3 Entry Point Design

**Validates Requirements: 3.1.6, 3.4.1**

The container uses the existing `main.py` module as the entry point:

```bash
CMD ["python", "-m", "uvicorn", "main:create_configured_app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2.3.1 Why This Approach?

1. **Reuses Existing Logic**: `main.py` already handles:
   - Environment variable loading (via python-dotenv)
   - Logging configuration
   - Component initialization (session manager, orchestrator)
   - Graceful shutdown hooks

2. **Factory Pattern**: `create_configured_app()` returns a configured FastAPI app with:
   - Startup event handlers
   - Shutdown event handlers
   - All middleware and routes registered

3. **Environment Variable Support**: The existing code reads:
   - `HOST` (default: 0.0.0.0)
   - `PORT` (default: 8000)
   - `COMPOSIO_API_KEY` (required)
   - `OPENAI_API_KEY` (required)
   - `LOG_LEVEL` (default: INFO)
   - `WORKERS` (default: 1)
   - `RELOAD` (default: false)

4. **No Custom Script Needed**: Avoids complexity of shell scripts or custom entrypoints

### 2.4 Environment Variable Configuration

**Validates Requirements: 3.4.2, 3.5.5**

The container accepts the following environment variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `COMPOSIO_API_KEY` | ✅ Yes | - | Composio API authentication |
| `OPENAI_API_KEY` | ✅ Yes | - | OpenAI API authentication |
| `HOST` | No | `0.0.0.0` | Server bind address |
| `PORT` | No | `8000` | Server port |
| `WORKERS` | No | `1` | Uvicorn worker processes |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |
| `CORS_ORIGINS` | No | `*` | Allowed CORS origins (comma-separated) |
| `RELOAD` | No | `false` | Hot reload (should be false in production) |

#### 2.4.1 Configuration Flow

```
Docker Run Command
    ↓
Environment Variables (-e flags)
    ↓
main.py reads os.environ
    ↓
Validates required vars (COMPOSIO_API_KEY, OPENAI_API_KEY)
    ↓
Configures logging (LOG_LEVEL)
    ↓
Initializes components (session manager, orchestrator)
    ↓
Starts uvicorn server (HOST, PORT, WORKERS)
```

## 3. File Specifications

### 3.1 Dockerfile

**Location**: `stock-portfolio-analysis-agent/Dockerfile`

**Validates Requirements**: 3.1.1, 3.1.2, 3.1.3, 3.1.4, 3.1.5, 3.1.6

**Content**: See section 2.1 for complete specification

**Key Features**:
- Python 3.12 slim base image
- Optimized layer caching
- Security-focused environment variables
- Minimal system dependencies
- Verification step for critical imports

### 3.2 .dockerignore

**Location**: `stock-portfolio-analysis-agent/.dockerignore`

**Validates Requirements**: 3.2.1, 3.2.2, 3.2.3, 3.2.4, 3.2.5, 3.2.6

**Content**: See section 2.2 for complete specification

**Impact**: Reduces image size by 100-500MB by excluding:
- Virtual environments (~200-400MB)
- Git history (~10-50MB)
- Test files (~10-50MB)
- Cache directories (~20-100MB)

### 3.3 README Updates

**Location**: `stock-portfolio-analysis-agent/README.md`

**Validates Requirements**: 3.5.1, 3.5.2, 3.5.3, 3.5.4, 3.5.5

**New Section**: "Run with Docker (production)"

**Content**:
```markdown
### Run with Docker (production)

```bash
# Build
docker build -t spa-agent .

# Run (set your API keys)
docker run \
  -e COMPOSIO_API_KEY=your_composio_api_key \
  -e OPENAI_API_KEY=your_openai_api_key \
  -e PORT=8000 -e HOST=0.0.0.0 \
  -p 8000:8000 \
  spa-agent
```

- Health: GET http://localhost:8000/health
- Sync analyze: POST http://localhost:8000/analyze/sync
- Streaming analyze (SSE): POST http://localhost:8000/analyze

Set `CORS_ORIGINS` env var (comma-separated) to restrict cross-origin access in production.
```

**Placement**: After the "Running the Application" section, before "User Flow"

## 4. Build and Deployment Workflow

### 4.1 Local Development Workflow

```bash
# 1. Build the image
docker build -t spa-agent:local .

# 2. Run with environment variables
docker run \
  -e COMPOSIO_API_KEY=sk_xxx \
  -e OPENAI_API_KEY=sk-xxx \
  -e LOG_LEVEL=DEBUG \
  -p 8000:8000 \
  spa-agent:local

# 3. Test health endpoint
curl http://localhost:8000/health

# 4. Test API endpoint
curl -X POST http://localhost:8000/analyze/sync \
  -H "Content-Type: application/json" \
  -d '{"query": "What if I invested $10k in AAPL since 2020?"}'

# 5. View logs
docker logs <container_id>

# 6. Stop container
docker stop <container_id>
```

### 4.2 Production Deployment Workflow

#### 4.2.1 Cloud Platform Deployment (Render, Railway)

```bash
# 1. Push to GitHub
git add Dockerfile .dockerignore
git commit -m "Add Docker support"
git push origin main

# 2. Configure platform (Render/Railway)
# - Connect GitHub repository
# - Set environment variables (COMPOSIO_API_KEY, OPENAI_API_KEY)
# - Platform auto-detects Dockerfile
# - Platform builds and deploys

# 3. Verify deployment
curl https://your-app.onrender.com/health
```

#### 4.2.2 Manual Docker Registry Deployment

```bash
# 1. Build and tag
docker build -t your-registry/spa-agent:v1.0.0 .

# 2. Push to registry
docker push your-registry/spa-agent:v1.0.0

# 3. Deploy to target environment
# (Kubernetes, ECS, etc. - platform-specific)
```

## 5. Testing Strategy

### 5.1 Build Testing

**Validates Requirements**: 3.3.1, 3.3.2, 3.3.3, 3.3.4

```bash
# Test 1: Build succeeds
docker build -t spa-agent:test .
# Expected: Exit code 0, no errors

# Test 2: Build time
time docker build -t spa-agent:test .
# Expected: < 5 minutes

# Test 3: Image size
docker images spa-agent:test
# Expected: < 1GB

# Test 4: Dependency verification
docker run --rm spa-agent:test python -c "import agent, fastapi, yfinance, pandas, numpy, pydantic; print('OK')"
# Expected: "OK" printed, exit code 0
```

### 5.2 Runtime Testing

**Validates Requirements**: 3.4.1, 3.4.2, 3.4.3, 3.4.4, 3.4.5

```bash
# Test 1: Container starts
docker run -d --name spa-test \
  -e COMPOSIO_API_KEY=test_key \
  -e OPENAI_API_KEY=test_key \
  -p 8000:8000 \
  spa-agent:test
# Expected: Container ID returned

# Test 2: Health endpoint responds
sleep 5  # Wait for startup
curl http://localhost:8000/health
# Expected: {"status": "degraded", "composio_configured": true, "openai_configured": true}

# Test 3: Logs visible
docker logs spa-test
# Expected: Startup logs visible, no errors

# Test 4: Graceful shutdown
docker stop spa-test
# Expected: Container stops within 10 seconds

# Cleanup
docker rm spa-test
```

### 5.3 Integration Testing

```bash
# Test with real API keys (manual)
docker run -d --name spa-integration \
  -e COMPOSIO_API_KEY=$COMPOSIO_API_KEY \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -p 8000:8000 \
  spa-agent:test

# Wait for startup
sleep 10

# Test analyze endpoint
curl -X POST http://localhost:8000/analyze/sync \
  -H "Content-Type: application/json" \
  -d '{"query": "What if I invested $10k in AAPL since 2020?"}'
# Expected: Valid portfolio analysis response

# Cleanup
docker stop spa-integration
docker rm spa-integration
```

## 6. Security Considerations

### 6.1 Image Security

1. **No Secrets in Image**: API keys passed via environment variables at runtime
2. **Minimal Base Image**: `python:3.12-slim` reduces attack surface
3. **No Cache**: `PIP_NO_CACHE_DIR=1` prevents cached credentials
4. **Clean Apt Lists**: Removes package manager metadata

### 6.2 Runtime Security

1. **Environment Variables**: Secrets injected at runtime, not build time
2. **Read-Only Filesystem**: Application doesn't write to disk (stateless)
3. **Non-Root User**: (Future enhancement) Run as non-root user
4. **Network Isolation**: Only port 8000 exposed

### 6.3 Supply Chain Security

1. **Official Base Image**: Use official Python image from Docker Hub
2. **Pinned Dependencies**: `uv.lock` ensures reproducible builds
3. **Verification Step**: Import check ensures dependencies installed correctly

## 7. Performance Considerations

### 7.1 Build Performance

**Validates Requirements**: 5.1

- **Layer Caching**: Dockerfile structured for optimal caching
  - System dependencies change rarely (cached)
  - Application code changes frequently (rebuilt)
- **Parallel Downloads**: pip installs dependencies in parallel
- **Expected Build Time**: 2-4 minutes on standard hardware

### 7.2 Runtime Performance

**Validates Requirements**: 5.1

- **Startup Time**: < 10 seconds
  - Python interpreter: ~1s
  - Import dependencies: ~3-5s
  - Initialize components: ~2-4s
- **Memory Usage**: ~200-400MB baseline
- **CPU Usage**: Minimal at idle, scales with requests

### 7.3 Image Size Optimization

**Validates Requirements**: 3.3.3

| Component | Size | Optimization |
|-----------|------|--------------|
| Base image | ~150MB | Use slim variant |
| System deps | ~20MB | Minimal packages, clean cache |
| Python deps | ~300-500MB | No dev dependencies |
| Application | ~5-10MB | Exclude tests, .git via .dockerignore |
| **Total** | **~500-700MB** | ✅ Under 1GB target |

## 8. Monitoring and Observability

### 8.1 Health Checks

**Validates Requirements**: 3.4.4

The existing `/health` endpoint provides container health status:

```json
{
  "status": "healthy",
  "composio_configured": true,
  "openai_configured": true
}
```

**Usage in Container Orchestration**:
```yaml
# Kubernetes example
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30
```

### 8.2 Logging

**Validates Requirements**: 3.4.3

- **Stdout/Stderr**: All logs go to stdout (Docker standard)
- **Log Level**: Configurable via `LOG_LEVEL` environment variable
- **Structured Logging**: Existing format includes timestamp, level, message
- **Request IDs**: Existing middleware adds X-Request-ID to responses

### 8.3 Metrics (Future Enhancement)

- Add Prometheus metrics endpoint
- Track request count, latency, error rates
- Monitor container resource usage

## 9. Troubleshooting Guide

### 9.1 Common Issues

**Issue**: Container exits immediately
```bash
# Check logs
docker logs <container_id>

# Common causes:
# - Missing API keys
# - Port already in use
# - Import errors
```

**Issue**: Health endpoint returns 503
```bash
# Check if API keys are set
docker exec <container_id> env | grep API_KEY

# Check logs for initialization errors
docker logs <container_id> | grep ERROR
```

**Issue**: Build fails on dependency installation
```bash
# Clear Docker cache and rebuild
docker build --no-cache -t spa-agent .

# Check if pyproject.toml is valid
cat pyproject.toml
```

### 9.2 Debug Mode

```bash
# Run with debug logging
docker run \
  -e COMPOSIO_API_KEY=xxx \
  -e OPENAI_API_KEY=xxx \
  -e LOG_LEVEL=DEBUG \
  -p 8000:8000 \
  spa-agent

# Interactive shell for debugging
docker run -it --entrypoint /bin/bash spa-agent
```

## 10. Future Enhancements

### 10.1 Multi-Stage Build

Optimize image size further by separating build and runtime stages:

```dockerfile
# Build stage
FROM python:3.12-slim AS builder
WORKDIR /app
COPY . .
RUN pip install --user .

# Runtime stage
FROM python:3.12-slim
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "main:create_configured_app", "--host", "0.0.0.0"]
```

**Benefit**: Reduces image size by ~100-200MB

### 10.2 Docker Compose

Add `docker-compose.yml` for local development:

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - COMPOSIO_API_KEY=${COMPOSIO_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=DEBUG
    volumes:
      - ./agent:/app/agent  # Hot reload for development
```

### 10.3 Non-Root User

Run container as non-root user for enhanced security:

```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

### 10.4 Health Check Directive

Add Docker health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

## 11. Correctness Properties

### 11.1 Build Correctness

**Property 1.1**: Image builds successfully with valid pyproject.toml
```
∀ valid_pyproject → docker build succeeds
```

**Property 1.2**: All dependencies are importable after build
```
∀ dependency ∈ pyproject.toml → import succeeds in container
```

**Property 1.3**: Image size is under threshold
```
image_size < 1GB
```

### 11.2 Runtime Correctness

**Property 2.1**: Container starts with required environment variables
```
∀ (COMPOSIO_API_KEY, OPENAI_API_KEY) → container starts successfully
```

**Property 2.2**: Health endpoint responds correctly
```
GET /health → status ∈ {healthy, degraded} ∧ response_time < 1s
```

**Property 2.3**: Server binds to correct port
```
container_running → port 8000 is listening
```

### 11.3 Configuration Correctness

**Property 3.1**: Environment variables override defaults
```
∀ env_var → container_config uses env_var value over default
```

**Property 3.2**: Missing required variables cause graceful failure
```
¬(COMPOSIO_API_KEY ∨ OPENAI_API_KEY) → health status = degraded
```

## 12. Implementation Checklist

- [ ] Create Dockerfile with Python 3.12 slim base
- [ ] Create .dockerignore with exclusions
- [ ] Update README with Docker section
- [ ] Test: Build image successfully
- [ ] Test: Image size under 1GB
- [ ] Test: Container starts and serves requests
- [ ] Test: Health endpoint responds
- [ ] Test: Environment variables work correctly
- [ ] Test: Graceful shutdown works
- [ ] Test: All existing tests still pass (no regression)
- [ ] Document: Add troubleshooting section
- [ ] Document: Add deployment examples

## 13. Acceptance Validation

This design satisfies all acceptance criteria from the requirements document:

| Requirement | Design Section | Validation Method |
|-------------|----------------|-------------------|
| AC 3.1.1-3.1.6 | Section 2.1, 3.1 | Build test |
| AC 3.2.1-3.2.6 | Section 2.2, 3.2 | File inspection |
| AC 3.3.1-3.3.4 | Section 5.1 | Build testing |
| AC 3.4.1-3.4.5 | Section 5.2 | Runtime testing |
| AC 3.5.1-3.5.5 | Section 3.3 | Documentation review |
