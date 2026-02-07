# Docker Support - Requirements

## 1. Overview

Add production-ready Docker support to the Stock Portfolio Analysis Agent to enable containerized deployment. This will allow users to run the application in isolated, reproducible environments and deploy to cloud platforms (Render, Railway, AWS, etc.) with minimal configuration.

## 2. User Stories

### 2.1 As a DevOps Engineer
I want to build a Docker image of the application so that I can deploy it consistently across different environments without worrying about Python version conflicts or dependency issues.

### 2.2 As a Developer
I want to run the application in a Docker container locally so that I can test the production environment on my machine before deploying.

### 2.3 As a Platform Engineer
I want a minimal Docker image so that deployment is fast, storage costs are low, and the attack surface is reduced.

### 2.4 As a Security Engineer
I want the Docker container to follow security best practices so that the application runs with minimal privileges and doesn't expose unnecessary files or secrets.

## 3. Acceptance Criteria

### 3.1 Dockerfile Creation
- **AC 3.1.1**: A `Dockerfile` must exist in the project root
- **AC 3.1.2**: The Dockerfile must use Python 3.12 slim base image for minimal size
- **AC 3.1.3**: The Dockerfile must install all dependencies from `pyproject.toml`
- **AC 3.1.4**: The Dockerfile must expose port 8000 for the FastAPI application
- **AC 3.1.5**: The Dockerfile must set appropriate environment variables (HOST, PORT, PYTHONUNBUFFERED, etc.)
- **AC 3.1.6**: The Dockerfile must use the correct entry point to start the uvicorn server via `main.py`

### 3.2 .dockerignore Creation
- **AC 3.2.1**: A `.dockerignore` file must exist in the project root
- **AC 3.2.2**: The `.dockerignore` must exclude version control files (.git)
- **AC 3.2.3**: The `.dockerignore` must exclude virtual environments (.venv, __pycache__)
- **AC 3.2.4**: The `.dockerignore` must exclude test files and caches (tests/, .pytest_cache/, .hypothesis/)
- **AC 3.2.5**: The `.dockerignore` must exclude local environment files (.env)
- **AC 3.2.6**: The `.dockerignore` must exclude OS-specific files (.DS_Store, Thumbs.db)

### 3.3 Build Process
- **AC 3.3.1**: The Docker image must build successfully with `docker build -t spa-agent .`
- **AC 3.3.2**: The build process must complete in under 5 minutes on standard hardware
- **AC 3.3.3**: The final image size must be under 1GB
- **AC 3.3.4**: All Python dependencies must be installed and importable in the container

### 3.4 Runtime Behavior
- **AC 3.4.1**: The container must start the FastAPI server on port 8000 when run
- **AC 3.4.2**: The container must accept COMPOSIO_API_KEY and OPENAI_API_KEY as environment variables
- **AC 3.4.3**: The container must log to stdout/stderr for container orchestration compatibility
- **AC 3.4.4**: The health endpoint (`/health`) must be accessible and return proper status
- **AC 3.4.5**: The container must handle graceful shutdown on SIGTERM/SIGINT

### 3.5 Documentation
- **AC 3.5.1**: The README must include a "Run with Docker" section
- **AC 3.5.2**: The README must document how to build the Docker image
- **AC 3.5.3**: The README must document how to run the container with required environment variables
- **AC 3.5.4**: The README must document how to access the health endpoint
- **AC 3.5.5**: The README must mention CORS_ORIGINS configuration for production

## 4. Technical Requirements

### 4.1 Base Image
- Use `python:3.12-slim` as the base image for optimal size/functionality balance
- Ensure compatibility with all project dependencies

### 4.2 Dependency Installation
- Use pip to install the project (via `pip install .`)
- Leverage the existing `pyproject.toml` and hatchling build system
- No need for uv in the container (simplifies the image)

### 4.3 Security Considerations
- Set `PYTHONDONTWRITEBYTECODE=1` to prevent .pyc file creation
- Set `PYTHONUNBUFFERED=1` for real-time logging
- Set `PIP_NO_CACHE_DIR=1` to reduce image size
- Install only minimal system dependencies (curl, ca-certificates)
- Clean up apt cache after installation

### 4.4 Entry Point
- Use the existing `main.py` entry point
- Command: `python -m uvicorn main:create_configured_app --host 0.0.0.0 --port 8000`
- Allow environment variables to override defaults

### 4.5 Port Configuration
- Expose port 8000 (FastAPI default)
- Set HOST=0.0.0.0 and PORT=8000 as default environment variables

## 5. Non-Functional Requirements

### 5.1 Performance
- Image build time: < 5 minutes
- Image size: < 1GB
- Container startup time: < 10 seconds

### 5.2 Maintainability
- Dockerfile must be well-commented
- Use multi-stage builds if beneficial (optional optimization)
- Follow Docker best practices (layer caching, minimal layers)

### 5.3 Compatibility
- Must work with Docker 20.10+
- Must work with Docker Compose (for future orchestration)
- Must be compatible with cloud container platforms (Render, Railway, AWS ECS, etc.)

## 6. Out of Scope

The following items are explicitly out of scope for this feature:

- Docker Compose configuration (future enhancement)
- Multi-stage builds (can be added later for optimization)
- Health check configuration in Dockerfile (handled by orchestration platforms)
- Volume mounts for persistent data (application is stateless)
- Custom entrypoint scripts (use existing main.py)

## 7. Dependencies

### 7.1 Existing Components
- `main.py` - Entry point for the FastAPI server
- `pyproject.toml` - Dependency specification
- `.env.example` - Environment variable template
- `agent/api.py` - FastAPI application

### 7.2 External Dependencies
- Docker 20.10 or higher
- Python 3.12 base image from Docker Hub

## 8. Success Metrics

- Docker image builds without errors
- Container starts and serves requests on port 8000
- Health endpoint returns 200 OK
- All 65 existing tests pass (no regression)
- README documentation is clear and actionable

## 9. Testing Strategy

### 9.1 Build Testing
- Verify Dockerfile builds successfully
- Verify final image size is acceptable
- Verify all dependencies are installed

### 9.2 Runtime Testing
- Verify container starts without errors
- Verify FastAPI server is accessible on port 8000
- Verify health endpoint responds correctly
- Verify environment variables are properly consumed
- Verify graceful shutdown works

### 9.3 Integration Testing
- Test with actual API keys (manual verification)
- Test API endpoints through the container
- Verify logs are visible via `docker logs`

## 10. Risks and Mitigations

### 10.1 Risk: Dependency Installation Failures
**Mitigation**: Use pip with explicit upgrade and verify imports after installation

### 10.2 Risk: Large Image Size
**Mitigation**: Use slim base image, clean up apt cache, use .dockerignore

### 10.3 Risk: Port Conflicts
**Mitigation**: Document port mapping clearly, allow PORT env var override

### 10.4 Risk: Missing System Dependencies
**Mitigation**: Install curl and ca-certificates for debugging and HTTPS support

## 11. Future Enhancements

- Add Docker Compose for local development with hot reload
- Add multi-stage build for smaller production images
- Add health check directive in Dockerfile
- Add support for different deployment targets (dev, staging, prod)
- Add container scanning for vulnerabilities
