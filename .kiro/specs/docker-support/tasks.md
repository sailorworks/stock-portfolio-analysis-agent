# Docker Support - Implementation Tasks

## Task Overview

This task list implements production-ready Docker support for the Stock Portfolio Analysis Agent. Tasks are organized sequentially with clear acceptance criteria and testing requirements.

---

## 1. Create Dockerfile

**Status**: [ ] Not Started
**Priority**: 🔴 MUST HAVE (MVP)

**Description**: Create a production-ready Dockerfile that builds a minimal, secure container image for the FastAPI application.

**Requirements**: AC 3.1.1, 3.1.2, 3.1.3, 3.1.4, 3.1.5, 3.1.6

**Implementation Details**:
- Use `python:3.12-slim` as base image
- Set environment variables: PYTHONDONTWRITEBYTECODE=1, PYTHONUNBUFFERED=1, PIP_NO_CACHE_DIR=1
- Install system dependencies: curl, ca-certificates
- Copy project files to /app
- Install Python dependencies via `pip install .`
- Add verification step: `python -c "import agent, fastapi, yfinance, pandas, numpy, pydantic; print('Deps OK')"`
- Set runtime environment: HOST=0.0.0.0, PORT=8000
- Expose port 8000
- Set CMD to start uvicorn: `python -m uvicorn main:create_configured_app --host 0.0.0.0 --port 8000`

**File Location**: `stock-portfolio-analysis-agent/Dockerfile`

**Acceptance Criteria**:
- [ ] Dockerfile exists in project root
- [ ] Uses Python 3.12 slim base image
- [ ] Sets all required environment variables
- [ ] Installs curl and ca-certificates
- [ ] Cleans up apt cache after installation
- [ ] Copies project files to /app
- [ ] Installs dependencies via pip
- [ ] Includes dependency verification step
- [ ] Exposes port 8000
- [ ] Uses correct uvicorn command as entry point

**Testing**:
```bash
# Verify file exists
test -f stock-portfolio-analysis-agent/Dockerfile

# Verify content includes key elements
grep "python:3.12-slim" stock-portfolio-analysis-agent/Dockerfile
grep "PYTHONUNBUFFERED=1" stock-portfolio-analysis-agent/Dockerfile
grep "pip install ." stock-portfolio-analysis-agent/Dockerfile
grep "uvicorn main:create_configured_app" stock-portfolio-analysis-agent/Dockerfile
```

---

## 2. Create .dockerignore

**Status**: [ ] Not Started
**Priority**: 🔴 MUST HAVE (MVP)

**Description**: Create a .dockerignore file to exclude unnecessary files from the Docker build context, reducing image size and build time.

**Requirements**: AC 3.2.1, 3.2.2, 3.2.3, 3.2.4, 3.2.5, 3.2.6

**Implementation Details**:
- Exclude version control: .git, .gitignore
- Exclude virtual environments: .venv, venv/, __pycache__/, *.pyc, *.pyo, *.pyd
- Exclude test files: tests/, .tests/, .pytest_cache/, .hypothesis/, *.coverage, htmlcov/
- Exclude OS files: .DS_Store, Thumbs.db, *.swp, *.swo
- Exclude local config: .env, .env.local, .env.*.local
- Exclude build artifacts: dist/, build/, *.egg-info/
- Exclude IDE files: .vscode/, .idea/, *.sublime-*

**File Location**: `stock-portfolio-analysis-agent/.dockerignore`

**Acceptance Criteria**:
- [ ] .dockerignore exists in project root
- [ ] Excludes .git directory
- [ ] Excludes .venv and __pycache__
- [ ] Excludes tests/ and test cache directories
- [ ] Excludes .env files
- [ ] Excludes .DS_Store and Thumbs.db
- [ ] Excludes build artifacts

**Testing**:
```bash
# Verify file exists
test -f stock-portfolio-analysis-agent/.dockerignore

# Verify key exclusions
grep "\.git" stock-portfolio-analysis-agent/.dockerignore
grep "\.venv" stock-portfolio-analysis-agent/.dockerignore
grep "tests/" stock-portfolio-analysis-agent/.dockerignore
grep "\.env" stock-portfolio-analysis-agent/.dockerignore
```

---

## 3. Build Docker Image

**Status**: [ ] Not Started
**Priority**: 🔴 MUST HAVE (MVP)

**Description**: Build the Docker image and verify it completes successfully within acceptable time and size constraints.

**Requirements**: AC 3.3.1, 3.3.2, 3.3.3

**Dependencies**: Tasks 1, 2

**Implementation Details**:
- Run `docker build -t spa-agent:test .` from project root
- Monitor build time (should be < 5 minutes)
- Check final image size (should be < 1GB)
- Verify no build errors

**Acceptance Criteria**:
- [ ] Docker build completes successfully (exit code 0)
- [ ] Build time is under 5 minutes
- [ ] Final image size is under 1GB
- [ ] No error messages in build output
- [ ] Build output shows "Deps OK" from verification step

**Testing**:
```bash
cd stock-portfolio-analysis-agent

# Build with timing
time docker build -t spa-agent:test .

# Check image size
docker images spa-agent:test --format "{{.Size}}"

# Verify image exists
docker images spa-agent:test | grep spa-agent
```

**Expected Output**:
- Build completes in 2-4 minutes
- Image size: ~500-700MB
- Exit code: 0

---

## 4. Verify Dependencies in Container

**Status**: [ ] Not Started
**Priority**: 🟡 NICE TO HAVE

**Description**: Verify that all required Python dependencies are correctly installed and importable in the container.

**Requirements**: AC 3.3.4

**Dependencies**: Task 3

**Implementation Details**:
- Run container with import test command
- Verify all critical dependencies are importable
- Check for any import errors

**Acceptance Criteria**:
- [ ] agent module imports successfully
- [ ] fastapi imports successfully
- [ ] yfinance imports successfully
- [ ] pandas imports successfully
- [ ] numpy imports successfully
- [ ] pydantic imports successfully
- [ ] No import errors in output

**Testing**:
```bash
# Test imports
docker run --rm spa-agent:test python -c "import agent, fastapi, yfinance, pandas, numpy, pydantic; print('All imports OK')"

# Test specific agent modules
docker run --rm spa-agent:test python -c "from agent.api import app; from agent.session import get_session_manager; print('Agent modules OK')"
```

**Expected Output**:
- "All imports OK" printed
- "Agent modules OK" printed
- Exit code: 0

---

## 5. Test Container Startup

**Status**: [ ] Not Started
**Priority**: 🔴 MUST HAVE (MVP)

**Description**: Test that the container starts successfully and the FastAPI server becomes available.

**Requirements**: AC 3.4.1, 3.4.3

**Dependencies**: Task 3

**Implementation Details**:
- Start container with test API keys
- Wait for server startup (max 10 seconds)
- Verify server is listening on port 8000
- Check logs for successful startup messages
- Verify no error messages in logs

**Acceptance Criteria**:
- [ ] Container starts without errors
- [ ] Server starts within 10 seconds
- [ ] Port 8000 is listening
- [ ] Logs show "API startup complete"
- [ ] No error messages in startup logs

**Testing**:
```bash
# Start container in background
docker run -d --name spa-test \
  -e COMPOSIO_API_KEY=test_key \
  -e OPENAI_API_KEY=test_key \
  -p 8000:8000 \
  spa-agent:test

# Wait for startup
sleep 10

# Check if container is running
docker ps | grep spa-test

# Check logs
docker logs spa-test | grep "API startup complete"

# Verify no errors
! docker logs spa-test | grep -i "error"

# Cleanup
docker stop spa-test
docker rm spa-test
```

**Expected Output**:
- Container ID returned
- Container status: Up
- Logs contain "API startup complete"
- No error messages

---

## 6. Test Environment Variable Configuration

**Status**: [ ] Not Started
**Priority**: 🟡 NICE TO HAVE

**Description**: Verify that environment variables are correctly passed to and consumed by the container.

**Requirements**: AC 3.4.2

**Dependencies**: Task 5

**Implementation Details**:
- Start container with various environment variables
- Verify API keys are recognized
- Test HOST, PORT, LOG_LEVEL configuration
- Verify CORS_ORIGINS configuration (if applicable)

**Acceptance Criteria**:
- [ ] COMPOSIO_API_KEY is recognized
- [ ] OPENAI_API_KEY is recognized
- [ ] HOST environment variable works
- [ ] PORT environment variable works
- [ ] LOG_LEVEL environment variable works
- [ ] Container uses provided values over defaults

**Testing**:
```bash
# Test with custom environment variables
docker run -d --name spa-env-test \
  -e COMPOSIO_API_KEY=custom_composio_key \
  -e OPENAI_API_KEY=custom_openai_key \
  -e LOG_LEVEL=DEBUG \
  -p 8001:8000 \
  spa-agent:test

# Wait for startup
sleep 10

# Check logs for DEBUG level
docker logs spa-env-test | grep DEBUG

# Verify environment variables are set
docker exec spa-env-test env | grep COMPOSIO_API_KEY
docker exec spa-env-test env | grep OPENAI_API_KEY
docker exec spa-env-test env | grep LOG_LEVEL

# Cleanup
docker stop spa-env-test
docker rm spa-env-test
```

**Expected Output**:
- Environment variables are set correctly
- DEBUG logs appear
- Container uses custom configuration

---

## 7. Test Health Endpoint

**Status**: [ ] Not Started
**Priority**: 🟡 NICE TO HAVE

**Description**: Verify that the /health endpoint is accessible and returns correct status information.

**Requirements**: AC 3.4.4

**Dependencies**: Task 5

**Implementation Details**:
- Start container with test API keys
- Wait for server startup
- Make HTTP request to /health endpoint
- Verify response structure and content
- Test both "healthy" and "degraded" states

**Acceptance Criteria**:
- [ ] /health endpoint responds with 200 OK
- [ ] Response includes "status" field
- [ ] Response includes "composio_configured" field
- [ ] Response includes "openai_configured" field
- [ ] Status is "healthy" when both keys are set
- [ ] Status is "degraded" when keys are missing
- [ ] Response time is under 1 second

**Testing**:
```bash
# Test with API keys (healthy state)
docker run -d --name spa-health-test \
  -e COMPOSIO_API_KEY=test_key \
  -e OPENAI_API_KEY=test_key \
  -p 8000:8000 \
  spa-agent:test

sleep 10

# Test health endpoint
curl -s http://localhost:8000/health | jq .

# Verify response structure
curl -s http://localhost:8000/health | jq -e '.status'
curl -s http://localhost:8000/health | jq -e '.composio_configured'
curl -s http://localhost:8000/health | jq -e '.openai_configured'

# Cleanup
docker stop spa-health-test
docker rm spa-health-test

# Test without API keys (degraded state)
docker run -d --name spa-health-degraded \
  -p 8000:8000 \
  spa-agent:test

sleep 10

# Check degraded status
curl -s http://localhost:8000/health | jq -e '.status == "degraded"'

# Cleanup
docker stop spa-health-degraded
docker rm spa-health-degraded
```

**Expected Output**:
- Healthy state: `{"status": "healthy", "composio_configured": true, "openai_configured": true}`
- Degraded state: `{"status": "degraded", "composio_configured": false, "openai_configured": false}`

---

## 8. Test Graceful Shutdown

**Status**: [ ] Not Started
**Priority**: 🟡 NICE TO HAVE

**Description**: Verify that the container handles shutdown signals gracefully and stops within acceptable time.

**Requirements**: AC 3.4.5

**Dependencies**: Task 5

**Implementation Details**:
- Start container
- Send SIGTERM signal via `docker stop`
- Measure shutdown time
- Verify cleanup logs appear
- Ensure container stops within 10 seconds

**Acceptance Criteria**:
- [ ] Container responds to SIGTERM
- [ ] Shutdown completes within 10 seconds
- [ ] Logs show "API shutdown complete"
- [ ] No forced kill required
- [ ] Cleanup hooks execute successfully

**Testing**:
```bash
# Start container
docker run -d --name spa-shutdown-test \
  -e COMPOSIO_API_KEY=test_key \
  -e OPENAI_API_KEY=test_key \
  -p 8000:8000 \
  spa-agent:test

sleep 10

# Stop with timing
time docker stop spa-shutdown-test

# Check logs for shutdown message
docker logs spa-shutdown-test | grep "API shutdown complete"

# Cleanup
docker rm spa-shutdown-test
```

**Expected Output**:
- Stop time: < 10 seconds
- Logs contain "API shutdown complete"
- Exit code: 0

---

## 9. Update README with Docker Instructions

**Status**: [x] Completed
**Priority**: 🔴 MUST HAVE (MVP)

**Description**: Add comprehensive Docker documentation to the README, including build, run, and usage instructions.

**Requirements**: AC 3.5.1, 3.5.2, 3.5.3, 3.5.4, 3.5.5

**Dependencies**: Tasks 1, 2, 3

**Implementation Details**:
- Add new section "Run with Docker (production)" after "Running the Application"
- Include build command
- Include run command with environment variables
- Document health endpoint access
- Document API endpoint access
- Mention CORS_ORIGINS configuration

**File Location**: `stock-portfolio-analysis-agent/README.md`

**Content to Add**:
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

**Acceptance Criteria**:
- [ ] New "Run with Docker" section added to README
- [ ] Build command documented
- [ ] Run command with environment variables documented
- [ ] Health endpoint URL documented
- [ ] API endpoint URLs documented
- [ ] CORS_ORIGINS configuration mentioned
- [ ] Section placed after "Running the Application"

**Testing**:
```bash
# Verify section exists
grep -A 20 "Run with Docker" stock-portfolio-analysis-agent/README.md

# Verify key elements
grep "docker build" stock-portfolio-analysis-agent/README.md
grep "docker run" stock-portfolio-analysis-agent/README.md
grep "COMPOSIO_API_KEY" stock-portfolio-analysis-agent/README.md
grep "CORS_ORIGINS" stock-portfolio-analysis-agent/README.md
```

---

## 10. Run Regression Tests

**Status**: [ ] Not Started
**Priority**: 🟡 NICE TO HAVE

**Description**: Verify that all existing tests still pass and no functionality has been broken by the Docker implementation.

**Requirements**: Success Metric - All 65 existing tests pass

**Dependencies**: Tasks 1, 2, 3, 9

**Implementation Details**:
- Run full test suite outside of Docker (baseline)
- Verify all 65 tests pass
- Ensure no new test failures
- Check for any warnings or deprecations

**Acceptance Criteria**:
- [ ] All 65 tests pass
- [ ] No new test failures introduced
- [ ] No regression in existing functionality
- [ ] Test execution time is similar to baseline

**Testing**:
```bash
cd stock-portfolio-analysis-agent

# Run full test suite
uv run pytest tests/ -v

# Check test count
uv run pytest tests/ --collect-only | grep "test session starts"

# Run with coverage (optional)
uv run pytest tests/ --cov=agent --cov-report=term
```

**Expected Output**:
- 65 tests passed
- 0 tests failed
- Exit code: 0

---

## 11. Integration Test with Real APIs (Manual)

**Status**: [ ] Not Started
**Priority**: 🟡 NICE TO HAVE

**Description**: Manually test the containerized application with real API keys to verify end-to-end functionality.

**Requirements**: Integration Testing Strategy

**Dependencies**: Task 5

**Implementation Details**:
- Start container with real COMPOSIO_API_KEY and OPENAI_API_KEY
- Test health endpoint
- Test analyze endpoint with sample query
- Verify response is valid
- Check logs for any errors

**Acceptance Criteria**:
- [ ] Container starts with real API keys
- [ ] Health endpoint returns "healthy" status
- [ ] Analyze endpoint accepts queries
- [ ] Valid portfolio analysis is returned
- [ ] No errors in logs

**Testing**:
```bash
# Start with real keys
docker run -d --name spa-integration \
  -e COMPOSIO_API_KEY=$COMPOSIO_API_KEY \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -p 8000:8000 \
  spa-agent:test

sleep 10

# Test health
curl http://localhost:8000/health

# Test analyze endpoint
curl -X POST http://localhost:8000/analyze/sync \
  -H "Content-Type: application/json" \
  -d '{"query": "What if I invested $10k in AAPL since 2020?"}'

# Check logs
docker logs spa-integration

# Cleanup
docker stop spa-integration
docker rm spa-integration
```

**Expected Output**:
- Health status: healthy
- Valid portfolio analysis response
- No errors in logs

---

## 12. Final Verification and Cleanup

**Status**: [ ] Not Started
**Priority**: 🟡 NICE TO HAVE

**Description**: Perform final verification of all deliverables and clean up test artifacts.

**Requirements**: All acceptance criteria

**Dependencies**: Tasks 1, 2, 3, 5, 9

**Implementation Details**:
- Verify all files are created
- Verify all tests pass
- Clean up test Docker images and containers
- Review documentation completeness
- Tag final image (optional)

**Acceptance Criteria**:
- [ ] Dockerfile exists and is correct
- [ ] .dockerignore exists and is correct
- [ ] README is updated
- [ ] All tests pass
- [ ] Docker image builds successfully
- [ ] Container runs successfully
- [ ] No test artifacts remain

**Testing**:
```bash
# Verify files exist
test -f stock-portfolio-analysis-agent/Dockerfile
test -f stock-portfolio-analysis-agent/.dockerignore
grep "Run with Docker" stock-portfolio-analysis-agent/README.md

# Clean up test containers
docker ps -a | grep spa- | awk '{print $1}' | xargs -r docker rm -f

# Clean up test images (optional - keep if needed)
# docker rmi spa-agent:test

# Final build test
cd stock-portfolio-analysis-agent
docker build -t spa-agent:latest .

# Final run test
docker run -d --name spa-final \
  -e COMPOSIO_API_KEY=test \
  -e OPENAI_API_KEY=test \
  -p 8000:8000 \
  spa-agent:latest

sleep 10
curl http://localhost:8000/health

docker stop spa-final
docker rm spa-final
```

**Expected Output**:
- All files present
- Build succeeds
- Container runs
- Health check passes

---

## Task Summary

| Task | Description | Priority | Dependencies | Estimated Time |
|------|-------------|----------|--------------|----------------|
| 1 | Create Dockerfile | 🔴 MUST HAVE | None | 15 min |
| 2 | Create .dockerignore | 🔴 MUST HAVE | None | 10 min |
| 3 | Build Docker Image | 🔴 MUST HAVE | 1, 2 | 5 min |
| 4 | Verify Dependencies | 🟡 NICE TO HAVE | 3 | 5 min |
| 5 | Test Container Startup | 🔴 MUST HAVE | 3 | 10 min |
| 6 | Test Environment Variables | 🟡 NICE TO HAVE | 5 | 10 min |
| 7 | Test Health Endpoint | 🟡 NICE TO HAVE | 5 | 10 min |
| 8 | Test Graceful Shutdown | 🟡 NICE TO HAVE | 5 | 5 min |
| 9 | Update README | 🔴 MUST HAVE | 1, 2, 3 | 10 min |
| 10 | Run Regression Tests | 🟡 NICE TO HAVE | 1, 2, 3, 9 | 5 min |
| 11 | Integration Test (Manual) | 🟡 NICE TO HAVE | 5 | 10 min |
| 12 | Final Verification | 🟡 NICE TO HAVE | 1, 2, 3, 5, 9 | 10 min |

**MVP Estimated Time**: ~40 minutes (Tasks 1, 2, 3, 5, 9)
**Total Estimated Time**: ~105 minutes (~1.75 hours)

---

## Success Criteria

### MVP Success Criteria (MUST HAVE)

These are the minimum requirements for a working Docker implementation:

- ✅ Dockerfile created and builds successfully (Task 1)
- ✅ .dockerignore created with proper exclusions (Task 2)
- ✅ Docker image builds without errors (Task 3)
- ✅ Container starts and serves requests (Task 5)
- ✅ README updated with Docker instructions (Task 9)

### Full Success Criteria (NICE TO HAVE)

Additional validation for production readiness:

- ✅ All dependencies verified in container (Task 4)
- ✅ Environment variables work correctly (Task 6)
- ✅ Health endpoint responds correctly (Task 7)
- ✅ Graceful shutdown works (Task 8)
- ✅ All 65 existing tests pass (Task 10)
- ✅ Integration test with real APIs succeeds (Task 11)
- ✅ Final verification complete (Task 12)

---

## Notes

### MVP Implementation Path (Fast Track)

For fastest MVP delivery, complete only the MUST HAVE tasks in this order:
1. Task 1: Create Dockerfile (15 min)
2. Task 2: Create .dockerignore (10 min)
3. Task 3: Build Docker Image (5 min)
4. Task 5: Test Container Startup (10 min)
5. Task 9: Update README (10 min)

**Total MVP Time: ~40 minutes**

This gives you a working Docker setup that can be built, run, and documented.

### Full Implementation Path

For production-ready implementation, complete all tasks:
- Tasks 1-2 can be done in parallel
- Tasks 3-5 must be done sequentially (MVP core)
- Tasks 6-8 add validation (can be done after MVP)
- Task 9 documents the implementation (MVP core)
- Tasks 10-12 add comprehensive testing (post-MVP)

### Additional Notes

- Task 11 requires real API keys (manual step)
- Keep test images/containers until final verification is complete
- NICE TO HAVE tasks can be completed incrementally after MVP is working
