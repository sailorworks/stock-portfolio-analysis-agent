"""Main entry point for the Stock Portfolio Analysis Agent API.

This module provides the uvicorn server configuration and startup hooks.

Requirements: 10.1 - Expose a POST endpoint that accepts user queries
Requirements: 9.1, 9.2, 9.3, 9.4 - Tool Router integration
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def configure_logging() -> logging.Logger:
    """Configure logging for the application.
    
    Returns:
        Configured logger instance
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    # Set specific log levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


# Configure logging
logger = configure_logging()


def validate_environment() -> bool:
    """Validate required environment variables are set.
    
    Returns:
        True if all required variables are set, False otherwise
    """
    required_vars = ["COMPOSIO_API_KEY", "OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(
            f"Missing environment variables: {', '.join(missing_vars)}. "
            "Some features may not work correctly."
        )
        return False
    
    return True


def initialize_components() -> None:
    """Initialize all application components on startup.
    
    Requirements: 9.1, 9.2, 9.3 - Initialize Composio and register tools
    
    This function:
    - Validates environment variables
    - Initializes the session manager
    - Registers custom tools with Composio
    - Initializes the agent orchestrator
    """
    logger.info("Initializing application components...")
    
    # Validate environment
    validate_environment()
    
    # Initialize session manager (this also registers tools with Composio)
    try:
        from agent.session import get_session_manager
        session_manager = get_session_manager()
        logger.info("Session manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize session manager: {e}")
        raise
    
    # Initialize agent orchestrator
    try:
        from agent.agent_config import get_orchestrator
        orchestrator = get_orchestrator()
        logger.info("Agent orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent orchestrator: {e}")
        raise
    
    logger.info("All components initialized successfully")


def cleanup_components() -> None:
    """Cleanup application components on shutdown.
    
    This function performs graceful cleanup of:
    - Session manager resources
    - Any open connections
    """
    logger.info("Cleaning up application components...")
    
    # Cleanup is handled by Python's garbage collector for most resources
    # Add specific cleanup logic here if needed
    
    logger.info("Cleanup complete")


@asynccontextmanager
async def lifespan(app) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI application.
    
    This handles startup and shutdown events for the application.
    
    Requirements: 9.1, 9.2, 9.3, 9.4 - Initialize Tool Router on startup
    
    Args:
        app: The FastAPI application instance
        
    Yields:
        None
    """
    # Startup
    logger.info("Starting Stock Portfolio Analysis Agent API...")
    initialize_components()
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Stock Portfolio Analysis Agent API...")
    cleanup_components()
    logger.info("API shutdown complete")


def create_configured_app():
    """Create and configure the FastAPI application with lifespan.
    
    Returns:
        Configured FastAPI application
    """
    # Import the app instance that has all endpoints registered
    from agent.api import app
    
    # Add startup/shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Handle application startup."""
        logger.info("Starting Stock Portfolio Analysis Agent API...")
        initialize_components()
        logger.info("API startup complete")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Handle application shutdown."""
        logger.info("Shutting down Stock Portfolio Analysis Agent API...")
        cleanup_components()
        logger.info("API shutdown complete")
    
    return app


def main():
    """Run the FastAPI server with uvicorn.
    
    Requirements: 10.1 - Expose a POST endpoint that accepts user queries
    """
    import uvicorn
    
    # Get configuration from environment
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("RELOAD", "false").lower() == "true"
    workers = int(os.environ.get("WORKERS", "1"))
    
    logger.info(f"Starting Stock Portfolio Analysis Agent API on {host}:{port}")
    logger.info(f"Reload mode: {reload}, Workers: {workers}")
    
    # Use the app factory pattern for proper initialization
    uvicorn.run(
        "main:create_configured_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,  # Workers not supported with reload
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
