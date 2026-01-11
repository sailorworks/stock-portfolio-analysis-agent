"""Error handling utilities for the Stock Portfolio Analysis Agent.

This module provides structured error handling for tools and API endpoints,
including custom exception classes and error response models.

Requirements: 4.4, 10.1
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Error Types
# =============================================================================


class ErrorCode(str, Enum):
    """Enumeration of error codes for structured error responses."""
    
    # Tool errors (4xx range conceptually)
    INVALID_TICKER = "INVALID_TICKER"
    INVALID_DATE_FORMAT = "INVALID_DATE_FORMAT"
    INVALID_DATE_RANGE = "INVALID_DATE_RANGE"
    INVALID_INTERVAL = "INVALID_INTERVAL"
    INVALID_STRATEGY = "INVALID_STRATEGY"
    NO_DATA_AVAILABLE = "NO_DATA_AVAILABLE"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    EMPTY_INPUT = "EMPTY_INPUT"
    
    # External service errors (5xx range conceptually)
    YFINANCE_API_ERROR = "YFINANCE_API_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    
    # General errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


# =============================================================================
# Error Response Models
# =============================================================================


class ToolError(BaseModel):
    """Structured error response for tool execution failures.
    
    Requirements: 4.4 - Return structured error responses
    """
    error_code: ErrorCode = Field(
        ...,
        description="Machine-readable error code"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details for debugging"
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="Name of the tool that raised the error"
    )


# =============================================================================
# Custom Exceptions
# =============================================================================


class ToolExecutionError(Exception):
    """Base exception for tool execution errors.
    
    This exception wraps tool errors with structured information
    for consistent error handling across all tools.
    """
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
    ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.tool_name = tool_name
        super().__init__(message)
    
    def to_error_response(self) -> ToolError:
        """Convert exception to a ToolError response model."""
        return ToolError(
            error_code=self.error_code,
            message=self.message,
            details=self.details,
            tool_name=self.tool_name,
        )


class InvalidTickerError(ToolExecutionError):
    """Raised when an invalid ticker symbol is provided.
    
    Requirements: 4.4 - Invalid ticker error handling
    """
    
    def __init__(self, tickers: list[str], message: Optional[str] = None):
        super().__init__(
            error_code=ErrorCode.INVALID_TICKER,
            message=message or f"Invalid ticker(s): {', '.join(tickers)}",
            details={"invalid_tickers": tickers},
        )


class InvalidDateError(ToolExecutionError):
    """Raised when date format or range is invalid."""
    
    def __init__(
        self,
        error_type: str = "format",
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        error_code = (
            ErrorCode.INVALID_DATE_FORMAT
            if error_type == "format"
            else ErrorCode.INVALID_DATE_RANGE
        )
        super().__init__(
            error_code=error_code,
            message=message or "Invalid date provided",
            details=details,
        )


class NoDataAvailableError(ToolExecutionError):
    """Raised when no data is available for the requested parameters."""
    
    def __init__(
        self,
        tickers: Optional[list[str]] = None,
        date_range: Optional[tuple[str, str]] = None,
        message: Optional[str] = None,
    ):
        details = {}
        if tickers:
            details["tickers"] = tickers
        if date_range:
            details["start_date"] = date_range[0]
            details["end_date"] = date_range[1]
        
        super().__init__(
            error_code=ErrorCode.NO_DATA_AVAILABLE,
            message=message or "No data available for the specified parameters",
            details=details,
        )


class YFinanceAPIError(ToolExecutionError):
    """Raised when Yahoo Finance API fails."""
    
    def __init__(self, original_error: Exception, message: Optional[str] = None):
        super().__init__(
            error_code=ErrorCode.YFINANCE_API_ERROR,
            message=message or f"Yahoo Finance API error: {str(original_error)}",
            details={"original_error": str(original_error)},
        )


class ValidationError(ToolExecutionError):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, message: str, value: Any = None):
        details = {"field": field}
        if value is not None:
            details["provided_value"] = str(value)
        
        super().__init__(
            error_code=ErrorCode.VALIDATION_ERROR,
            message=message,
            details=details,
        )


# =============================================================================
# Error Handling Utilities
# =============================================================================


def log_tool_error(
    tool_name: str,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a tool error with context for debugging.
    
    Requirements: 4.4 - Log errors for debugging
    
    Args:
        tool_name: Name of the tool that raised the error
        error: The exception that was raised
        context: Additional context information
    """
    error_info = {
        "tool_name": tool_name,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    
    if context:
        error_info["context"] = context
    
    if isinstance(error, ToolExecutionError):
        error_info["error_code"] = error.error_code.value
        error_info["details"] = error.details
        logger.error(f"Tool execution error: {error_info}")
    else:
        logger.exception(f"Unexpected error in tool {tool_name}: {error_info}")


def wrap_tool_error(
    error: Exception,
    tool_name: str,
) -> ToolExecutionError:
    """Wrap any exception in a ToolExecutionError.
    
    This ensures all errors from tools are consistently structured.
    
    Args:
        error: The original exception
        tool_name: Name of the tool that raised the error
        
    Returns:
        A ToolExecutionError wrapping the original error
    """
    if isinstance(error, ToolExecutionError):
        error.tool_name = tool_name
        return error
    
    # Handle common exception types
    if isinstance(error, ValueError):
        return ToolExecutionError(
            error_code=ErrorCode.VALIDATION_ERROR,
            message=str(error),
            tool_name=tool_name,
        )
    
    # Default to internal error for unknown exceptions
    return ToolExecutionError(
        error_code=ErrorCode.INTERNAL_ERROR,
        message=f"Internal error: {str(error)}",
        details={"original_error_type": type(error).__name__},
        tool_name=tool_name,
    )
