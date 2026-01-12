"""FastAPI application for the Stock Portfolio Analysis Agent.

This module provides the REST API endpoint for portfolio analysis including:
- FastAPI app with CORS configuration
- Request/response models
- SSE streaming endpoint for real-time progress updates
- Exception handlers for proper error responses

Requirements: 3.1, 3.2, 3.3, 3.4, 10.1, 10.2, 10.3, 10.4
"""

import asyncio
import json
import logging
import re
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from agents import Runner

from agent.models import (
    InvestmentSummary,
    Insights,
    PerformancePoint,
)
from agent.portfolio import Portfolio, PortfolioHolding, get_portfolio_manager
from agent.agent_config import get_orchestrator, create_portfolio_agent
from agent.session import get_session_manager
from agent.tools import (
    fetch_stock_data,
    fetch_benchmark_data,
    simulate_portfolio,
    simulate_spy_investment,
    calculate_metrics,
)
from agent.models import (
    FetchStockDataInput,
    FetchBenchmarkInput,
    SimulatePortfolioInput,
    SimulateSPYInput,
    CalculateMetricsInput,
)
from agent.errors import (
    ErrorCode,
    ToolError,
    ToolExecutionError,
    log_tool_error,
)
from agent.insights import generate_insights_from_summary


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================


class PortfolioHoldingRequest(BaseModel):
    """A single holding in the portfolio request."""
    ticker_symbol: str = Field(
        ...,
        description="Stock ticker symbol (e.g., 'AAPL')"
    )
    investment_amount: float = Field(
        ...,
        gt=0,
        description="Amount invested in this holding"
    )
    purchase_date: str = Field(
        ...,
        description="Date of purchase in YYYY-MM-DD format"
    )


class PortfolioRequest(BaseModel):
    """Portfolio state in the request."""
    holdings: List[PortfolioHoldingRequest] = Field(
        default_factory=list,
        description="List of portfolio holdings"
    )


class AnalyzeRequest(BaseModel):
    """Request model for the /analyze endpoint.
    
    Requirements: 10.1 - Accept user queries and portfolio state
    """
    query: str = Field(
        ...,
        description="Natural language investment query"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user ID for session management"
    )
    portfolio: Optional[PortfolioRequest] = Field(
        default=None,
        description="Optional portfolio state"
    )


class SSEEvent(BaseModel):
    """Server-Sent Event model.
    
    Requirements: 10.2, 10.3 - Stream progress events using SSE
    """
    event: str = Field(
        ...,
        description="Event type: 'status', 'tool_log', 'portfolio_update', 'result', 'error'"
    )
    data: Any = Field(
        ...,
        description="Event data payload"
    )


class AnalyzeResponse(BaseModel):
    """Response model for the /analyze endpoint.
    
    Requirements: 10.4 - Return investment summary with all calculated metrics
    """
    success: bool = Field(
        ...,
        description="Whether the analysis was successful"
    )
    summary: Optional[InvestmentSummary] = Field(
        default=None,
        description="Investment summary with all calculated metrics"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if analysis failed"
    )


class APIErrorResponse(BaseModel):
    """Structured error response for API errors.
    
    Requirements: 10.1 - Return appropriate HTTP status codes with error details
    """
    error_code: str = Field(
        ...,
        description="Machine-readable error code"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking"
    )


# =============================================================================
# FastAPI Application Setup
# =============================================================================


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Requirements: 10.1 - Expose a POST endpoint that accepts user queries
    
    Returns:
        Configured FastAPI application with exception handlers
    """
    app = FastAPI(
        title="Stock Portfolio Analysis Agent",
        description="AI-powered stock portfolio analysis using Composio Tool Router",
        version="0.1.0",
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ==========================================================================
    # Exception Handlers - Requirements: 10.1
    # ==========================================================================
    
    @app.exception_handler(ToolExecutionError)
    async def tool_execution_error_handler(
        request: Request,
        exc: ToolExecutionError,
    ) -> JSONResponse:
        """Handle tool execution errors with structured responses.
        
        Requirements: 10.1 - Return appropriate HTTP status codes
        """
        request_id = getattr(request.state, "request_id", None)
        
        # Log the error
        logger.error(
            f"Tool execution error: {exc.error_code.value} - {exc.message}",
            extra={
                "request_id": request_id,
                "error_code": exc.error_code.value,
                "tool_name": exc.tool_name,
                "details": exc.details,
            },
        )
        
        # Map error codes to HTTP status codes
        status_code_map = {
            ErrorCode.INVALID_TICKER: 400,
            ErrorCode.INVALID_DATE_FORMAT: 400,
            ErrorCode.INVALID_DATE_RANGE: 400,
            ErrorCode.INVALID_INTERVAL: 400,
            ErrorCode.INVALID_STRATEGY: 400,
            ErrorCode.EMPTY_INPUT: 400,
            ErrorCode.VALIDATION_ERROR: 400,
            ErrorCode.NO_DATA_AVAILABLE: 404,
            ErrorCode.INSUFFICIENT_FUNDS: 422,
            ErrorCode.YFINANCE_API_ERROR: 502,
            ErrorCode.NETWORK_ERROR: 502,
            ErrorCode.INTERNAL_ERROR: 500,
            ErrorCode.UNKNOWN_ERROR: 500,
        }
        
        status_code = status_code_map.get(exc.error_code, 500)
        
        return JSONResponse(
            status_code=status_code,
            content=APIErrorResponse(
                error_code=exc.error_code.value,
                message=exc.message,
                details=exc.details,
                request_id=request_id,
            ).model_dump(),
        )
    
    @app.exception_handler(PydanticValidationError)
    async def pydantic_validation_error_handler(
        request: Request,
        exc: PydanticValidationError,
    ) -> JSONResponse:
        """Handle Pydantic validation errors.
        
        Requirements: 10.1 - Return appropriate HTTP status codes
        """
        request_id = getattr(request.state, "request_id", None)
        
        # Extract validation error details
        errors = exc.errors()
        error_details = [
            {
                "field": ".".join(str(loc) for loc in err.get("loc", [])),
                "message": err.get("msg", "Validation error"),
                "type": err.get("type", "unknown"),
            }
            for err in errors
        ]
        
        logger.warning(
            f"Validation error: {len(errors)} error(s)",
            extra={
                "request_id": request_id,
                "errors": error_details,
            },
        )
        
        return JSONResponse(
            status_code=422,
            content=APIErrorResponse(
                error_code="VALIDATION_ERROR",
                message="Request validation failed",
                details={"validation_errors": error_details},
                request_id=request_id,
            ).model_dump(),
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        """Handle HTTP exceptions with structured responses.
        
        Requirements: 10.1 - Return appropriate HTTP status codes
        """
        request_id = getattr(request.state, "request_id", None)
        
        logger.warning(
            f"HTTP exception: {exc.status_code} - {exc.detail}",
            extra={"request_id": request_id},
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=APIErrorResponse(
                error_code=f"HTTP_{exc.status_code}",
                message=str(exc.detail),
                request_id=request_id,
            ).model_dump(),
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle unexpected exceptions with structured responses.
        
        Requirements: 10.1 - Return appropriate HTTP status codes
        """
        request_id = getattr(request.state, "request_id", None)
        
        # Log the full exception for debugging
        logger.exception(
            f"Unexpected error: {type(exc).__name__} - {str(exc)}",
            extra={"request_id": request_id},
        )
        
        return JSONResponse(
            status_code=500,
            content=APIErrorResponse(
                error_code="INTERNAL_ERROR",
                message="An unexpected error occurred. Please try again later.",
                details={"error_type": type(exc).__name__} if logger.isEnabledFor(logging.DEBUG) else None,
                request_id=request_id,
            ).model_dump(),
        )
    
    # ==========================================================================
    # Middleware for request tracking
    # ==========================================================================
    
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """Add a unique request ID to each request for tracking."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response
    
    return app


# Create the FastAPI app instance
app = create_app()


# =============================================================================
# Helper Functions
# =============================================================================


def format_sse_event(event_type: str, data: Any) -> str:
    """Format data as a Server-Sent Event.
    
    Requirements: 10.2 - Stream progress events using SSE
    
    Args:
        event_type: The type of event (status, tool_log, result, error)
        data: The event data payload
        
    Returns:
        Formatted SSE string
    """
    if isinstance(data, BaseModel):
        json_data = data.model_dump_json()
    elif isinstance(data, dict):
        json_data = json.dumps(data)
    else:
        json_data = json.dumps({"message": str(data)})
    
    return f"event: {event_type}\ndata: {json_data}\n\n"


def is_tool_error(result: Any) -> bool:
    """Check if a tool result is an error response.
    
    Args:
        result: The result from a tool execution
        
    Returns:
        True if the result is a ToolError, False otherwise
    """
    return isinstance(result, ToolError)


def format_tool_error_message(error: ToolError) -> str:
    """Format a ToolError into a user-friendly message.
    
    Args:
        error: The ToolError to format
        
    Returns:
        Formatted error message string
    """
    message = error.message
    if error.details:
        # Add relevant details to the message
        if "invalid_tickers" in error.details:
            message += f" (tickers: {', '.join(error.details['invalid_tickers'])})"
        elif "field" in error.details:
            message += f" (field: {error.details['field']})"
    return message


def parse_query_simple(query: str) -> Dict[str, Any]:
    """Simple query parser to extract investment parameters.
    
    This is a simplified parser for demonstration. In production,
    the LLM agent would handle query parsing.
    
    Args:
        query: Natural language investment query
        
    Returns:
        Dict with extracted parameters
    """
    from datetime import datetime, timedelta
    import re
    
    # Default values
    params = {
        "ticker_symbols": [],
        "investment_amounts": {},
        "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "strategy": "single_shot",
        "dca_interval": None,
        "total_cash": 10000.0,
    }
    
    # Common ticker mappings
    ticker_map = {
        "apple": "AAPL",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "tesla": "TSLA",
        "microsoft": "MSFT",
        "amazon": "AMZN",
        "meta": "META",
        "facebook": "META",
        "netflix": "NFLX",
        "nvidia": "NVDA",
    }
    
    query_lower = query.lower()
    
    # Extract tickers
    # First, check for explicit ticker symbols (uppercase letters)
    explicit_tickers = re.findall(r'\b([A-Z]{2,5})\b', query)
    for ticker in explicit_tickers:
        if ticker not in ["DCA", "USD", "ETF", "SPY"]:
            params["ticker_symbols"].append(ticker)
    
    # Then check for company names
    for name, ticker in ticker_map.items():
        if name in query_lower and ticker not in params["ticker_symbols"]:
            params["ticker_symbols"].append(ticker)
    
    # Extract amount
    amount_patterns = [
        r'\$?([\d,]+)k\b',  # 10k, $10k
        r'\$?([\d,]+)\s*(?:dollars?|usd)',  # 10000 dollars
        r'\$([\d,]+)',  # $10000
    ]
    
    for pattern in amount_patterns:
        match = re.search(pattern, query_lower)
        if match:
            amount_str = match.group(1).replace(",", "")
            amount = float(amount_str)
            if "k" in query_lower[match.start():match.end()+2]:
                amount *= 1000
            params["total_cash"] = amount
            break
    
    # Extract dates
    # Look for year patterns
    year_match = re.search(r'(?:since|from|in)\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)?\s*(\d{4})', query_lower)
    if year_match:
        year = int(year_match.group(1))
        # Check for month
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        month = 1
        for m_name, m_num in month_map.items():
            if m_name in query_lower:
                month = m_num
                break
        params["start_date"] = f"{year}-{month:02d}-01"
    
    # Check for DCA strategy
    dca_indicators = ["monthly", "quarterly", "weekly", "dca", "dollar cost", "spread"]
    for indicator in dca_indicators:
        if indicator in query_lower:
            params["strategy"] = "dca"
            if "monthly" in query_lower:
                params["dca_interval"] = "monthly"
            elif "quarterly" in query_lower:
                params["dca_interval"] = "quarterly"
            elif "weekly" in query_lower:
                params["dca_interval"] = "weekly"
            else:
                params["dca_interval"] = "monthly"  # Default
            break
    
    # Distribute amount among tickers
    if params["ticker_symbols"]:
        amount_per_ticker = params["total_cash"] / len(params["ticker_symbols"])
        for ticker in params["ticker_symbols"]:
            params["investment_amounts"][ticker] = amount_per_ticker
    
    return params


async def run_agent_analysis(
    query: str,
    user_id: str,
    portfolio_holdings: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the portfolio analysis agent with the given query.
    
    Requirements: 3.1, 3.2 - Use Runner.run(agent, query) for analysis
    
    This function:
    1. Gets or creates a portfolio agent for the user
    2. Runs the agent with the user's query using Runner.run
    3. Extracts tool call results from result.new_items
    
    Args:
        query: Natural language investment query
        user_id: User ID for session management
        portfolio_holdings: Optional list of existing portfolio holdings
        
    Returns:
        Dict containing the analysis results or error information
    """
    try:
        # Get the orchestrator and create an agent
        orchestrator = get_orchestrator()
        
        # Create portfolio context if holdings provided
        portfolio = None
        if portfolio_holdings:
            from agent.portfolio import Portfolio, PortfolioHolding
            holdings = [
                PortfolioHolding(
                    ticker_symbol=h.get("ticker_symbol", ""),
                    investment_amount=h.get("investment_amount", 0),
                    purchase_date=h.get("purchase_date", ""),
                )
                for h in portfolio_holdings
            ]
            portfolio = Portfolio(user_id=user_id, holdings=holdings)
        
        # Get the portfolio agent with tools from Composio session
        agent = orchestrator.get_portfolio_agent(user_id=user_id, portfolio=portfolio)
        
        # Run the agent with the user's query
        # Requirements: 3.1, 3.2 - Use Runner.run instead of direct tool calls
        result = await Runner.run(agent, query)
        
        # Parse the agent's final output
        final_output = result.final_output if hasattr(result, 'final_output') else str(result)
        
        # Extract tool call results from result.new_items
        tool_results = {}
        tool_calls_made = []
        
        # Track tool calls and their outputs
        pending_tool_calls = {}  # call_id -> tool_name
        
        if hasattr(result, 'new_items'):
            for item in result.new_items:
                item_type = type(item).__name__
                
                if item_type == 'ToolCallItem':
                    # Extract tool name from raw_item
                    raw_item = getattr(item, 'raw_item', None)
                    if raw_item:
                        # raw_item is typically a ResponseFunctionToolCall
                        tool_name = getattr(raw_item, 'name', None)
                        call_id = getattr(raw_item, 'call_id', None)
                        if tool_name:
                            tool_calls_made.append(tool_name)
                            if call_id:
                                pending_tool_calls[call_id] = tool_name
                                logger.debug(f"Registered tool call: {tool_name} with call_id: {call_id}")
                            logger.info(f"Tool called via Composio: {tool_name}")
                
                elif item_type == 'ToolCallOutputItem':
                    # Extract output and match to tool call
                    output = getattr(item, 'output', None)
                    raw_item = getattr(item, 'raw_item', None)
                    
                    if output:
                        # Try to get call_id from raw_item
                        call_id = None
                        if raw_item:
                            if isinstance(raw_item, dict):
                                # raw_item is a dict
                                call_id = raw_item.get('call_id') or raw_item.get('tool_call_id') or raw_item.get('id')
                            else:
                                # raw_item is an object
                                for attr in ['call_id', 'tool_call_id', 'id']:
                                    if hasattr(raw_item, attr):
                                        call_id = getattr(raw_item, attr)
                                        if call_id:
                                            break
                        
                        tool_name = pending_tool_calls.get(call_id) if call_id else None
                        
                        if tool_name:
                            # Store the latest output for each tool
                            tool_results[tool_name] = output
                            logger.info(f"Got output for tool: {tool_name}, output type: {type(output)}")
                        else:
                            # If we can't match by call_id, try to infer from output structure
                            # This is a fallback - try to identify the tool from the output
                            if isinstance(output, str):
                                try:
                                    output_data = json.loads(output)
                                except:
                                    output_data = {}
                            else:
                                output_data = output if isinstance(output, dict) else {}
                            
                            # Infer tool name from output structure
                            inferred_tool = None
                            if 'prices' in output_data and 'metadata' in output_data:
                                inferred_tool = 'FETCH_STOCK_DATA'
                            elif 'spy_prices' in output_data:
                                inferred_tool = 'FETCH_BENCHMARK_DATA'
                            elif 'holdings' in output_data and 'transaction_log' in output_data:
                                inferred_tool = 'SIMULATE_PORTFOLIO'
                            elif 'spy_shares' in output_data:
                                inferred_tool = 'SIMULATE_SPY_INVESTMENT'
                            elif 'total_value' in output_data and 'performance_data' in output_data:
                                inferred_tool = 'CALCULATE_METRICS'
                            
                            if inferred_tool:
                                tool_results[inferred_tool] = output
                                logger.info(f"Inferred tool from output structure: {inferred_tool}")
        
        # Log what tools were called
        if tool_calls_made:
            logger.info(f"Agent called {len(tool_calls_made)} tools via Composio: {list(set(tool_calls_made))}")
        else:
            logger.warning("Agent did not call any tools - will need fallback")
        
        return {
            "success": True,
            "output": final_output,
            "result": result,
            "tool_results": tool_results,
            "tool_calls_made": tool_calls_made,
        }
        
    except Exception as e:
        logger.exception(f"Error running agent analysis: {str(e)}")
        return {
            "success": False,
            "error": str(e),
        }


def build_summary_from_tool_results(
    tool_results: Dict[str, Any],
) -> Optional[InvestmentSummary]:
    """Build an InvestmentSummary from Composio tool call results.
    
    This function extracts data from the tool results returned by the agent
    when it successfully calls tools via Composio.
    
    Args:
        tool_results: Dict mapping tool names to their outputs
        
    Returns:
        InvestmentSummary if we have enough data, None otherwise
    """
    try:
        # Normalize tool names to uppercase for matching
        normalized_results = {}
        for k, v in tool_results.items():
            logger.info(f"Processing tool result for {k}: type={type(v)}")
            
            # Parse the result - it might be a string (JSON) or already a dict
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                    logger.info(f"Parsed JSON for {k}: keys={list(v.keys()) if isinstance(v, dict) else 'not a dict'}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON for {k}: {e}")
                    continue
            
            # Handle wrapped output format: {'successful': True, 'error': None, 'data': {...}}
            if isinstance(v, dict) and 'data' in v and 'successful' in v:
                logger.info(f"Unwrapping data for {k}: successful={v.get('successful')}, error={v.get('error')}, data_type={type(v.get('data'))}")
                if v.get('successful') and v.get('data'):
                    v = v.get('data', {})
                    if v:
                        logger.info(f"Unwrapped data for {k}: keys={list(v.keys()) if isinstance(v, dict) else 'not a dict'}")
                else:
                    # Tool execution failed - log the error and skip
                    logger.warning(f"Tool {k} failed: error={v.get('error')}")
                    continue
            
            normalized_results[k.upper()] = v
        
        # We need at least simulate_portfolio and calculate_metrics results
        simulation_result = normalized_results.get('SIMULATE_PORTFOLIO')
        metrics_result = normalized_results.get('CALCULATE_METRICS')
        stock_data_result = normalized_results.get('FETCH_STOCK_DATA')
        benchmark_result = normalized_results.get('FETCH_BENCHMARK_DATA')
        spy_simulation_result = normalized_results.get('SIMULATE_SPY_INVESTMENT')
        
        if simulation_result:
            logger.info(f"Found simulation result: {list(simulation_result.keys()) if isinstance(simulation_result, dict) else type(simulation_result)}")
        if metrics_result:
            logger.info(f"Found metrics result: {list(metrics_result.keys()) if isinstance(metrics_result, dict) else type(metrics_result)}")
        if stock_data_result:
            logger.info(f"Found stock data result: {list(stock_data_result.keys()) if isinstance(stock_data_result, dict) else type(stock_data_result)}")
        
        # If we don't have metrics, try to build from simulation + stock data
        if not metrics_result and simulation_result and stock_data_result:
            logger.info("No CALCULATE_METRICS result, building summary from simulation and stock data")
            
            holdings = simulation_result.get('holdings', {})
            remaining_cash = simulation_result.get('remaining_cash', 0.0)
            
            # Get current prices from stock data
            current_prices = {}
            prices_data = stock_data_result.get('prices', {})
            for ticker, prices in prices_data.items():
                if prices:
                    latest_date = max(prices.keys())
                    current_prices[ticker] = prices[latest_date]
            
            # Calculate total value
            total_value = remaining_cash
            for ticker, shares in holdings.items():
                if ticker in current_prices:
                    total_value += shares * current_prices[ticker]
            
            # Calculate returns
            returns = {}
            invested_amounts = {}
            for txn in simulation_result.get('transaction_log', []):
                if isinstance(txn, dict):
                    ticker = txn.get('ticker')
                    cost = txn.get('cost', 0)
                    invested_amounts[ticker] = invested_amounts.get(ticker, 0) + cost
            
            for ticker, shares in holdings.items():
                invested = invested_amounts.get(ticker, 0)
                current_val = shares * current_prices.get(ticker, 0)
                returns[ticker] = current_val - invested
            
            # Calculate percent returns
            percent_returns = {}
            for ticker, ret in returns.items():
                invested = invested_amounts.get(ticker, 0)
                if invested > 0:
                    percent_returns[ticker] = (ret / invested) * 100
            
            # Calculate allocations
            allocations = {}
            stock_value = total_value - remaining_cash
            for ticker, shares in holdings.items():
                ticker_value = shares * current_prices.get(ticker, 0)
                if stock_value > 0:
                    allocations[ticker] = (ticker_value / stock_value) * 100
            
            # Build investment log
            investment_log = []
            for txn in simulation_result.get('transaction_log', []):
                if isinstance(txn, dict):
                    investment_log.append(
                        f"Bought {txn.get('shares', 0):.2f} shares of {txn.get('ticker', 'N/A')} "
                        f"at ${txn.get('price', 0):.2f} on {txn.get('date', 'N/A')} "
                        f"(${txn.get('cost', 0):,.2f})"
                    )
            
            # Build performance data from historical prices
            performance_data = []
            if prices_data and benchmark_result:
                spy_prices = benchmark_result.get('spy_prices', {})
                # Get all dates
                all_dates = set()
                for ticker_prices in prices_data.values():
                    all_dates.update(ticker_prices.keys())
                
                sorted_dates = sorted(all_dates)
                initial_investment = sum(invested_amounts.values()) + remaining_cash
                
                for date in sorted_dates:
                    # Calculate portfolio value at this date
                    portfolio_value = remaining_cash
                    for ticker, shares in holdings.items():
                        ticker_prices = prices_data.get(ticker, {})
                        # Find the price at or before this date
                        price = None
                        for d in sorted(ticker_prices.keys()):
                            if d <= date:
                                price = ticker_prices[d]
                        if price:
                            portfolio_value += shares * price
                    
                    # Get SPY value at this date
                    spy_price = spy_prices.get(date)
                    spy_value = initial_investment  # Default
                    if spy_price and spy_simulation_result:
                        spy_shares = spy_simulation_result.get('spy_shares', 0)
                        spy_cash = spy_simulation_result.get('remaining_cash', 0)
                        spy_value = spy_shares * spy_price + spy_cash
                    
                    performance_data.append(PerformancePoint(
                        date=date,
                        portfolio=portfolio_value,
                        spy=spy_value,
                    ))
            
            summary = InvestmentSummary(
                holdings=holdings,
                final_prices=current_prices,
                cash=remaining_cash,
                returns=returns,
                total_value=total_value,
                investment_log=investment_log,
                percent_allocation=allocations,
                percent_return=percent_returns,
                performance_data=performance_data,
                insights=None,
            )
            
            logger.info(f"Successfully built summary from Composio tool results (without metrics)! Total value: {summary.total_value}")
            return summary
        
        # We need at least metrics to build a summary the standard way
        if not metrics_result:
            logger.warning(f"No CALCULATE_METRICS result found. Available tools: {list(normalized_results.keys())}")
            return None
        
        # Extract data from metrics result
        holdings = metrics_result.get('holdings', {})
        if not holdings and simulation_result:
            holdings = simulation_result.get('holdings', {})
        
        current_prices = {}
        if stock_data_result and 'prices' in stock_data_result:
            # Get latest price for each ticker
            for ticker, prices in stock_data_result['prices'].items():
                if prices:
                    latest_date = max(prices.keys())
                    current_prices[ticker] = prices[latest_date]
        
        # Get remaining cash
        remaining_cash = 0.0
        if simulation_result:
            remaining_cash = simulation_result.get('remaining_cash', 0.0)
        
        # Build investment log from simulation transactions
        investment_log = []
        if simulation_result and 'transaction_log' in simulation_result:
            for txn in simulation_result['transaction_log']:
                if isinstance(txn, dict):
                    investment_log.append(
                        f"Bought {txn.get('shares', 0):.2f} shares of {txn.get('ticker', 'N/A')} "
                        f"at ${txn.get('price', 0):.2f} on {txn.get('date', 'N/A')} "
                        f"(${txn.get('cost', 0):,.2f})"
                    )
        
        # Build performance data
        performance_data = []
        if 'performance_data' in metrics_result:
            for point in metrics_result['performance_data']:
                if isinstance(point, dict):
                    performance_data.append(PerformancePoint(
                        date=point.get('date', ''),
                        portfolio=point.get('portfolio', 0.0),
                        spy=point.get('spy', 0.0),
                    ))
        
        summary = InvestmentSummary(
            holdings=holdings,
            final_prices=current_prices or metrics_result.get('current_prices', {}),
            cash=remaining_cash,
            returns=metrics_result.get('returns', {}),
            total_value=metrics_result.get('total_value', 0.0),
            investment_log=investment_log,
            percent_allocation=metrics_result.get('allocations', {}),
            percent_return=metrics_result.get('percent_returns', {}),
            performance_data=performance_data,
            insights=None,
        )
        
        logger.info(f"Successfully built summary from Composio tool results! Total value: {summary.total_value}")
        return summary
        
    except Exception as e:
        logger.warning(f"Failed to build summary from tool results: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        return None


def parse_agent_output_to_summary(
    agent_output: str,
    fallback_params: Optional[Dict[str, Any]] = None,
) -> Optional[InvestmentSummary]:
    """Parse the agent's output to extract investment summary data.
    
    The agent may return structured data or natural language. This function
    attempts to extract the relevant information and construct an InvestmentSummary.
    
    Args:
        agent_output: The raw output from the agent
        fallback_params: Optional fallback parameters if parsing fails
        
    Returns:
        InvestmentSummary if parsing succeeds, None otherwise
    """
    try:
        # Try to parse as JSON first
        if isinstance(agent_output, str):
            # Look for JSON in the output
            json_match = re.search(r'\{[\s\S]*\}', agent_output)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    # Check if it looks like an investment summary
                    if "holdings" in data or "total_value" in data:
                        return InvestmentSummary(
                            holdings=data.get("holdings", {}),
                            final_prices=data.get("final_prices", {}),
                            cash=data.get("cash", 0.0),
                            returns=data.get("returns", {}),
                            total_value=data.get("total_value", 0.0),
                            investment_log=data.get("investment_log", []),
                            percent_allocation=data.get("percent_allocation", {}),
                            percent_return=data.get("percent_return", {}),
                            performance_data=[
                                PerformancePoint(**p) if isinstance(p, dict) else p
                                for p in data.get("performance_data", [])
                            ],
                            insights=data.get("insights"),
                        )
                except json.JSONDecodeError:
                    pass
        
        # If we can't parse the output, return None
        # The caller should fall back to direct tool execution
        return None
        
    except Exception as e:
        logger.warning(f"Failed to parse agent output: {e}")
        return None


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "Stock Portfolio Analysis Agent",
        "version": "0.1.0",
        "description": "AI-powered stock portfolio analysis using Composio Tool Router",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_portfolio(request: AnalyzeRequest):
    """Analyze a portfolio based on a natural language query.
    
    This endpoint uses Runner.run(agent, query) to process the user's
    natural language query through the AI agent, which orchestrates
    tool execution via Composio Tool Router.
    
    Requirements: 3.1, 3.2, 10.1, 10.2, 10.3, 10.4
    
    This endpoint:
    - Accepts user queries and portfolio state (10.1)
    - Streams progress events using SSE (10.2)
    - Emits state delta events for UI updates (10.3)
    - Returns investment summary with all calculated metrics (10.4)
    - Uses Runner.run(agent, query) for tool orchestration (3.1, 3.2)
    
    Args:
        request: AnalyzeRequest with query, user_id, and portfolio
        
    Returns:
        StreamingResponse with SSE events
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for the analysis process."""
        try:
            # Generate user_id if not provided
            user_id = request.user_id or str(uuid.uuid4())
            
            # Emit status: Starting analysis
            yield format_sse_event("status", {"message": "Starting portfolio analysis..."})
            await asyncio.sleep(0.1)  # Small delay for UI responsiveness
            
            # Prepare portfolio holdings for the agent context
            portfolio_holdings = None
            if request.portfolio and request.portfolio.holdings:
                portfolio_holdings = [
                    {
                        "ticker_symbol": h.ticker_symbol,
                        "investment_amount": h.investment_amount,
                        "purchase_date": h.purchase_date,
                    }
                    for h in request.portfolio.holdings
                ]
            
            # Run the agent with the user's query
            # Requirements: 3.1, 3.2 - Use Runner.run(agent, query) instead of direct tool calls
            yield format_sse_event("status", {"message": "Running AI agent for portfolio analysis..."})
            
            agent_result = await run_agent_analysis(
                query=request.query,
                user_id=user_id,
                portfolio_holdings=portfolio_holdings,
            )
            
            if not agent_result.get("success"):
                yield format_sse_event("error", {
                    "message": agent_result.get("error", "Agent execution failed")
                })
                return
            
            yield format_sse_event("tool_log", {
                "tool": "agent",
                "message": "Agent completed analysis"
            })
            
            # Try to parse the agent output into an InvestmentSummary
            agent_output = agent_result.get("output", "")
            summary = parse_agent_output_to_summary(agent_output)
            
            if summary:
                # Emit portfolio update
                yield format_sse_event("portfolio_update", {
                    "holdings": summary.holdings,
                    "remaining_cash": summary.cash,
                    "total_invested": summary.total_value - summary.cash,
                })
                
                # Emit the final result
                yield format_sse_event("result", summary.model_dump())
                yield format_sse_event("status", {"message": "Analysis complete!"})
                return
            
            # If we couldn't parse the agent output, fall back to direct tool execution
            # This ensures backward compatibility while the agent learns to return structured data
            logger.info("Agent output could not be parsed, falling back to direct tool execution")
            
            # Parse the query to extract parameters
            yield format_sse_event("status", {"message": "Parsing investment query..."})
            params = parse_query_simple(request.query)
            
            # Validate we have tickers
            if not params["ticker_symbols"]:
                yield format_sse_event("error", {
                    "message": "Could not identify any stock tickers in your query. Please specify ticker symbols (e.g., AAPL, GOOGL) or company names."
                })
                return
            
            yield format_sse_event("tool_log", {
                "tool": "query_parser",
                "message": f"Extracted tickers: {params['ticker_symbols']}, Amount: ${params['total_cash']:,.2f}, Strategy: {params['strategy']}"
            })
            await asyncio.sleep(0.1)
            
            # Merge with existing portfolio if provided
            if request.portfolio and request.portfolio.holdings:
                yield format_sse_event("status", {"message": "Merging with existing portfolio..."})
                for holding in request.portfolio.holdings:
                    if holding.ticker_symbol not in params["ticker_symbols"]:
                        params["ticker_symbols"].append(holding.ticker_symbol)
                    # Add to investment amounts
                    existing_amount = params["investment_amounts"].get(holding.ticker_symbol, 0)
                    params["investment_amounts"][holding.ticker_symbol] = existing_amount + holding.investment_amount
                    params["total_cash"] += holding.investment_amount
            
            # Step 1: Fetch stock data
            yield format_sse_event("status", {"message": f"Fetching stock data for {', '.join(params['ticker_symbols'])}..."})
            
            stock_data_input = FetchStockDataInput(
                ticker_symbols=params["ticker_symbols"],
                start_date=params["start_date"],
                end_date=params["end_date"],
                interval="3mo",
            )
            stock_data = fetch_stock_data(stock_data_input)
            
            # Check for tool error
            if is_tool_error(stock_data):
                yield format_sse_event("error", {
                    "error_code": stock_data.error_code.value,
                    "message": format_tool_error_message(stock_data),
                })
                return
            
            yield format_sse_event("tool_log", {
                "tool": "fetch_stock_data",
                "message": f"Retrieved {stock_data.metadata.data_points} data points for {len(stock_data.metadata.tickers)} tickers"
            })
            
            await asyncio.sleep(0.1)
            
            # Get portfolio dates for SPY alignment
            portfolio_dates = sorted(set(
                date for ticker_prices in stock_data.prices.values()
                for date in ticker_prices.keys()
            ))
            
            # Step 2: Fetch benchmark data
            yield format_sse_event("status", {"message": "Fetching SPY benchmark data..."})
            
            benchmark_input = FetchBenchmarkInput(
                start_date=params["start_date"],
                end_date=params["end_date"],
                portfolio_dates=portfolio_dates,
            )
            benchmark_data = fetch_benchmark_data(benchmark_input)
            
            # Check for tool error
            if is_tool_error(benchmark_data):
                yield format_sse_event("error", {
                    "error_code": benchmark_data.error_code.value,
                    "message": format_tool_error_message(benchmark_data),
                })
                return
            
            yield format_sse_event("tool_log", {
                "tool": "fetch_benchmark_data",
                "message": f"Retrieved SPY data for {len(benchmark_data.spy_prices)} dates"
            })
            
            await asyncio.sleep(0.1)
            
            # Step 3: Simulate portfolio
            yield format_sse_event("status", {"message": f"Simulating {params['strategy']} investment strategy..."})
            
            simulate_input = SimulatePortfolioInput(
                stock_prices=stock_data.prices,
                ticker_amounts=params["investment_amounts"],
                strategy=params["strategy"],
                available_cash=params["total_cash"],
                dca_interval=params["dca_interval"],
            )
            simulation = simulate_portfolio(simulate_input)
            
            # Check for tool error
            if is_tool_error(simulation):
                yield format_sse_event("error", {
                    "error_code": simulation.error_code.value,
                    "message": format_tool_error_message(simulation),
                })
                return
            
            yield format_sse_event("tool_log", {
                "tool": "simulate_portfolio",
                "message": f"Executed {len(simulation.transaction_log)} transactions, remaining cash: ${simulation.remaining_cash:,.2f}"
            })
            
            # Emit portfolio update
            yield format_sse_event("portfolio_update", {
                "holdings": simulation.holdings,
                "remaining_cash": simulation.remaining_cash,
                "total_invested": simulation.total_invested,
            })
            
            await asyncio.sleep(0.1)
            
            # Step 4: Simulate SPY investment for comparison
            yield format_sse_event("status", {"message": "Simulating SPY benchmark investment..."})
            
            spy_input = SimulateSPYInput(
                total_amount=simulation.total_invested + simulation.remaining_cash,
                spy_prices=benchmark_data.spy_prices,
                strategy=params["strategy"],
                dca_interval=params["dca_interval"],
            )
            spy_simulation = simulate_spy_investment(spy_input)
            
            # Check for tool error
            if is_tool_error(spy_simulation):
                yield format_sse_event("error", {
                    "error_code": spy_simulation.error_code.value,
                    "message": format_tool_error_message(spy_simulation),
                })
                return
            
            yield format_sse_event("tool_log", {
                "tool": "simulate_spy_investment",
                "message": f"SPY simulation: {spy_simulation.spy_shares:.2f} shares, invested: ${spy_simulation.total_invested:,.2f}"
            })
            
            await asyncio.sleep(0.1)
            
            # Step 5: Calculate metrics
            yield format_sse_event("status", {"message": "Calculating portfolio metrics..."})
            
            # Get current prices (last price for each ticker)
            current_prices = {}
            for ticker, prices in stock_data.prices.items():
                if prices:
                    latest_date = max(prices.keys())
                    current_prices[ticker] = prices[latest_date]
            
            # Calculate invested amounts from transactions
            invested_amounts = {}
            for txn in simulation.transaction_log:
                invested_amounts[txn.ticker] = invested_amounts.get(txn.ticker, 0) + txn.cost
            
            metrics_input = CalculateMetricsInput(
                holdings=simulation.holdings,
                current_prices=current_prices,
                invested_amounts=invested_amounts,
                historical_prices=stock_data.prices,
                spy_prices=benchmark_data.spy_prices,
                remaining_cash=simulation.remaining_cash,
            )
            metrics = calculate_metrics(metrics_input)
            
            # Check for tool error
            if is_tool_error(metrics):
                yield format_sse_event("error", {
                    "error_code": metrics.error_code.value,
                    "message": format_tool_error_message(metrics),
                })
                return
            
            yield format_sse_event("tool_log", {
                "tool": "calculate_metrics",
                "message": f"Portfolio value: ${metrics.total_value:,.2f}, Performance data points: {len(metrics.performance_data)}"
            })
            
            await asyncio.sleep(0.1)
            
            # Step 6: Generate investment log
            investment_log = []
            for txn in simulation.transaction_log:
                investment_log.append(
                    f"Bought {txn.shares:.2f} shares of {txn.ticker} at ${txn.price:.2f} on {txn.date} (${txn.cost:,.2f})"
                )
            
            # Build the initial investment summary (without insights)
            yield format_sse_event("status", {"message": "Generating investment summary..."})
            
            summary = InvestmentSummary(
                holdings=simulation.holdings,
                final_prices=current_prices,
                cash=simulation.remaining_cash,
                returns=metrics.returns,
                total_value=metrics.total_value,
                investment_log=investment_log,
                percent_allocation=metrics.allocations,
                percent_return=metrics.percent_returns,
                performance_data=metrics.performance_data,
                insights=None,
            )
            
            # Step 7: Generate insights via LLM (Requirements 4.1, 4.2, 4.3, 4.4)
            yield format_sse_event("status", {"message": "Generating AI insights..."})
            
            try:
                insights = await generate_insights_from_summary(summary)
                if insights:
                    summary.insights = insights
                    yield format_sse_event("tool_log", {
                        "tool": "insights_generator",
                        "message": f"Generated {len(insights.bull_insights)} bull and {len(insights.bear_insights)} bear insights"
                    })
                else:
                    yield format_sse_event("tool_log", {
                        "tool": "insights_generator",
                        "message": "Could not generate insights"
                    })
            except Exception as e:
                logger.warning(f"Failed to generate insights: {e}")
                yield format_sse_event("tool_log", {
                    "tool": "insights_generator",
                    "message": f"Insights generation failed: {str(e)}"
                })
            
            # Emit the final result
            yield format_sse_event("result", summary.model_dump())
            
            yield format_sse_event("status", {"message": "Analysis complete!"})
            
        except Exception as e:
            yield format_sse_event("error", {"message": f"Unexpected error: {str(e)}"})
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.post("/analyze/sync", response_model=AnalyzeResponse)
async def analyze_portfolio_sync(request: AnalyzeRequest) -> AnalyzeResponse:
    """Synchronous version of portfolio analysis (non-streaming).
    
    This endpoint uses Runner.run(agent, query) to process the user's
    natural language query through the AI agent, which orchestrates
    tool execution via Composio Tool Router.
    
    Requirements: 3.1, 3.2, 10.1, 10.4
    
    Args:
        request: AnalyzeRequest with query, user_id, and portfolio
        
    Returns:
        AnalyzeResponse with success status and investment summary
    """
    try:
        # Generate user_id if not provided
        user_id = request.user_id or str(uuid.uuid4())
        
        # Prepare portfolio holdings for the agent context
        portfolio_holdings = None
        if request.portfolio and request.portfolio.holdings:
            portfolio_holdings = [
                {
                    "ticker_symbol": h.ticker_symbol,
                    "investment_amount": h.investment_amount,
                    "purchase_date": h.purchase_date,
                }
                for h in request.portfolio.holdings
            ]
        
        # Run the agent with the user's query
        # Requirements: 3.1, 3.2 - Use Runner.run(agent, query) instead of direct tool calls
        agent_result = await run_agent_analysis(
            query=request.query,
            user_id=user_id,
            portfolio_holdings=portfolio_holdings,
        )
        
        if not agent_result.get("success"):
            return AnalyzeResponse(
                success=False,
                error=agent_result.get("error", "Agent execution failed")
            )
        
        # Check if agent called tools via Composio
        tool_calls_made = agent_result.get("tool_calls_made", [])
        tool_results = agent_result.get("tool_results", {})
        
        if tool_calls_made:
            logger.info(f"Agent successfully called tools via Composio: {tool_calls_made}")
            
            # Try to build summary from tool results
            summary = build_summary_from_tool_results(tool_results)
            if summary:
                logger.info("Successfully built summary from Composio tool results")
                # Generate insights via LLM (Requirements 4.1, 4.2, 4.3, 4.4)
                try:
                    insights = await generate_insights_from_summary(summary)
                    if insights:
                        summary.insights = insights
                        logger.info(f"Generated {len(insights.bull_insights)} bull and {len(insights.bear_insights)} bear insights")
                except Exception as e:
                    logger.warning(f"Failed to generate insights: {e}")
                    # Continue without insights - they are optional
                return AnalyzeResponse(success=True, summary=summary)
            else:
                logger.warning("Could not build summary from tool results, trying text parsing")
        
        # Try to parse the agent output into an InvestmentSummary
        agent_output = agent_result.get("output", "")
        summary = parse_agent_output_to_summary(agent_output)
        
        if summary:
            # Generate insights via LLM (Requirements 4.1, 4.2, 4.3, 4.4)
            try:
                insights = await generate_insights_from_summary(summary)
                if insights:
                    summary.insights = insights
                    logger.info(f"Generated {len(insights.bull_insights)} bull and {len(insights.bear_insights)} bear insights")
            except Exception as e:
                logger.warning(f"Failed to generate insights: {e}")
                # Continue without insights - they are optional
            return AnalyzeResponse(success=True, summary=summary)
        
        # If we couldn't parse the agent output, fall back to direct tool execution
        # This ensures backward compatibility while the agent learns to return structured data
        logger.info("Agent output could not be parsed, falling back to direct tool execution")
        
        # Parse the query to extract parameters
        params = parse_query_simple(request.query)
        
        # Validate we have tickers
        if not params["ticker_symbols"]:
            return AnalyzeResponse(
                success=False,
                error="Could not identify any stock tickers in your query."
            )
        
        # Merge with existing portfolio if provided
        if request.portfolio and request.portfolio.holdings:
            for holding in request.portfolio.holdings:
                if holding.ticker_symbol not in params["ticker_symbols"]:
                    params["ticker_symbols"].append(holding.ticker_symbol)
                existing_amount = params["investment_amounts"].get(holding.ticker_symbol, 0)
                params["investment_amounts"][holding.ticker_symbol] = existing_amount + holding.investment_amount
                params["total_cash"] += holding.investment_amount
        
        # Fetch stock data
        stock_data_input = FetchStockDataInput(
            ticker_symbols=params["ticker_symbols"],
            start_date=params["start_date"],
            end_date=params["end_date"],
            interval="3mo",
        )
        stock_data = fetch_stock_data(stock_data_input)
        
        # Check for tool error
        if is_tool_error(stock_data):
            return AnalyzeResponse(
                success=False,
                error=format_tool_error_message(stock_data)
            )
        
        # Get portfolio dates
        portfolio_dates = sorted(set(
            date for ticker_prices in stock_data.prices.values()
            for date in ticker_prices.keys()
        ))
        
        # Fetch benchmark data
        benchmark_input = FetchBenchmarkInput(
            start_date=params["start_date"],
            end_date=params["end_date"],
            portfolio_dates=portfolio_dates,
        )
        benchmark_data = fetch_benchmark_data(benchmark_input)
        
        # Check for tool error
        if is_tool_error(benchmark_data):
            return AnalyzeResponse(
                success=False,
                error=format_tool_error_message(benchmark_data)
            )
        
        # Simulate portfolio
        simulate_input = SimulatePortfolioInput(
            stock_prices=stock_data.prices,
            ticker_amounts=params["investment_amounts"],
            strategy=params["strategy"],
            available_cash=params["total_cash"],
            dca_interval=params["dca_interval"],
        )
        simulation = simulate_portfolio(simulate_input)
        
        # Check for tool error
        if is_tool_error(simulation):
            return AnalyzeResponse(
                success=False,
                error=format_tool_error_message(simulation)
            )
        
        # Get current prices
        current_prices = {}
        for ticker, prices in stock_data.prices.items():
            if prices:
                latest_date = max(prices.keys())
                current_prices[ticker] = prices[latest_date]
        
        # Calculate invested amounts
        invested_amounts = {}
        for txn in simulation.transaction_log:
            invested_amounts[txn.ticker] = invested_amounts.get(txn.ticker, 0) + txn.cost
        
        # Calculate metrics
        metrics_input = CalculateMetricsInput(
            holdings=simulation.holdings,
            current_prices=current_prices,
            invested_amounts=invested_amounts,
            historical_prices=stock_data.prices,
            spy_prices=benchmark_data.spy_prices,
            remaining_cash=simulation.remaining_cash,
        )
        metrics = calculate_metrics(metrics_input)
        
        # Check for tool error
        if is_tool_error(metrics):
            return AnalyzeResponse(
                success=False,
                error=format_tool_error_message(metrics)
            )
        
        # Generate investment log
        investment_log = [
            f"Bought {txn.shares:.2f} shares of {txn.ticker} at ${txn.price:.2f} on {txn.date} (${txn.cost:,.2f})"
            for txn in simulation.transaction_log
        ]
        
        # Build summary (without insights initially)
        summary = InvestmentSummary(
            holdings=simulation.holdings,
            final_prices=current_prices,
            cash=simulation.remaining_cash,
            returns=metrics.returns,
            total_value=metrics.total_value,
            investment_log=investment_log,
            percent_allocation=metrics.allocations,
            percent_return=metrics.percent_returns,
            performance_data=metrics.performance_data,
            insights=None,
        )
        
        # Generate insights via LLM (Requirements 4.1, 4.2, 4.3, 4.4)
        try:
            insights = await generate_insights_from_summary(summary)
            if insights:
                summary.insights = insights
                logger.info(f"Generated {len(insights.bull_insights)} bull and {len(insights.bear_insights)} bear insights")
        except Exception as e:
            logger.warning(f"Failed to generate insights: {e}")
            # Continue without insights - they are optional
        
        return AnalyzeResponse(success=True, summary=summary)
        
    except ToolExecutionError as e:
        logger.error(f"Tool execution error in sync endpoint: {e.error_code.value} - {e.message}")
        return AnalyzeResponse(success=False, error=e.message)
    except Exception as e:
        logger.exception(f"Unexpected error in sync endpoint: {str(e)}")
        return AnalyzeResponse(success=False, error=str(e))


# =============================================================================
# Session-Based Analysis Endpoints
# =============================================================================


class SessionAnalyzeRequest(BaseModel):
    """Request model for session-based analysis.
    
    Requirements: 9.2 - Use Composio sessions for user-scoped state management
    """
    query: str = Field(
        ...,
        description="Natural language investment query"
    )
    user_id: str = Field(
        ...,
        description="User ID for session management"
    )
    use_agent: bool = Field(
        default=False,
        description="Whether to use the LLM agent for query parsing"
    )


class SessionInfoResponse(BaseModel):
    """Response model for session info endpoint."""
    user_id: str = Field(..., description="User ID")
    session_active: bool = Field(..., description="Whether session is active")
    tools_registered: int = Field(..., description="Number of registered tools")


def get_session_manager_instance():
    """Get the session manager instance.
    
    Requirements: 9.1, 9.2, 9.3 - Initialize Composio and register tools
    
    Returns:
        SessionManager instance
    """
    from agent.session import get_session_manager
    return get_session_manager()


def get_orchestrator_instance():
    """Get the agent orchestrator instance.
    
    Requirements: 9.4 - Tool Router handles LLM interactions
    
    Returns:
        AgentOrchestrator instance
    """
    from agent.agent_config import get_orchestrator
    return get_orchestrator()


@app.get("/session/{user_id}")
async def get_session_info(user_id: str) -> SessionInfoResponse:
    """Get session information for a user.
    
    Requirements: 9.2 - Use Composio sessions for user-scoped state management
    
    Args:
        user_id: User ID to get session info for
        
    Returns:
        SessionInfoResponse with session details
    """
    try:
        session_manager = get_session_manager_instance()
        session = session_manager.get_session(user_id)
        
        if session is None:
            return SessionInfoResponse(
                user_id=user_id,
                session_active=False,
                tools_registered=0,
            )
        
        # Get tools count
        tools = session_manager.get_tools(user_id)
        
        return SessionInfoResponse(
            user_id=user_id,
            session_active=True,
            tools_registered=len(tools) if tools else 0,
        )
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{user_id}/create")
async def create_session(user_id: str) -> SessionInfoResponse:
    """Create a new session for a user.
    
    Requirements: 9.2 - Use Composio sessions for user-scoped state management
    
    Args:
        user_id: User ID to create session for
        
    Returns:
        SessionInfoResponse with session details
    """
    try:
        session_manager = get_session_manager_instance()
        session = session_manager.create_session(user_id)
        
        # Get tools count
        tools = session_manager.get_tools(user_id)
        
        return SessionInfoResponse(
            user_id=user_id,
            session_active=True,
            tools_registered=len(tools) if tools else 0,
        )
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{user_id}")
async def close_session(user_id: str) -> Dict[str, Any]:
    """Close a user session.
    
    Args:
        user_id: User ID to close session for
        
    Returns:
        Success message
    """
    try:
        session_manager = get_session_manager_instance()
        session_manager.close_session(user_id)
        
        return {"message": f"Session closed for user {user_id}"}
    except Exception as e:
        logger.error(f"Error closing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools")
async def list_registered_tools() -> Dict[str, Any]:
    """List all registered custom tools.
    
    Requirements: 9.3 - Register all custom tools with Composio
    
    Returns:
        Dict with list of registered tools
    """
    tools = [
        {
            "name": "fetch_stock_data",
            "description": "Fetch historical closing prices for specified stock tickers",
        },
        {
            "name": "fetch_benchmark_data",
            "description": "Fetch SPY benchmark prices aligned to portfolio dates",
        },
        {
            "name": "simulate_portfolio",
            "description": "Simulate buying stocks based on investment strategy",
        },
        {
            "name": "simulate_spy_investment",
            "description": "Simulate investing in SPY using the same strategy as the portfolio",
        },
        {
            "name": "calculate_metrics",
            "description": "Calculate portfolio performance metrics",
        },
    ]
    
    return {
        "tools": tools,
        "count": len(tools),
    }


@app.get("/config")
async def get_config() -> Dict[str, Any]:
    """Get current API configuration.
    
    Returns:
        Dict with configuration details
    """
    import os
    
    return {
        "version": "0.1.0",
        "model": os.environ.get("MODEL", "gpt-4o-mini"),
        "composio_configured": bool(os.environ.get("COMPOSIO_API_KEY")),
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
    }
