"""Session management for the Stock Portfolio Analysis Agent.

This module provides Composio session management functionality including:
- Initializing Composio with OpenAI Agents provider
- Creating user sessions
- Registering all custom tools

Requirements: 2.1, 2.2, 2.3
"""

import os
from typing import Any, Dict, List, Optional

from composio import Composio
from composio_openai_agents import OpenAIAgentsProvider

from agent.models import (
    BenchmarkDataOutput,
    CalculateMetricsInput,
    FetchBenchmarkInput,
    FetchStockDataInput,
    MetricsOutput,
    SimulatePortfolioInput,
    SimulateSPYInput,
    SimulationOutput,
    SPYSimulationOutput,
    StockDataOutput,
)
from agent.tools import (
    _fetch_stock_data_impl,
    _fetch_benchmark_data_impl,
    _simulate_portfolio_impl,
    _simulate_spy_investment_impl,
    _calculate_metrics_impl,
)

# Default user ID for session creation
DEFAULT_USER_ID = "default"

# Custom tool slugs - these are the tools registered with @composio.tools.custom_tool
CUSTOM_TOOL_SLUGS = [
    "FETCH_STOCK_DATA",
    "FETCH_BENCHMARK_DATA",
    "SIMULATE_PORTFOLIO",
    "SIMULATE_SPY_INVESTMENT",
    "CALCULATE_METRICS",
]


class SessionManager:
    """Manages Composio sessions and tool registration for users.
    
    This class provides session-scoped tool access through Composio,
    initializing the Composio client with OpenAI Agents provider and
    registering all custom tools.
    
    Requirements: 2.1, 2.2, 2.3
    """
    
    def __init__(self, api_key: Optional[str] = None, auto_create_default: bool = True):
        """Initialize the session manager with Composio.
        
        Creates a Composio client with OpenAI Agents provider and optionally
        creates a default session on initialization.
        
        Requirements: 2.1, 2.3 - Create session on application startup
        
        Args:
            api_key: Optional Composio API key. If not provided, will use
                     COMPOSIO_API_KEY environment variable.
            auto_create_default: If True, creates a default session on initialization.
        """
        self._api_key = api_key or os.environ.get("COMPOSIO_API_KEY")
        self._composio: Optional[Composio] = None
        self._sessions: Dict[str, Any] = {}
        self._tools_registered = False
        self._default_session: Optional[Any] = None
        
        # Initialize Composio and create default session on startup (Requirement 2.3)
        if auto_create_default:
            self._initialize()
    
    def _initialize(self) -> None:
        """Initialize Composio and create the default session.
        
        Requirements: 2.1, 2.3 - Create session using composio.create(user_id="default")
        """
        # Initialize Composio client
        self._get_composio()
        # Create default session on startup
        self._default_session = self.create_session(DEFAULT_USER_ID)
    
    def _get_composio(self) -> Composio:
        """Get or create the Composio client instance.
        
        Requirements: 2.1 - Initialize Composio with OpenAI provider
        
        Returns:
            Composio client instance configured with OpenAI Agents provider
        """
        if self._composio is None:
            self._composio = Composio(
                api_key=self._api_key,
                provider=OpenAIAgentsProvider(),
            )
            self._register_custom_tools()
        return self._composio
    
    def _register_custom_tools(self) -> None:
        """Register all custom tools with Composio.
        
        Requirements: 2.2 - Session provides access to all registered custom tools
        
        This method registers the following custom tools:
        - fetch_stock_data: Fetch historical stock prices
        - fetch_benchmark_data: Fetch SPY benchmark data
        - simulate_portfolio: Simulate portfolio investment
        - simulate_spy_investment: Simulate SPY benchmark investment
        - calculate_metrics: Calculate portfolio performance metrics
        """
        if self._tools_registered:
            return
        
        composio = self._composio
        if composio is None:
            return
        
        # Register fetch_stock_data as a Composio custom tool
        @composio.tools.custom_tool
        def fetch_stock_data(request: FetchStockDataInput) -> dict:
            """Fetch historical closing prices for specified stock tickers.
            
            Downloads historical stock price data from Yahoo Finance
            for the specified tickers and date range.
            
            Args:
                request: Input containing ticker_symbols, start_date, end_date, and interval
                
            Returns:
                Dict with prices and metadata
            """
            result = _fetch_stock_data_impl(request)
            return result.model_dump()
        
        # Register fetch_benchmark_data as a Composio custom tool
        @composio.tools.custom_tool
        def fetch_benchmark_data(request: FetchBenchmarkInput) -> dict:
            """Fetch SPY benchmark prices aligned to portfolio dates.
            
            Downloads SPY (S&P 500 ETF) price data from Yahoo Finance
            and aligns it to the portfolio dates using forward-fill.
            
            Args:
                request: Input containing start_date, end_date, and portfolio_dates
                
            Returns:
                Dict with SPY prices aligned to portfolio dates
            """
            result = _fetch_benchmark_data_impl(request)
            return result.model_dump()
        
        # Register simulate_portfolio as a Composio custom tool
        @composio.tools.custom_tool
        def simulate_portfolio(request: SimulatePortfolioInput) -> dict:
            """Simulate buying stocks based on investment strategy.
            
            Simulates portfolio investment using either single-shot
            or DCA (Dollar-Cost Averaging) strategy.
            
            Args:
                request: Input containing stock_prices, ticker_amounts, strategy,
                         available_cash, and optional dca_interval
                
            Returns:
                Dict with holdings, remaining_cash, transaction_log,
                and insufficient_funds list
            """
            result = _simulate_portfolio_impl(request)
            return result.model_dump()
        
        # Register simulate_spy_investment as a Composio custom tool
        @composio.tools.custom_tool
        def simulate_spy_investment(request: SimulateSPYInput) -> dict:
            """Simulate investing in SPY using the same strategy as the portfolio.
            
            Simulates SPY investment using either single-shot
            or DCA (Dollar-Cost Averaging) strategy, matching the portfolio strategy.
            
            Args:
                request: Input containing total_amount, spy_prices, strategy,
                         and optional dca_interval
                
            Returns:
                Dict with spy_shares, remaining_cash, total_invested,
                transaction_log, and value_over_time
            """
            result = _simulate_spy_investment_impl(request)
            return result.model_dump()
        
        # Register calculate_metrics as a Composio custom tool
        @composio.tools.custom_tool
        def calculate_metrics(request: CalculateMetricsInput) -> dict:
            """Calculate portfolio performance metrics.
            
            Calculates comprehensive portfolio metrics including total value,
            returns, allocations, and time-series performance data.
            
            Args:
                request: Input containing holdings, current_prices, invested_amounts,
                         historical_prices, spy_prices, and remaining_cash
                
            Returns:
                Dict with total_value, returns, percent_returns,
                allocations, and performance_data
            """
            result = _calculate_metrics_impl(request)
            return result.model_dump()
        
        self._tools_registered = True
    
    def create_session(self, user_id: str) -> Any:
        """Create a Composio session for a user.
        
        Requirements: 2.1 - Create session using composio.create(user_id="default")
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Composio session object for the user
        """
        composio = self._get_composio()
        
        # Create a session for the user using composio.create(user_id=...)
        session = composio.create(user_id=user_id)
        
        # Store the session for later retrieval
        self._sessions[user_id] = session
        
        return session
    
    def get_session(self, user_id: str) -> Optional[Any]:
        """Get an existing session for a user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Composio session object if exists, None otherwise
        """
        return self._sessions.get(user_id)
    
    def get_default_session(self) -> Any:
        """Get the default session created on initialization.
        
        Requirements: 2.3 - Session created on application startup
        
        Returns:
            The default Composio session
        """
        if self._default_session is None:
            self._default_session = self.create_session(DEFAULT_USER_ID)
        return self._default_session
    
    def get_or_create_session(self, user_id: str) -> Any:
        """Get an existing session or create a new one for a user.
        
        Requirements: 2.1 - Create session using composio.create(user_id=...)
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Composio session object for the user
        """
        session = self.get_session(user_id)
        if session is None:
            session = self.create_session(user_id)
        return session
    
    def get_tools(self, user_id: Optional[str] = None) -> List[Any]:
        """Get all registered custom tools as FunctionTool objects.
        
        Requirements: 2.2 - Session provides access to tools via session.tools()
        
        This method retrieves the custom tools registered with Composio and
        returns them as OpenAI Agents SDK FunctionTool objects.
        
        Args:
            user_id: Unique identifier for the user. If None, uses default user.
            
        Returns:
            List of FunctionTool objects for all 5 custom tools
        """
        composio = self._get_composio()
        effective_user_id = user_id or DEFAULT_USER_ID
        
        # Get custom tools as FunctionTool objects using composio.tools.get()
        # This wraps the custom tools for use with OpenAI Agents SDK
        tools = composio.tools.get(
            user_id=effective_user_id,
            tools=CUSTOM_TOOL_SLUGS,
        )
        return tools
    
    def tools(self) -> List[Any]:
        """Get all registered custom tools from the default session.
        
        Requirements: 2.2 - Session provides access to tools via session.tools()
        
        This is a convenience method that returns custom tools for the default user.
        
        Returns:
            List of FunctionTool objects for all 5 custom tools
        """
        return self.get_tools()
    
    def close_session(self, user_id: str) -> None:
        """Close and remove a user session.
        
        Args:
            user_id: Unique identifier for the user
        """
        if user_id in self._sessions:
            del self._sessions[user_id]
        # Reset default session if closing the default user
        if user_id == DEFAULT_USER_ID:
            self._default_session = None


# Global session manager instance - created lazily
_session_manager: Optional[SessionManager] = None


def get_session_manager(api_key: Optional[str] = None) -> SessionManager:
    """Get or create the global session manager instance.
    
    This function returns the global SessionManager, creating it if necessary.
    The SessionManager automatically creates a default session on initialization.
    
    Requirements: 2.3 - Session created on application startup
    
    Args:
        api_key: Optional Composio API key
        
    Returns:
        The global SessionManager instance with default session ready
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(api_key=api_key, auto_create_default=True)
    return _session_manager


def get_default_tools(api_key: Optional[str] = None) -> List[Any]:
    """Get tools from the default session.
    
    Convenience function to get tools from the default session.
    
    Requirements: 2.2 - Session provides access to tools via session.tools()
    
    Args:
        api_key: Optional Composio API key
        
    Returns:
        List of tools available in the default session
    """
    manager = get_session_manager(api_key)
    return manager.tools()
