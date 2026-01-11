"""Session management for the Stock Portfolio Analysis Agent.

This module provides Composio session management functionality including:
- Initializing Composio with OpenAI Agents provider
- Creating user sessions
- Registering all custom tools

Requirements: 9.1, 9.2, 9.3
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


class SessionManager:
    """Manages Composio sessions and tool registration for users.
    
    This class provides session-scoped tool access through Composio,
    initializing the Composio client with OpenAI Agents provider and
    registering all custom tools.
    
    Requirements: 9.1, 9.2, 9.3
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the session manager with Composio.
        
        Requirements: 9.1 - Use Tool Router to orchestrate the workflow
        
        Args:
            api_key: Optional Composio API key. If not provided, will use
                     COMPOSIO_API_KEY environment variable.
        """
        self._api_key = api_key or os.environ.get("COMPOSIO_API_KEY")
        self._composio: Optional[Composio] = None
        self._sessions: Dict[str, Any] = {}
        self._tools_registered = False
    
    def _get_composio(self) -> Composio:
        """Get or create the Composio client instance.
        
        Requirements: 9.1 - Initialize Composio with OpenAI provider
        
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
        
        Requirements: 9.3 - Register all custom tools with Composio
        
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
        def fetch_stock_data(request: FetchStockDataInput) -> StockDataOutput:
            """Fetch historical closing prices for specified stock tickers.
            
            Downloads historical stock price data from Yahoo Finance
            for the specified tickers and date range.
            
            Args:
                request: Input containing ticker_symbols, start_date, end_date, and interval
                
            Returns:
                StockDataOutput with prices dict and metadata
            """
            return _fetch_stock_data_impl(request)
        
        # Register fetch_benchmark_data as a Composio custom tool
        @composio.tools.custom_tool
        def fetch_benchmark_data(request: FetchBenchmarkInput) -> BenchmarkDataOutput:
            """Fetch SPY benchmark prices aligned to portfolio dates.
            
            Downloads SPY (S&P 500 ETF) price data from Yahoo Finance
            and aligns it to the portfolio dates using forward-fill.
            
            Args:
                request: Input containing start_date, end_date, and portfolio_dates
                
            Returns:
                BenchmarkDataOutput with SPY prices aligned to portfolio dates
            """
            return _fetch_benchmark_data_impl(request)
        
        # Register simulate_portfolio as a Composio custom tool
        @composio.tools.custom_tool
        def simulate_portfolio(request: SimulatePortfolioInput) -> SimulationOutput:
            """Simulate buying stocks based on investment strategy.
            
            Simulates portfolio investment using either single-shot
            or DCA (Dollar-Cost Averaging) strategy.
            
            Args:
                request: Input containing stock_prices, ticker_amounts, strategy,
                         available_cash, and optional dca_interval
                
            Returns:
                SimulationOutput with holdings, remaining_cash, transaction_log,
                and insufficient_funds list
            """
            return _simulate_portfolio_impl(request)
        
        # Register simulate_spy_investment as a Composio custom tool
        @composio.tools.custom_tool
        def simulate_spy_investment(request: SimulateSPYInput) -> SPYSimulationOutput:
            """Simulate investing in SPY using the same strategy as the portfolio.
            
            Simulates SPY investment using either single-shot
            or DCA (Dollar-Cost Averaging) strategy, matching the portfolio strategy.
            
            Args:
                request: Input containing total_amount, spy_prices, strategy,
                         and optional dca_interval
                
            Returns:
                SPYSimulationOutput with spy_shares, remaining_cash, total_invested,
                transaction_log, and value_over_time
            """
            return _simulate_spy_investment_impl(request)
        
        # Register calculate_metrics as a Composio custom tool
        @composio.tools.custom_tool
        def calculate_metrics(request: CalculateMetricsInput) -> MetricsOutput:
            """Calculate portfolio performance metrics.
            
            Calculates comprehensive portfolio metrics including total value,
            returns, allocations, and time-series performance data.
            
            Args:
                request: Input containing holdings, current_prices, invested_amounts,
                         historical_prices, spy_prices, and remaining_cash
                
            Returns:
                MetricsOutput with total_value, returns, percent_returns,
                allocations, and performance_data
            """
            return _calculate_metrics_impl(request)
        
        self._tools_registered = True
    
    def create_session(self, user_id: str) -> Any:
        """Create a Composio session for a user.
        
        Requirements: 9.2 - Use Composio sessions for user-scoped state management
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Composio session object for the user
        """
        composio = self._get_composio()
        
        # Create a session for the user
        # The session provides user-scoped tool access
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
    
    def get_or_create_session(self, user_id: str) -> Any:
        """Get an existing session or create a new one for a user.
        
        Requirements: 9.2 - Use Composio sessions for user-scoped state management
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Composio session object for the user
        """
        session = self.get_session(user_id)
        if session is None:
            session = self.create_session(user_id)
        return session
    
    def get_tools(self, user_id: str) -> List[Any]:
        """Get all registered tools for a user session.
        
        Requirements: 9.3 - Register all custom tools with Composio
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            List of tools available for the user session
        """
        session = self.get_or_create_session(user_id)
        return session.tools()
    
    def close_session(self, user_id: str) -> None:
        """Close and remove a user session.
        
        Args:
            user_id: Unique identifier for the user
        """
        if user_id in self._sessions:
            del self._sessions[user_id]


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager(api_key: Optional[str] = None) -> SessionManager:
    """Get or create the global session manager instance.
    
    Args:
        api_key: Optional Composio API key
        
    Returns:
        The global SessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(api_key=api_key)
    return _session_manager
