"""Composio Stock Portfolio Analysis Agent."""

from agent.models import (
    # Input models
    FetchStockDataInput,
    FetchBenchmarkInput,
    SimulatePortfolioInput,
    SimulateSPYInput,
    CalculateMetricsInput,
    # Output models
    StockDataOutput,
    StockDataMetadata,
    BenchmarkDataOutput,
    Transaction,
    SimulationOutput,
    SPYSimulationOutput,
    PerformancePoint,
    MetricsOutput,
    Insight,
    Insights,
    InvestmentSummary,
)

from agent.tools import (
    fetch_stock_data,
    fetch_benchmark_data,
    simulate_portfolio,
    simulate_spy_investment,
    calculate_metrics,
    register_tools,
)

from agent.portfolio import (
    PortfolioHolding,
    Portfolio,
    PortfolioManager,
    get_portfolio_manager,
)

from agent.prompts import (
    QUERY_PARSER_SYSTEM_PROMPT,
    INSIGHTS_GENERATOR_PROMPT,
    INSIGHTS_GENERATOR_WITH_CONTEXT_PROMPT,
    format_portfolio_context,
    get_query_parser_prompt,
    get_insights_prompt,
    format_portfolio_data_for_insights,
    get_insights_prompt_with_context,
)

from agent.session import (
    SessionManager,
    get_session_manager,
)

from agent.agent_config import (
    DEFAULT_MODEL,
    create_portfolio_agent,
    create_insights_agent,
    AgentOrchestrator,
    get_orchestrator,
)

__all__ = [
    # Input models
    "FetchStockDataInput",
    "FetchBenchmarkInput",
    "SimulatePortfolioInput",
    "SimulateSPYInput",
    "CalculateMetricsInput",
    # Output models
    "StockDataOutput",
    "StockDataMetadata",
    "BenchmarkDataOutput",
    "Transaction",
    "SimulationOutput",
    "SPYSimulationOutput",
    "PerformancePoint",
    "MetricsOutput",
    "Insight",
    "Insights",
    "InvestmentSummary",
    # Custom tools
    "fetch_stock_data",
    "fetch_benchmark_data",
    "simulate_portfolio",
    "simulate_spy_investment",
    "calculate_metrics",
    "register_tools",
    # Portfolio management
    "PortfolioHolding",
    "Portfolio",
    "PortfolioManager",
    "get_portfolio_manager",
    # Prompts
    "QUERY_PARSER_SYSTEM_PROMPT",
    "INSIGHTS_GENERATOR_PROMPT",
    "INSIGHTS_GENERATOR_WITH_CONTEXT_PROMPT",
    "format_portfolio_context",
    "get_query_parser_prompt",
    "get_insights_prompt",
    "format_portfolio_data_for_insights",
    "get_insights_prompt_with_context",
    # Session management
    "SessionManager",
    "get_session_manager",
    # Agent configuration
    "DEFAULT_MODEL",
    "create_portfolio_agent",
    "create_insights_agent",
    "AgentOrchestrator",
    "get_orchestrator",
]
