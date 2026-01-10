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
]
