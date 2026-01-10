"""Composio Stock Portfolio Analysis Agent."""

from agent.models import (
    # Input models
    FetchStockDataInput,
    FetchBenchmarkInput,
    SimulatePortfolioInput,
    CalculateMetricsInput,
    # Output models
    StockDataOutput,
    StockDataMetadata,
    BenchmarkDataOutput,
    Transaction,
    SimulationOutput,
    PerformancePoint,
    MetricsOutput,
    Insight,
    Insights,
    InvestmentSummary,
)

from agent.tools import fetch_stock_data, fetch_benchmark_data, simulate_portfolio, register_tools

__all__ = [
    # Input models
    "FetchStockDataInput",
    "FetchBenchmarkInput",
    "SimulatePortfolioInput",
    "CalculateMetricsInput",
    # Output models
    "StockDataOutput",
    "StockDataMetadata",
    "BenchmarkDataOutput",
    "Transaction",
    "SimulationOutput",
    "PerformancePoint",
    "MetricsOutput",
    "Insight",
    "Insights",
    "InvestmentSummary",
    # Custom tools
    "fetch_stock_data",
    "fetch_benchmark_data",
    "simulate_portfolio",
    "register_tools",
]
