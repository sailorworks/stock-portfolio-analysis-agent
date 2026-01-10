"""Data models for the Stock Portfolio Analysis Agent.

This module defines all Pydantic models for input/output of custom tools
and API responses.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Input Models for Custom Tools
# =============================================================================


class FetchStockDataInput(BaseModel):
    """Input model for the fetch_stock_data custom tool.
    
    Requirements: 1.1, 4.1
    """
    ticker_symbols: List[str] = Field(
        ...,
        description="List of stock ticker symbols (e.g., ['AAPL', 'GOOGL'])"
    )
    start_date: str = Field(
        ...,
        description="Start date for historical data in YYYY-MM-DD format"
    )
    end_date: str = Field(
        ...,
        description="End date for historical data in YYYY-MM-DD format"
    )
    interval: str = Field(
        default="3mo",
        description="Data interval: '1d', '1wk', '1mo', or '3mo'"
    )


class FetchBenchmarkInput(BaseModel):
    """Input model for the fetch_benchmark_data custom tool.
    
    Requirements: 1.2
    """
    start_date: str = Field(
        ...,
        description="Start date matching portfolio in YYYY-MM-DD format"
    )
    end_date: str = Field(
        ...,
        description="End date matching portfolio in YYYY-MM-DD format"
    )
    portfolio_dates: List[str] = Field(
        ...,
        description="List of dates to align SPY data to"
    )


class SimulatePortfolioInput(BaseModel):
    """Input model for the simulate_portfolio custom tool.
    
    Requirements: 5.1, 6.1
    """
    stock_prices: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Historical price data: ticker -> date -> price"
    )
    ticker_amounts: Dict[str, float] = Field(
        ...,
        description="Investment amounts per ticker: ticker -> amount"
    )
    strategy: Literal["single_shot", "dca"] = Field(
        ...,
        description="Investment strategy: 'single_shot' or 'dca'"
    )
    available_cash: float = Field(
        ...,
        description="Starting cash amount for investment"
    )
    dca_interval: Optional[str] = Field(
        default=None,
        description="DCA interval: 'monthly', 'quarterly', etc. Required for DCA strategy"
    )


class CalculateMetricsInput(BaseModel):
    """Input model for the calculate_metrics custom tool.
    
    Requirements: 6.1
    """
    holdings: Dict[str, float] = Field(
        ...,
        description="Current share holdings per ticker: ticker -> shares"
    )
    current_prices: Dict[str, float] = Field(
        ...,
        description="Latest prices per ticker: ticker -> price"
    )
    invested_amounts: Dict[str, float] = Field(
        ...,
        description="Amount invested per ticker: ticker -> amount"
    )
    historical_prices: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Historical price data: ticker -> date -> price"
    )
    spy_prices: Dict[str, float] = Field(
        ...,
        description="SPY benchmark prices: date -> price"
    )
    remaining_cash: float = Field(
        default=0.0,
        description="Remaining cash after purchases"
    )


# =============================================================================
# Output Models
# =============================================================================


class StockDataMetadata(BaseModel):
    """Metadata about fetched stock data."""
    tickers: List[str] = Field(..., description="List of tickers in the data")
    start_date: str = Field(..., description="Actual start date of data")
    end_date: str = Field(..., description="Actual end date of data")
    data_points: int = Field(..., description="Number of data points returned")


class StockDataOutput(BaseModel):
    """Output model for the fetch_stock_data custom tool.
    
    Requirements: 1.4
    """
    prices: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Price data: ticker -> date -> closing price"
    )
    metadata: StockDataMetadata = Field(
        ...,
        description="Metadata about the fetched data"
    )


class BenchmarkDataOutput(BaseModel):
    """Output model for the fetch_benchmark_data custom tool."""
    spy_prices: Dict[str, float] = Field(
        ...,
        description="SPY prices aligned to portfolio dates: date -> price"
    )


class Transaction(BaseModel):
    """A single buy transaction in the portfolio."""
    ticker: str = Field(..., description="Stock ticker symbol")
    date: str = Field(..., description="Transaction date")
    shares: float = Field(..., description="Number of shares purchased")
    price: float = Field(..., description="Price per share")
    cost: float = Field(..., description="Total cost of transaction")


class SimulateSPYInput(BaseModel):
    """Input model for the simulate_spy_investment tool.
    
    Requirements: 7.1, 7.2
    """
    total_amount: float = Field(
        ...,
        description="Total amount to invest in SPY (same as portfolio total)"
    )
    spy_prices: Dict[str, float] = Field(
        ...,
        description="SPY prices: date -> price"
    )
    strategy: Literal["single_shot", "dca"] = Field(
        ...,
        description="Investment strategy: 'single_shot' or 'dca' (same as portfolio)"
    )
    dca_interval: Optional[str] = Field(
        default=None,
        description="DCA interval: 'monthly', 'quarterly', etc. Required for DCA strategy"
    )


class SPYSimulationOutput(BaseModel):
    """Output model for the simulate_spy_investment tool.
    
    Requirements: 7.1, 7.2
    """
    spy_shares: float = Field(
        ...,
        description="Total SPY shares purchased"
    )
    remaining_cash: float = Field(
        ...,
        description="Cash remaining after SPY purchases"
    )
    total_invested: float = Field(
        ...,
        description="Total amount invested in SPY"
    )
    transaction_log: List[Transaction] = Field(
        default_factory=list,
        description="List of all SPY buy transactions"
    )
    value_over_time: Dict[str, float] = Field(
        default_factory=dict,
        description="SPY portfolio value at each date: date -> value"
    )


class SimulationOutput(BaseModel):
    """Output model for the simulate_portfolio custom tool.
    
    Requirements: 5.3
    """
    holdings: Dict[str, float] = Field(
        ...,
        description="Final share holdings per ticker: ticker -> shares"
    )
    remaining_cash: float = Field(
        ...,
        description="Cash remaining after all purchases"
    )
    transaction_log: List[Transaction] = Field(
        default_factory=list,
        description="List of all buy transactions"
    )
    insufficient_funds: List[str] = Field(
        default_factory=list,
        description="List of tickers where purchase failed due to insufficient funds"
    )
    total_invested: float = Field(
        default=0.0,
        description="Total amount invested across all transactions"
    )


class PerformancePoint(BaseModel):
    """A single point in the performance time-series."""
    date: str = Field(..., description="Date of the data point")
    portfolio: float = Field(..., description="Portfolio value on this date")
    spy: float = Field(..., description="SPY benchmark value on this date")


class MetricsOutput(BaseModel):
    """Output model for the calculate_metrics custom tool.
    
    Requirements: 6.1, 6.2, 6.3, 6.4
    """
    total_value: float = Field(
        ...,
        description="Current total portfolio value including cash"
    )
    returns: Dict[str, float] = Field(
        ...,
        description="Absolute returns per ticker: ticker -> return amount"
    )
    percent_returns: Dict[str, float] = Field(
        ...,
        description="Percentage returns per ticker: ticker -> percent"
    )
    allocations: Dict[str, float] = Field(
        ...,
        description="Percentage allocation per ticker: ticker -> percent"
    )
    performance_data: List[PerformancePoint] = Field(
        default_factory=list,
        description="Time-series data for charting"
    )


class Insight(BaseModel):
    """A single investment insight (bull or bear).
    
    Requirements: 8.3
    """
    title: str = Field(..., description="Short title for the insight")
    description: str = Field(..., description="Detailed description of the insight")
    emoji: str = Field(..., description="Emoji representing the insight")


class Insights(BaseModel):
    """Collection of bull and bear insights.
    
    Requirements: 8.3
    """
    bull_insights: List[Insight] = Field(
        default_factory=list,
        description="Bullish (positive) insights about the portfolio"
    )
    bear_insights: List[Insight] = Field(
        default_factory=list,
        description="Bearish (risk) insights about the portfolio"
    )


class InvestmentSummary(BaseModel):
    """Complete investment analysis summary combining all results.
    
    Requirements: 1.4, 5.3, 6.1, 6.2, 6.3, 6.4, 8.3
    """
    holdings: Dict[str, float] = Field(
        ...,
        description="Final share holdings per ticker"
    )
    final_prices: Dict[str, float] = Field(
        ...,
        description="Latest prices per ticker"
    )
    cash: float = Field(
        ...,
        description="Remaining cash after investments"
    )
    returns: Dict[str, float] = Field(
        ...,
        description="Absolute returns per ticker"
    )
    total_value: float = Field(
        ...,
        description="Total portfolio value including cash"
    )
    investment_log: List[str] = Field(
        default_factory=list,
        description="Human-readable log of all transactions"
    )
    percent_allocation: Dict[str, float] = Field(
        ...,
        description="Percentage allocation per ticker"
    )
    percent_return: Dict[str, float] = Field(
        ...,
        description="Percentage return per ticker"
    )
    performance_data: List[PerformancePoint] = Field(
        default_factory=list,
        description="Time-series data for portfolio vs SPY comparison"
    )
    insights: Optional[Insights] = Field(
        default=None,
        description="Bull and bear insights about the portfolio"
    )
