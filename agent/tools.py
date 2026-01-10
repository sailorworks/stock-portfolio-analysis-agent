"""Custom tools for the Stock Portfolio Analysis Agent.

This module defines Composio custom tools for fetching stock data,
simulating portfolios, and calculating metrics.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from agent.models import (
    BenchmarkDataOutput,
    FetchBenchmarkInput,
    FetchStockDataInput,
    SimulatePortfolioInput,
    SimulationOutput,
    StockDataMetadata,
    StockDataOutput,
    Transaction,
)

# Lazy initialization of Composio client
_composio: Optional["Composio"] = None


def get_composio():
    """Get or create the Composio client instance."""
    global _composio
    if _composio is None:
        from composio import Composio
        _composio = Composio()
    return _composio


def _fetch_stock_data_impl(request: FetchStockDataInput) -> StockDataOutput:
    """Fetch historical closing prices for specified stock tickers.
    
    This tool downloads historical stock price data from Yahoo Finance
    for the specified tickers and date range.
    
    Requirements: 1.1, 4.1, 4.2
    
    Args:
        request: Input containing ticker_symbols, start_date, end_date, and interval
        
    Returns:
        StockDataOutput with prices dict and metadata
        
    Raises:
        ValueError: If date range exceeds 4 years or tickers are invalid
    """
    # Parse dates
    try:
        start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")
    
    # Validate date range (max 4 years per requirement 4.2)
    max_range = timedelta(days=4 * 365)
    if end_dt - start_dt > max_range:
        # Truncate to 4 years from end_date
        start_dt = end_dt - max_range
    
    # Ensure end_date is not in the future
    today = datetime.now()
    if end_dt > today:
        end_dt = today
    
    # Ensure start_date is before end_date
    if start_dt >= end_dt:
        raise ValueError("start_date must be before end_date")
    
    # Validate interval
    valid_intervals = ["1d", "1wk", "1mo", "3mo"]
    if request.interval not in valid_intervals:
        raise ValueError(f"Invalid interval. Must be one of: {valid_intervals}")
    
    # Download data from Yahoo Finance
    tickers = request.ticker_symbols
    if not tickers:
        raise ValueError("At least one ticker symbol is required")
    
    # Download stock data
    data = yf.download(
        tickers=tickers,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval=request.interval,
        progress=False,
    )
    
    # Handle empty data
    if data.empty:
        raise ValueError(f"No data available for tickers: {tickers}")
    
    # Extract closing prices
    if "Close" in data.columns:
        close_data = data["Close"]
    elif len(tickers) == 1:
        # Single ticker returns flat columns
        close_data = data["Close"] if "Close" in data.columns else data
    else:
        close_data = data["Close"]
    
    # Convert to dict format: ticker -> date -> price
    prices: Dict[str, Dict[str, float]] = {}
    
    if isinstance(close_data, pd.Series):
        # Single ticker case
        ticker = tickers[0]
        prices[ticker] = {}
        for date_idx, price in close_data.items():
            if pd.notna(price):
                date_str = date_idx.strftime("%Y-%m-%d")
                prices[ticker][date_str] = float(price)
    else:
        # Multiple tickers case
        for ticker in tickers:
            prices[ticker] = {}
            if ticker in close_data.columns:
                for date_idx, price in close_data[ticker].items():
                    if pd.notna(price):
                        date_str = date_idx.strftime("%Y-%m-%d")
                        prices[ticker][date_str] = float(price)
    
    # Check if we got data for all tickers
    missing_tickers = [t for t in tickers if t not in prices or not prices[t]]
    if missing_tickers:
        raise ValueError(f"Invalid ticker(s) or no data available: {missing_tickers}")
    
    # Get actual date range from data
    all_dates = []
    for ticker_prices in prices.values():
        all_dates.extend(ticker_prices.keys())
    
    if not all_dates:
        raise ValueError(f"No price data retrieved for tickers: {tickers}")
    
    sorted_dates = sorted(set(all_dates))
    actual_start = sorted_dates[0]
    actual_end = sorted_dates[-1]
    
    # Create metadata
    metadata = StockDataMetadata(
        tickers=tickers,
        start_date=actual_start,
        end_date=actual_end,
        data_points=len(sorted_dates),
    )
    
    return StockDataOutput(prices=prices, metadata=metadata)


def fetch_stock_data(request: FetchStockDataInput) -> StockDataOutput:
    """Fetch historical closing prices for specified stock tickers.
    
    This is the main entry point for the fetch_stock_data tool.
    It can be used directly or registered with Composio.
    
    Requirements: 1.1, 4.1, 4.2
    
    Args:
        request: Input containing ticker_symbols, start_date, end_date, and interval
        
    Returns:
        StockDataOutput with prices dict and metadata
    """
    return _fetch_stock_data_impl(request)


def _fetch_benchmark_data_impl(request: FetchBenchmarkInput) -> BenchmarkDataOutput:
    """Fetch SPY benchmark prices aligned to portfolio dates.
    
    This tool downloads SPY (S&P 500 ETF) price data from Yahoo Finance
    and aligns it to the portfolio dates using forward-fill for missing dates.
    
    Requirements: 1.2, 7.4
    
    Args:
        request: Input containing start_date, end_date, and portfolio_dates
        
    Returns:
        BenchmarkDataOutput with SPY prices aligned to portfolio dates
        
    Raises:
        ValueError: If dates are invalid or no SPY data is available
    """
    # Parse dates
    try:
        start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")
    
    # Ensure end_date is not in the future
    today = datetime.now()
    if end_dt > today:
        end_dt = today
    
    # Ensure start_date is before end_date
    if start_dt >= end_dt:
        raise ValueError("start_date must be before end_date")
    
    # Validate portfolio_dates
    if not request.portfolio_dates:
        raise ValueError("portfolio_dates cannot be empty")
    
    # Download SPY data at daily interval for accurate alignment
    spy_data = yf.download(
        tickers="SPY",
        start=start_dt.strftime("%Y-%m-%d"),
        end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),  # Include end_date
        interval="1d",
        progress=False,
    )
    
    # Handle empty data
    if spy_data.empty:
        raise ValueError("No SPY data available for the specified date range")
    
    # Extract closing prices - handle both single and multi-column formats
    if isinstance(spy_data.columns, pd.MultiIndex):
        # Multi-level columns (e.g., ('Close', 'SPY'))
        spy_close = spy_data["Close"]["SPY"] if "SPY" in spy_data["Close"].columns else spy_data["Close"].iloc[:, 0]
    elif "Close" in spy_data.columns:
        spy_close = spy_data["Close"]
    else:
        raise ValueError("Unable to extract SPY closing prices")
    
    # Ensure spy_close is a Series with a simple index
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    
    # Convert portfolio_dates to datetime index for alignment
    portfolio_dates_dt = []
    for date_str in request.portfolio_dates:
        try:
            portfolio_dates_dt.append(datetime.strptime(date_str, "%Y-%m-%d"))
        except ValueError:
            raise ValueError(f"Invalid portfolio date format: {date_str}. Use YYYY-MM-DD.")
    
    # Create a pandas DatetimeIndex from portfolio dates
    portfolio_index = pd.DatetimeIndex(portfolio_dates_dt)
    
    # Reindex SPY data to portfolio dates using forward-fill
    # This aligns SPY prices to the exact dates in the portfolio
    spy_aligned = spy_close.reindex(portfolio_index, method="ffill")
    
    # Handle any remaining NaN values (dates before SPY data starts)
    # Use backward fill for dates before the first SPY data point
    spy_aligned = spy_aligned.bfill()
    
    # Convert to dict format: date -> price
    spy_prices: Dict[str, float] = {}
    for date_idx in spy_aligned.index:
        price = spy_aligned.loc[date_idx]
        # Handle scalar values properly
        if hasattr(price, 'item'):
            price_val = price.item()
        else:
            price_val = float(price)
        
        if pd.notna(price_val):
            date_str = date_idx.strftime("%Y-%m-%d")
            spy_prices[date_str] = float(price_val)
    
    # Verify we have prices for all portfolio dates
    missing_dates = [d for d in request.portfolio_dates if d not in spy_prices]
    if missing_dates:
        raise ValueError(f"Unable to align SPY prices for dates: {missing_dates}")
    
    return BenchmarkDataOutput(spy_prices=spy_prices)


def fetch_benchmark_data(request: FetchBenchmarkInput) -> BenchmarkDataOutput:
    """Fetch SPY benchmark prices aligned to portfolio dates.
    
    This is the main entry point for the fetch_benchmark_data tool.
    It can be used directly or registered with Composio.
    
    Requirements: 1.2, 7.4
    
    Args:
        request: Input containing start_date, end_date, and portfolio_dates
        
    Returns:
        BenchmarkDataOutput with SPY prices aligned to portfolio dates
    """
    return _fetch_benchmark_data_impl(request)


def _simulate_single_shot(
    ticker: str,
    amount: float,
    prices: Dict[str, float],
    available_cash: float,
) -> tuple[float, float, float, Optional[Transaction]]:
    """Execute single-shot investment strategy for a single ticker.
    
    Calculates shares purchased at the first available price using integer
    division for whole shares.
    
    Requirements: 5.1, 5.4
    
    Args:
        ticker: Stock ticker symbol
        amount: Amount to invest in this ticker
        prices: Dict of date -> price for this ticker
        available_cash: Current available cash
        
    Returns:
        Tuple of (shares_purchased, cost, remaining_cash, transaction)
    """
    if not prices:
        return 0.0, 0.0, available_cash, None
    
    # Get the first available price (earliest date)
    sorted_dates = sorted(prices.keys())
    first_date = sorted_dates[0]
    first_price = prices[first_date]
    
    if first_price <= 0:
        return 0.0, 0.0, available_cash, None
    
    # Determine how much we can actually invest
    invest_amount = min(amount, available_cash)
    
    if invest_amount <= 0:
        return 0.0, 0.0, available_cash, None
    
    # Calculate shares using integer division for whole shares (Requirement 5.4)
    shares = int(invest_amount / first_price)
    
    if shares <= 0:
        return 0.0, 0.0, available_cash, None
    
    # Calculate actual cost
    cost = shares * first_price
    remaining = available_cash - cost
    
    # Create transaction record
    transaction = Transaction(
        ticker=ticker,
        date=first_date,
        shares=float(shares),
        price=first_price,
        cost=cost,
    )
    
    return float(shares), cost, remaining, transaction


def _simulate_dca(
    ticker: str,
    amount: float,
    prices: Dict[str, float],
    available_cash: float,
    dca_interval: str,
) -> tuple[float, float, float, list[Transaction], bool]:
    """Execute DCA (Dollar-Cost Averaging) investment strategy for a single ticker.
    
    Spreads purchases across intervals, calculating shares at each interval.
    
    Requirements: 5.2, 5.3
    
    Args:
        ticker: Stock ticker symbol
        amount: Total amount to invest in this ticker
        prices: Dict of date -> price for this ticker
        available_cash: Current available cash
        dca_interval: Interval for purchases ('monthly', 'quarterly', etc.)
        
    Returns:
        Tuple of (total_shares, total_cost, remaining_cash, transactions, had_insufficient_funds)
    """
    if not prices:
        return 0.0, 0.0, available_cash, [], False
    
    # Get sorted dates
    sorted_dates = sorted(prices.keys())
    
    # Determine interval step based on dca_interval
    # The step determines how many data points to skip between purchases
    interval_map = {
        "monthly": 1,      # Every data point (assuming monthly data)
        "quarterly": 3,    # Every 3 data points
        "weekly": 1,       # Every data point (assuming weekly data)
        "daily": 1,        # Every data point
    }
    
    step = interval_map.get(dca_interval, 1)
    
    # Select dates at intervals
    purchase_dates = sorted_dates[::step] if step > 0 else sorted_dates
    
    # Ensure we have at least 2 purchase dates for DCA to be meaningful
    # If step is too large, use all available dates
    if len(purchase_dates) < 2 and len(sorted_dates) >= 2:
        purchase_dates = sorted_dates
    
    # If we still have no purchase dates, use all dates
    if not purchase_dates:
        purchase_dates = sorted_dates
    
    num_purchases = len(purchase_dates)
    if num_purchases == 0:
        return 0.0, 0.0, available_cash, [], False
    
    # Calculate amount per purchase
    amount_per_purchase = amount / num_purchases
    
    total_shares = 0.0
    total_cost = 0.0
    remaining = available_cash
    transactions: list[Transaction] = []
    had_insufficient_funds = False
    
    for date in purchase_dates:
        price = prices[date]
        
        if price <= 0:
            continue
        
        # Determine how much we can invest this round
        invest_this_round = min(amount_per_purchase, remaining)
        
        if invest_this_round <= 0:
            had_insufficient_funds = True
            continue
        
        # Calculate shares using integer division (Requirement 5.4)
        shares = int(invest_this_round / price)
        
        if shares <= 0:
            # Price too high for available amount
            continue
        
        # Calculate actual cost
        cost = shares * price
        
        # Check if we have enough cash
        if cost > remaining:
            had_insufficient_funds = True
            # Try to buy fewer shares
            shares = int(remaining / price)
            if shares <= 0:
                continue
            cost = shares * price
        
        # Update totals
        total_shares += shares
        total_cost += cost
        remaining -= cost
        
        # Create transaction record
        transaction = Transaction(
            ticker=ticker,
            date=date,
            shares=float(shares),
            price=price,
            cost=cost,
        )
        transactions.append(transaction)
    
    return total_shares, total_cost, remaining, transactions, had_insufficient_funds


def _simulate_portfolio_impl(request: SimulatePortfolioInput) -> SimulationOutput:
    """Simulate buying stocks based on investment strategy.
    
    This tool simulates portfolio investment using either single-shot
    or DCA (Dollar-Cost Averaging) strategy.
    
    Requirements: 5.1, 5.2, 5.3, 5.4
    
    Args:
        request: Input containing stock_prices, ticker_amounts, strategy,
                 available_cash, and optional dca_interval
        
    Returns:
        SimulationOutput with holdings, remaining_cash, transaction_log,
        and insufficient_funds list
    """
    holdings: Dict[str, float] = {}
    transaction_log: list[Transaction] = []
    insufficient_funds: list[str] = []
    remaining_cash = request.available_cash
    total_invested = 0.0
    
    # Validate DCA interval if using DCA strategy
    if request.strategy == "dca" and not request.dca_interval:
        # Default to monthly if not specified
        dca_interval = "monthly"
    else:
        dca_interval = request.dca_interval or "monthly"
    
    # Process each ticker
    for ticker, amount in request.ticker_amounts.items():
        # Get prices for this ticker
        ticker_prices = request.stock_prices.get(ticker, {})
        
        if not ticker_prices:
            insufficient_funds.append(ticker)
            holdings[ticker] = 0.0
            continue
        
        if request.strategy == "single_shot":
            # Single-shot strategy: buy all at first available price
            shares, cost, remaining_cash, transaction = _simulate_single_shot(
                ticker=ticker,
                amount=amount,
                prices=ticker_prices,
                available_cash=remaining_cash,
            )
            
            holdings[ticker] = shares
            total_invested += cost
            
            if transaction:
                transaction_log.append(transaction)
            
            if shares == 0 and amount > 0:
                insufficient_funds.append(ticker)
                
        elif request.strategy == "dca":
            # DCA strategy: spread purchases across intervals
            shares, cost, remaining_cash, transactions, had_insufficient = _simulate_dca(
                ticker=ticker,
                amount=amount,
                prices=ticker_prices,
                available_cash=remaining_cash,
                dca_interval=dca_interval,
            )
            
            holdings[ticker] = shares
            total_invested += cost
            transaction_log.extend(transactions)
            
            if had_insufficient or (shares == 0 and amount > 0):
                insufficient_funds.append(ticker)
        else:
            raise ValueError(f"Invalid strategy: {request.strategy}. Must be 'single_shot' or 'dca'")
    
    return SimulationOutput(
        holdings=holdings,
        remaining_cash=remaining_cash,
        transaction_log=transaction_log,
        insufficient_funds=insufficient_funds,
        total_invested=total_invested,
    )


def simulate_portfolio(request: SimulatePortfolioInput) -> SimulationOutput:
    """Simulate buying stocks based on investment strategy.
    
    This is the main entry point for the simulate_portfolio tool.
    It can be used directly or registered with Composio.
    
    Requirements: 5.1, 5.2, 5.3, 5.4
    
    Args:
        request: Input containing stock_prices, ticker_amounts, strategy,
                 available_cash, and optional dca_interval
        
    Returns:
        SimulationOutput with holdings, remaining_cash, transaction_log,
        and insufficient_funds list
    """
    return _simulate_portfolio_impl(request)


def register_tools():
    """Register all custom tools with Composio.
    
    Call this function after setting up the Composio API key
    to register the custom tools with the Tool Router.
    """
    composio = get_composio()
    
    # Register fetch_stock_data as a Composio custom tool
    @composio.tools.custom_tool
    def fetch_stock_data_tool(request: FetchStockDataInput) -> StockDataOutput:
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
    def fetch_benchmark_data_tool(request: FetchBenchmarkInput) -> BenchmarkDataOutput:
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
    def simulate_portfolio_tool(request: SimulatePortfolioInput) -> SimulationOutput:
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
    
    return {
        "fetch_stock_data": fetch_stock_data_tool,
        "fetch_benchmark_data": fetch_benchmark_data_tool,
        "simulate_portfolio": simulate_portfolio_tool,
    }
