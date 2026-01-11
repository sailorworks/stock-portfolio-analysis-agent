"""Custom tools for the Stock Portfolio Analysis Agent.

This module defines Composio custom tools for fetching stock data,
simulating portfolios, and calculating metrics.

Requirements: 4.4 - Error handling for all tools
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Union

import pandas as pd
import yfinance as yf

from agent.models import (
    BenchmarkDataOutput,
    CalculateMetricsInput,
    FetchBenchmarkInput,
    FetchStockDataInput,
    MetricsOutput,
    PerformancePoint,
    SimulatePortfolioInput,
    SimulateSPYInput,
    SimulationOutput,
    SPYSimulationOutput,
    StockDataMetadata,
    StockDataOutput,
    Transaction,
)
from agent.errors import (
    ErrorCode,
    InvalidDateError,
    InvalidTickerError,
    NoDataAvailableError,
    ToolError,
    ToolExecutionError,
    ValidationError,
    YFinanceAPIError,
    log_tool_error,
    wrap_tool_error,
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
    
    Requirements: 1.1, 4.1, 4.2, 4.4
    
    Args:
        request: Input containing ticker_symbols, start_date, end_date, and interval
        
    Returns:
        StockDataOutput with prices dict and metadata
        
    Raises:
        InvalidDateError: If date format is invalid or range exceeds limits
        InvalidTickerError: If tickers are invalid or no data available
        ValidationError: If input validation fails
        YFinanceAPIError: If Yahoo Finance API fails
    """
    # Validate tickers
    if not request.ticker_symbols:
        raise ValidationError(
            field="ticker_symbols",
            message="At least one ticker symbol is required",
        )
    
    # Parse dates
    try:
        start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError as e:
        raise InvalidDateError(
            error_type="format",
            message=f"Invalid date format. Use YYYY-MM-DD. Error: {e}",
            details={
                "start_date": request.start_date,
                "end_date": request.end_date,
            },
        )
    
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
        raise InvalidDateError(
            error_type="range",
            message="start_date must be before end_date",
            details={
                "start_date": start_dt.strftime("%Y-%m-%d"),
                "end_date": end_dt.strftime("%Y-%m-%d"),
            },
        )
    
    # Validate interval
    valid_intervals = ["1d", "1wk", "1mo", "3mo"]
    if request.interval not in valid_intervals:
        raise ValidationError(
            field="interval",
            message=f"Invalid interval. Must be one of: {valid_intervals}",
            value=request.interval,
        )
    
    # Download data from Yahoo Finance
    tickers = request.ticker_symbols
    
    try:
        # Download stock data
        data = yf.download(
            tickers=tickers,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval=request.interval,
            progress=False,
        )
    except Exception as e:
        raise YFinanceAPIError(
            original_error=e,
            message=f"Failed to download stock data from Yahoo Finance: {str(e)}",
        )
    
    # Handle empty data
    if data.empty:
        raise NoDataAvailableError(
            tickers=tickers,
            date_range=(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")),
            message=f"No data available for tickers: {tickers}",
        )
    
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
    
    # Check if we got data for all tickers - Requirement 4.4
    missing_tickers = [t for t in tickers if t not in prices or not prices[t]]
    if missing_tickers:
        raise InvalidTickerError(
            tickers=missing_tickers,
            message=f"Invalid ticker(s) or no data available: {', '.join(missing_tickers)}",
        )
    
    # Get actual date range from data
    all_dates = []
    for ticker_prices in prices.values():
        all_dates.extend(ticker_prices.keys())
    
    if not all_dates:
        raise NoDataAvailableError(
            tickers=tickers,
            message=f"No price data retrieved for tickers: {tickers}",
        )
    
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


def fetch_stock_data(request: FetchStockDataInput) -> Union[StockDataOutput, ToolError]:
    """Fetch historical closing prices for specified stock tickers.
    
    This is the main entry point for the fetch_stock_data tool.
    It can be used directly or registered with Composio.
    
    Requirements: 1.1, 4.1, 4.2, 4.4
    
    Args:
        request: Input containing ticker_symbols, start_date, end_date, and interval
        
    Returns:
        StockDataOutput with prices dict and metadata, or ToolError on failure
    """
    try:
        return _fetch_stock_data_impl(request)
    except ToolExecutionError as e:
        e.tool_name = "fetch_stock_data"
        log_tool_error("fetch_stock_data", e, {
            "tickers": request.ticker_symbols,
            "start_date": request.start_date,
            "end_date": request.end_date,
        })
        return e.to_error_response()
    except Exception as e:
        wrapped = wrap_tool_error(e, "fetch_stock_data")
        log_tool_error("fetch_stock_data", e, {
            "tickers": request.ticker_symbols,
            "start_date": request.start_date,
            "end_date": request.end_date,
        })
        return wrapped.to_error_response()


def _fetch_benchmark_data_impl(request: FetchBenchmarkInput) -> BenchmarkDataOutput:
    """Fetch SPY benchmark prices aligned to portfolio dates.
    
    This tool downloads SPY (S&P 500 ETF) price data from Yahoo Finance
    and aligns it to the portfolio dates using forward-fill for missing dates.
    
    Requirements: 1.2, 7.4, 4.4
    
    Args:
        request: Input containing start_date, end_date, and portfolio_dates
        
    Returns:
        BenchmarkDataOutput with SPY prices aligned to portfolio dates
        
    Raises:
        InvalidDateError: If dates are invalid
        ValidationError: If portfolio_dates is empty
        NoDataAvailableError: If no SPY data is available
        YFinanceAPIError: If Yahoo Finance API fails
    """
    # Validate portfolio_dates
    if not request.portfolio_dates:
        raise ValidationError(
            field="portfolio_dates",
            message="portfolio_dates cannot be empty",
        )
    
    # Parse dates
    try:
        start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError as e:
        raise InvalidDateError(
            error_type="format",
            message=f"Invalid date format. Use YYYY-MM-DD. Error: {e}",
            details={
                "start_date": request.start_date,
                "end_date": request.end_date,
            },
        )
    
    # Ensure end_date is not in the future
    today = datetime.now()
    if end_dt > today:
        end_dt = today
    
    # Ensure start_date is before end_date
    if start_dt >= end_dt:
        raise InvalidDateError(
            error_type="range",
            message="start_date must be before end_date",
            details={
                "start_date": start_dt.strftime("%Y-%m-%d"),
                "end_date": end_dt.strftime("%Y-%m-%d"),
            },
        )
    
    # Download SPY data at daily interval for accurate alignment
    try:
        spy_data = yf.download(
            tickers="SPY",
            start=start_dt.strftime("%Y-%m-%d"),
            end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),  # Include end_date
            interval="1d",
            progress=False,
        )
    except Exception as e:
        raise YFinanceAPIError(
            original_error=e,
            message=f"Failed to download SPY data from Yahoo Finance: {str(e)}",
        )
    
    # Handle empty data
    if spy_data.empty:
        raise NoDataAvailableError(
            tickers=["SPY"],
            date_range=(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")),
            message="No SPY data available for the specified date range",
        )
    
    # Extract closing prices - handle both single and multi-column formats
    try:
        if isinstance(spy_data.columns, pd.MultiIndex):
            # Multi-level columns (e.g., ('Close', 'SPY'))
            spy_close = spy_data["Close"]["SPY"] if "SPY" in spy_data["Close"].columns else spy_data["Close"].iloc[:, 0]
        elif "Close" in spy_data.columns:
            spy_close = spy_data["Close"]
        else:
            raise NoDataAvailableError(
                tickers=["SPY"],
                message="Unable to extract SPY closing prices from data",
            )
    except Exception as e:
        if isinstance(e, ToolExecutionError):
            raise
        raise NoDataAvailableError(
            tickers=["SPY"],
            message=f"Unable to extract SPY closing prices: {str(e)}",
        )
    
    # Ensure spy_close is a Series with a simple index
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    
    # Convert portfolio_dates to datetime index for alignment
    portfolio_dates_dt = []
    for date_str in request.portfolio_dates:
        try:
            portfolio_dates_dt.append(datetime.strptime(date_str, "%Y-%m-%d"))
        except ValueError:
            raise InvalidDateError(
                error_type="format",
                message=f"Invalid portfolio date format: {date_str}. Use YYYY-MM-DD.",
                details={"invalid_date": date_str},
            )
    
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
        raise NoDataAvailableError(
            tickers=["SPY"],
            message=f"Unable to align SPY prices for dates: {', '.join(missing_dates[:5])}{'...' if len(missing_dates) > 5 else ''}",
        )
    
    return BenchmarkDataOutput(spy_prices=spy_prices)


def fetch_benchmark_data(request: FetchBenchmarkInput) -> Union[BenchmarkDataOutput, ToolError]:
    """Fetch SPY benchmark prices aligned to portfolio dates.
    
    This is the main entry point for the fetch_benchmark_data tool.
    It can be used directly or registered with Composio.
    
    Requirements: 1.2, 7.4, 4.4
    
    Args:
        request: Input containing start_date, end_date, and portfolio_dates
        
    Returns:
        BenchmarkDataOutput with SPY prices aligned to portfolio dates, or ToolError on failure
    """
    try:
        return _fetch_benchmark_data_impl(request)
    except ToolExecutionError as e:
        e.tool_name = "fetch_benchmark_data"
        log_tool_error("fetch_benchmark_data", e, {
            "start_date": request.start_date,
            "end_date": request.end_date,
            "portfolio_dates_count": len(request.portfolio_dates) if request.portfolio_dates else 0,
        })
        return e.to_error_response()
    except Exception as e:
        wrapped = wrap_tool_error(e, "fetch_benchmark_data")
        log_tool_error("fetch_benchmark_data", e, {
            "start_date": request.start_date,
            "end_date": request.end_date,
        })
        return wrapped.to_error_response()


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
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 4.4
    
    Args:
        request: Input containing stock_prices, ticker_amounts, strategy,
                 available_cash, and optional dca_interval
        
    Returns:
        SimulationOutput with holdings, remaining_cash, transaction_log,
        and insufficient_funds list
        
    Raises:
        ValidationError: If strategy is invalid or inputs are missing
    """
    # Validate strategy
    valid_strategies = ["single_shot", "dca"]
    if request.strategy not in valid_strategies:
        raise ValidationError(
            field="strategy",
            message=f"Invalid strategy: {request.strategy}. Must be 'single_shot' or 'dca'",
            value=request.strategy,
        )
    
    # Validate available_cash
    if request.available_cash < 0:
        raise ValidationError(
            field="available_cash",
            message="available_cash cannot be negative",
            value=request.available_cash,
        )
    
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
    
    return SimulationOutput(
        holdings=holdings,
        remaining_cash=remaining_cash,
        transaction_log=transaction_log,
        insufficient_funds=insufficient_funds,
        total_invested=total_invested,
    )


def simulate_portfolio(request: SimulatePortfolioInput) -> Union[SimulationOutput, ToolError]:
    """Simulate buying stocks based on investment strategy.
    
    This is the main entry point for the simulate_portfolio tool.
    It can be used directly or registered with Composio.
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 4.4
    
    Args:
        request: Input containing stock_prices, ticker_amounts, strategy,
                 available_cash, and optional dca_interval
        
    Returns:
        SimulationOutput with holdings, remaining_cash, transaction_log,
        and insufficient_funds list, or ToolError on failure
    """
    try:
        return _simulate_portfolio_impl(request)
    except ToolExecutionError as e:
        e.tool_name = "simulate_portfolio"
        log_tool_error("simulate_portfolio", e, {
            "strategy": request.strategy,
            "tickers": list(request.ticker_amounts.keys()),
            "available_cash": request.available_cash,
        })
        return e.to_error_response()
    except Exception as e:
        wrapped = wrap_tool_error(e, "simulate_portfolio")
        log_tool_error("simulate_portfolio", e, {
            "strategy": request.strategy,
            "tickers": list(request.ticker_amounts.keys()),
        })
        return wrapped.to_error_response()


def _simulate_spy_single_shot(
    total_amount: float,
    spy_prices: Dict[str, float],
) -> tuple[float, float, float, list[Transaction], Dict[str, float]]:
    """Execute single-shot investment strategy for SPY.
    
    Calculates shares purchased at the first available price using integer
    division for whole shares.
    
    Requirements: 7.1, 7.2
    
    Args:
        total_amount: Total amount to invest in SPY
        spy_prices: Dict of date -> price for SPY
        
    Returns:
        Tuple of (shares_purchased, total_invested, remaining_cash, transactions, value_over_time)
    """
    if not spy_prices:
        return 0.0, 0.0, total_amount, [], {}
    
    # Get the first available price (earliest date)
    sorted_dates = sorted(spy_prices.keys())
    first_date = sorted_dates[0]
    first_price = spy_prices[first_date]
    
    if first_price <= 0:
        return 0.0, 0.0, total_amount, [], {}
    
    # Calculate shares using integer division for whole shares
    shares = int(total_amount / first_price)
    
    if shares <= 0:
        return 0.0, 0.0, total_amount, [], {}
    
    # Calculate actual cost
    cost = shares * first_price
    remaining = total_amount - cost
    
    # Create transaction record
    transaction = Transaction(
        ticker="SPY",
        date=first_date,
        shares=float(shares),
        price=first_price,
        cost=cost,
    )
    
    # Calculate value over time
    value_over_time: Dict[str, float] = {}
    for date in sorted_dates:
        price = spy_prices[date]
        value_over_time[date] = shares * price + remaining
    
    return float(shares), cost, remaining, [transaction], value_over_time


def _simulate_spy_dca(
    total_amount: float,
    spy_prices: Dict[str, float],
    dca_interval: str,
) -> tuple[float, float, float, list[Transaction], Dict[str, float]]:
    """Execute DCA (Dollar-Cost Averaging) investment strategy for SPY.
    
    Spreads purchases across intervals, calculating shares at each interval.
    
    Requirements: 7.1, 7.2
    
    Args:
        total_amount: Total amount to invest in SPY
        spy_prices: Dict of date -> price for SPY
        dca_interval: Interval for purchases ('monthly', 'quarterly', etc.)
        
    Returns:
        Tuple of (total_shares, total_invested, remaining_cash, transactions, value_over_time)
    """
    if not spy_prices:
        return 0.0, 0.0, total_amount, [], {}
    
    # Get sorted dates
    sorted_dates = sorted(spy_prices.keys())
    
    # Determine interval step based on dca_interval
    interval_map = {
        "monthly": 1,
        "quarterly": 3,
        "weekly": 1,
        "daily": 1,
    }
    
    step = interval_map.get(dca_interval, 1)
    
    # Select dates at intervals
    purchase_dates = sorted_dates[::step] if step > 0 else sorted_dates
    
    # Ensure we have at least 2 purchase dates for DCA to be meaningful
    if len(purchase_dates) < 2 and len(sorted_dates) >= 2:
        purchase_dates = sorted_dates
    
    if not purchase_dates:
        purchase_dates = sorted_dates
    
    num_purchases = len(purchase_dates)
    if num_purchases == 0:
        return 0.0, 0.0, total_amount, [], {}
    
    # Calculate amount per purchase
    amount_per_purchase = total_amount / num_purchases
    
    total_shares = 0.0
    total_invested = 0.0
    remaining = total_amount
    transactions: list[Transaction] = []
    
    # Track cumulative shares for value calculation
    cumulative_shares = 0.0
    purchase_date_set = set(purchase_dates)
    
    for date in purchase_dates:
        price = spy_prices[date]
        
        if price <= 0:
            continue
        
        # Determine how much we can invest this round
        invest_this_round = min(amount_per_purchase, remaining)
        
        if invest_this_round <= 0:
            continue
        
        # Calculate shares using integer division
        shares = int(invest_this_round / price)
        
        if shares <= 0:
            continue
        
        # Calculate actual cost
        cost = shares * price
        
        if cost > remaining:
            shares = int(remaining / price)
            if shares <= 0:
                continue
            cost = shares * price
        
        # Update totals
        total_shares += shares
        total_invested += cost
        remaining -= cost
        cumulative_shares += shares
        
        # Create transaction record
        transaction = Transaction(
            ticker="SPY",
            date=date,
            shares=float(shares),
            price=price,
            cost=cost,
        )
        transactions.append(transaction)
    
    # Calculate value over time
    # For DCA, we need to track cumulative shares at each date
    value_over_time: Dict[str, float] = {}
    running_shares = 0.0
    running_cash = total_amount
    
    for date in sorted_dates:
        # Check if this is a purchase date
        if date in purchase_date_set:
            # Find the transaction for this date
            for txn in transactions:
                if txn.date == date:
                    running_shares += txn.shares
                    running_cash -= txn.cost
                    break
        
        # Calculate value at this date
        price = spy_prices[date]
        value_over_time[date] = running_shares * price + running_cash
    
    return total_shares, total_invested, remaining, transactions, value_over_time


def _simulate_spy_investment_impl(request: SimulateSPYInput) -> SPYSimulationOutput:
    """Simulate investing in SPY using the same strategy as the portfolio.
    
    This tool simulates SPY investment using either single-shot
    or DCA (Dollar-Cost Averaging) strategy, matching the portfolio strategy.
    
    Requirements: 7.1, 7.2, 4.4
    
    Args:
        request: Input containing total_amount, spy_prices, strategy,
                 and optional dca_interval
        
    Returns:
        SPYSimulationOutput with spy_shares, remaining_cash, total_invested,
        transaction_log, and value_over_time
        
    Raises:
        ValidationError: If strategy is invalid or inputs are missing
    """
    # Validate strategy
    valid_strategies = ["single_shot", "dca"]
    if request.strategy not in valid_strategies:
        raise ValidationError(
            field="strategy",
            message=f"Invalid strategy: {request.strategy}. Must be 'single_shot' or 'dca'",
            value=request.strategy,
        )
    
    # Validate total_amount
    if request.total_amount < 0:
        raise ValidationError(
            field="total_amount",
            message="total_amount cannot be negative",
            value=request.total_amount,
        )
    
    # Validate spy_prices
    if not request.spy_prices:
        raise ValidationError(
            field="spy_prices",
            message="spy_prices cannot be empty",
        )
    
    # Validate DCA interval if using DCA strategy
    if request.strategy == "dca" and not request.dca_interval:
        dca_interval = "monthly"
    else:
        dca_interval = request.dca_interval or "monthly"
    
    if request.strategy == "single_shot":
        shares, invested, remaining, transactions, value_over_time = _simulate_spy_single_shot(
            total_amount=request.total_amount,
            spy_prices=request.spy_prices,
        )
    elif request.strategy == "dca":
        shares, invested, remaining, transactions, value_over_time = _simulate_spy_dca(
            total_amount=request.total_amount,
            spy_prices=request.spy_prices,
            dca_interval=dca_interval,
        )
    
    return SPYSimulationOutput(
        spy_shares=shares,
        remaining_cash=remaining,
        total_invested=invested,
        transaction_log=transactions,
        value_over_time=value_over_time,
    )


def simulate_spy_investment(request: SimulateSPYInput) -> Union[SPYSimulationOutput, ToolError]:
    """Simulate investing in SPY using the same strategy as the portfolio.
    
    This is the main entry point for the simulate_spy_investment tool.
    It can be used directly or registered with Composio.
    
    Requirements: 7.1, 7.2, 4.4
    
    Args:
        request: Input containing total_amount, spy_prices, strategy,
                 and optional dca_interval
        
    Returns:
        SPYSimulationOutput with spy_shares, remaining_cash, total_invested,
        transaction_log, and value_over_time, or ToolError on failure
    """
    try:
        return _simulate_spy_investment_impl(request)
    except ToolExecutionError as e:
        e.tool_name = "simulate_spy_investment"
        log_tool_error("simulate_spy_investment", e, {
            "strategy": request.strategy,
            "total_amount": request.total_amount,
        })
        return e.to_error_response()
    except Exception as e:
        wrapped = wrap_tool_error(e, "simulate_spy_investment")
        log_tool_error("simulate_spy_investment", e, {
            "strategy": request.strategy,
            "total_amount": request.total_amount,
        })
        return wrapped.to_error_response()


def _calculate_portfolio_value(
    holdings: Dict[str, float],
    current_prices: Dict[str, float],
    remaining_cash: float,
) -> float:
    """Calculate total portfolio value.
    
    Computes the sum of (holdings × current_price) for each ticker,
    plus remaining cash.
    
    Requirements: 6.1
    
    Args:
        holdings: Dict mapping tickers to shares owned
        current_prices: Dict mapping tickers to current prices
        remaining_cash: Cash remaining after purchases
        
    Returns:
        Total portfolio value
    """
    total_value = remaining_cash
    
    for ticker, shares in holdings.items():
        if ticker in current_prices and shares > 0:
            total_value += shares * current_prices[ticker]
    
    return total_value


def _calculate_returns(
    holdings: Dict[str, float],
    current_prices: Dict[str, float],
    invested_amounts: Dict[str, float],
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Calculate absolute and percentage returns per ticker.
    
    Absolute return = (holdings × current_price) - invested_amount
    Percentage return = ((holdings × current_price - invested) / invested) × 100
    
    Requirements: 6.2, 6.3
    
    Args:
        holdings: Dict mapping tickers to shares owned
        current_prices: Dict mapping tickers to current prices
        invested_amounts: Dict mapping tickers to amounts invested
        
    Returns:
        Tuple of (absolute_returns, percent_returns) dicts
    """
    absolute_returns: Dict[str, float] = {}
    percent_returns: Dict[str, float] = {}
    
    for ticker in holdings.keys():
        shares = holdings.get(ticker, 0.0)
        current_price = current_prices.get(ticker, 0.0)
        invested = invested_amounts.get(ticker, 0.0)
        
        # Calculate current value for this ticker
        current_value = shares * current_price
        
        # Calculate absolute return
        absolute_return = current_value - invested
        absolute_returns[ticker] = absolute_return
        
        # Calculate percentage return (avoid division by zero)
        if invested > 0:
            percent_return = ((current_value - invested) / invested) * 100
        else:
            percent_return = 0.0
        
        percent_returns[ticker] = percent_return
    
    return absolute_returns, percent_returns


def _calculate_allocations(
    invested_amounts: Dict[str, float],
) -> Dict[str, float]:
    """Calculate percentage allocation per ticker.
    
    Allocation = (invested_per_ticker / total_invested) × 100
    Ensures allocations sum to 100% (within floating-point tolerance).
    
    Requirements: 6.4
    
    Args:
        invested_amounts: Dict mapping tickers to amounts invested
        
    Returns:
        Dict mapping tickers to allocation percentages
    """
    allocations: Dict[str, float] = {}
    
    # Calculate total invested
    total_invested = sum(invested_amounts.values())
    
    if total_invested <= 0:
        # No investments, return 0% for all tickers
        return {ticker: 0.0 for ticker in invested_amounts.keys()}
    
    # Calculate allocation for each ticker
    for ticker, invested in invested_amounts.items():
        allocation = (invested / total_invested) * 100
        allocations[ticker] = allocation
    
    return allocations


def _generate_performance_data(
    holdings: Dict[str, float],
    historical_prices: Dict[str, Dict[str, float]],
    spy_prices: Dict[str, float],
    remaining_cash: float,
    invested_amounts: Dict[str, float],
) -> list[PerformancePoint]:
    """Generate time-series performance data for charting.
    
    Creates a list of data points with portfolio value and SPY value
    for each date in the historical data.
    
    Requirements: 6.5, 7.3
    
    Args:
        holdings: Dict mapping tickers to shares owned
        historical_prices: Dict mapping tickers to date->price dicts
        spy_prices: Dict mapping dates to SPY prices
        remaining_cash: Cash remaining after purchases
        invested_amounts: Dict mapping tickers to amounts invested
        
    Returns:
        List of PerformancePoint objects for charting
    """
    performance_data: list[PerformancePoint] = []
    
    # Get all unique dates from historical prices
    all_dates: set[str] = set()
    for ticker_prices in historical_prices.values():
        all_dates.update(ticker_prices.keys())
    
    # Also include dates from SPY prices
    all_dates.update(spy_prices.keys())
    
    if not all_dates:
        return performance_data
    
    # Sort dates chronologically
    sorted_dates = sorted(all_dates)
    
    # Calculate total invested for SPY comparison
    total_invested = sum(invested_amounts.values())
    
    # Get the first SPY price for calculating SPY shares
    first_spy_date = sorted_dates[0] if sorted_dates else None
    first_spy_price = spy_prices.get(first_spy_date, 0.0) if first_spy_date else 0.0
    
    # Calculate equivalent SPY shares (as if we invested the same amount in SPY)
    spy_shares = total_invested / first_spy_price if first_spy_price > 0 else 0.0
    
    # Generate performance point for each date
    for date in sorted_dates:
        # Calculate portfolio value on this date
        portfolio_value = remaining_cash
        for ticker, shares in holdings.items():
            ticker_prices = historical_prices.get(ticker, {})
            # Use the price on this date, or find the most recent price
            if date in ticker_prices:
                price = ticker_prices[date]
            else:
                # Find the most recent price before this date
                available_dates = [d for d in ticker_prices.keys() if d <= date]
                if available_dates:
                    most_recent = max(available_dates)
                    price = ticker_prices[most_recent]
                else:
                    # No price available yet, use 0
                    price = 0.0
            
            portfolio_value += shares * price
        
        # Calculate SPY value on this date
        spy_price = spy_prices.get(date, 0.0)
        if spy_price == 0.0:
            # Find the most recent SPY price
            available_spy_dates = [d for d in spy_prices.keys() if d <= date]
            if available_spy_dates:
                most_recent_spy = max(available_spy_dates)
                spy_price = spy_prices[most_recent_spy]
        
        spy_value = spy_shares * spy_price + remaining_cash
        
        performance_data.append(PerformancePoint(
            date=date,
            portfolio=portfolio_value,
            spy=spy_value,
        ))
    
    return performance_data


def _calculate_metrics_impl(request: CalculateMetricsInput) -> MetricsOutput:
    """Calculate portfolio performance metrics.
    
    This tool calculates comprehensive portfolio metrics including:
    - Total portfolio value (holdings × prices + cash)
    - Absolute and percentage returns per ticker
    - Allocation percentages per ticker
    - Time-series performance data for charting
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 7.3
    
    Args:
        request: Input containing holdings, current_prices, invested_amounts,
                 historical_prices, spy_prices, and remaining_cash
        
    Returns:
        MetricsOutput with total_value, returns, percent_returns,
        allocations, and performance_data
    """
    # 7.1: Calculate portfolio value
    total_value = _calculate_portfolio_value(
        holdings=request.holdings,
        current_prices=request.current_prices,
        remaining_cash=request.remaining_cash,
    )
    
    # 7.2: Calculate returns
    returns, percent_returns = _calculate_returns(
        holdings=request.holdings,
        current_prices=request.current_prices,
        invested_amounts=request.invested_amounts,
    )
    
    # 7.3: Calculate allocations
    allocations = _calculate_allocations(
        invested_amounts=request.invested_amounts,
    )
    
    # 7.4: Generate performance data
    performance_data = _generate_performance_data(
        holdings=request.holdings,
        historical_prices=request.historical_prices,
        spy_prices=request.spy_prices,
        remaining_cash=request.remaining_cash,
        invested_amounts=request.invested_amounts,
    )
    
    return MetricsOutput(
        total_value=total_value,
        returns=returns,
        percent_returns=percent_returns,
        allocations=allocations,
        performance_data=performance_data,
    )


def calculate_metrics(request: CalculateMetricsInput) -> Union[MetricsOutput, ToolError]:
    """Calculate portfolio performance metrics.
    
    This is the main entry point for the calculate_metrics tool.
    It can be used directly or registered with Composio.
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 7.3, 4.4
    
    Args:
        request: Input containing holdings, current_prices, invested_amounts,
                 historical_prices, spy_prices, and remaining_cash
        
    Returns:
        MetricsOutput with total_value, returns, percent_returns,
        allocations, and performance_data, or ToolError on failure
    """
    try:
        return _calculate_metrics_impl(request)
    except ToolExecutionError as e:
        e.tool_name = "calculate_metrics"
        log_tool_error("calculate_metrics", e, {
            "tickers": list(request.holdings.keys()),
        })
        return e.to_error_response()
    except Exception as e:
        wrapped = wrap_tool_error(e, "calculate_metrics")
        log_tool_error("calculate_metrics", e, {
            "tickers": list(request.holdings.keys()),
        })
        return wrapped.to_error_response()


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
    
    # Register calculate_metrics as a Composio custom tool
    @composio.tools.custom_tool
    def calculate_metrics_tool(request: CalculateMetricsInput) -> MetricsOutput:
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
    
    return {
        "fetch_stock_data": fetch_stock_data_tool,
        "fetch_benchmark_data": fetch_benchmark_data_tool,
        "simulate_portfolio": simulate_portfolio_tool,
        "calculate_metrics": calculate_metrics_tool,
    }
