"""Tests for custom tools in the Stock Portfolio Analysis Agent.

This module contains tests to verify that the custom tools work correctly:
- fetch_stock_data
- fetch_benchmark_data
- simulate_portfolio
- calculate_metrics
"""

import pytest
from datetime import datetime, timedelta

from agent.models import (
    FetchStockDataInput,
    FetchBenchmarkInput,
    SimulatePortfolioInput,
    SimulateSPYInput,
    CalculateMetricsInput,
)
from agent.tools import (
    fetch_stock_data,
    fetch_benchmark_data,
    simulate_portfolio,
    simulate_spy_investment,
    calculate_metrics,
)


class TestFetchStockData:
    """Tests for the fetch_stock_data tool."""

    def test_fetch_single_ticker(self):
        """Test fetching data for a single ticker."""
        # Use a date range that's definitely in the past
        end_date = datetime.now() - timedelta(days=7)
        start_date = end_date - timedelta(days=90)
        
        request = FetchStockDataInput(
            ticker_symbols=["AAPL"],
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval="1mo",
        )
        
        result = fetch_stock_data(request)
        
        # Verify structure
        assert result.prices is not None
        assert "AAPL" in result.prices
        assert len(result.prices["AAPL"]) > 0
        
        # Verify metadata
        assert result.metadata.tickers == ["AAPL"]
        assert result.metadata.data_points > 0
        
        # Verify all prices are positive numbers
        for date, price in result.prices["AAPL"].items():
            assert isinstance(price, float)
            assert price > 0

    def test_fetch_multiple_tickers(self):
        """Test fetching data for multiple tickers."""
        end_date = datetime.now() - timedelta(days=7)
        start_date = end_date - timedelta(days=180)
        
        request = FetchStockDataInput(
            ticker_symbols=["AAPL", "MSFT"],
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval="1mo",
        )
        
        result = fetch_stock_data(request)
        
        # Verify both tickers are present
        assert "AAPL" in result.prices
        assert "MSFT" in result.prices
        assert len(result.prices["AAPL"]) > 0
        assert len(result.prices["MSFT"]) > 0

    def test_invalid_ticker_raises_error(self):
        """Test that invalid ticker raises ValueError."""
        end_date = datetime.now() - timedelta(days=7)
        start_date = end_date - timedelta(days=30)
        
        request = FetchStockDataInput(
            ticker_symbols=["INVALIDTICKER123XYZ"],
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval="1d",
        )
        
        with pytest.raises(ValueError) as exc_info:
            fetch_stock_data(request)
        
        assert "Invalid ticker" in str(exc_info.value) or "No data" in str(exc_info.value)

    def test_date_range_validation(self):
        """Test that date range exceeding 4 years is truncated."""
        end_date = datetime.now() - timedelta(days=7)
        start_date = end_date - timedelta(days=5 * 365)  # 5 years
        
        request = FetchStockDataInput(
            ticker_symbols=["AAPL"],
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval="3mo",
        )
        
        # Should not raise - date range should be truncated
        result = fetch_stock_data(request)
        assert result.prices is not None


class TestFetchBenchmarkData:
    """Tests for the fetch_benchmark_data tool."""

    def test_fetch_spy_data(self):
        """Test fetching SPY benchmark data."""
        end_date = datetime.now() - timedelta(days=7)
        start_date = end_date - timedelta(days=90)
        
        # Create some portfolio dates
        portfolio_dates = [
            (start_date + timedelta(days=i * 30)).strftime("%Y-%m-%d")
            for i in range(3)
        ]
        
        request = FetchBenchmarkInput(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            portfolio_dates=portfolio_dates,
        )
        
        result = fetch_benchmark_data(request)
        
        # Verify SPY prices are returned
        assert result.spy_prices is not None
        assert len(result.spy_prices) > 0
        
        # Verify all portfolio dates have prices
        for date in portfolio_dates:
            assert date in result.spy_prices
            assert result.spy_prices[date] > 0

    def test_spy_date_alignment(self):
        """Test that SPY prices are aligned to portfolio dates."""
        end_date = datetime.now() - timedelta(days=7)
        start_date = end_date - timedelta(days=60)
        
        # Use specific dates
        portfolio_dates = [
            start_date.strftime("%Y-%m-%d"),
            (start_date + timedelta(days=30)).strftime("%Y-%m-%d"),
        ]
        
        request = FetchBenchmarkInput(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            portfolio_dates=portfolio_dates,
        )
        
        result = fetch_benchmark_data(request)
        
        # All portfolio dates should have SPY prices
        for date in portfolio_dates:
            assert date in result.spy_prices


class TestSimulatePortfolio:
    """Tests for the simulate_portfolio tool."""

    def test_single_shot_strategy(self):
        """Test single-shot investment strategy."""
        # Create mock price data
        stock_prices = {
            "AAPL": {
                "2024-01-01": 100.0,
                "2024-02-01": 110.0,
                "2024-03-01": 105.0,
            }
        }
        
        request = SimulatePortfolioInput(
            stock_prices=stock_prices,
            ticker_amounts={"AAPL": 1000.0},
            strategy="single_shot",
            available_cash=1000.0,
        )
        
        result = simulate_portfolio(request)
        
        # Verify holdings
        assert "AAPL" in result.holdings
        assert result.holdings["AAPL"] == 10.0  # 1000 / 100 = 10 shares
        
        # Verify cash conservation
        assert result.remaining_cash == 0.0  # 10 * 100 = 1000
        assert result.total_invested == 1000.0
        
        # Verify transaction log
        assert len(result.transaction_log) == 1
        assert result.transaction_log[0].ticker == "AAPL"
        assert result.transaction_log[0].shares == 10.0
        assert result.transaction_log[0].price == 100.0

    def test_dca_strategy(self):
        """Test DCA investment strategy."""
        # Create mock price data with multiple dates
        stock_prices = {
            "AAPL": {
                "2024-01-01": 100.0,
                "2024-02-01": 110.0,
                "2024-03-01": 90.0,
                "2024-04-01": 105.0,
            }
        }
        
        request = SimulatePortfolioInput(
            stock_prices=stock_prices,
            ticker_amounts={"AAPL": 1000.0},
            strategy="dca",
            available_cash=1000.0,
            dca_interval="monthly",
        )
        
        result = simulate_portfolio(request)
        
        # Verify holdings exist
        assert "AAPL" in result.holdings
        assert result.holdings["AAPL"] > 0
        
        # Verify multiple transactions (DCA spreads purchases)
        assert len(result.transaction_log) > 1
        
        # Verify cash conservation invariant
        total_cost = sum(t.cost for t in result.transaction_log)
        assert abs(result.remaining_cash + total_cost - 1000.0) < 0.01

    def test_cash_conservation_invariant(self):
        """Test that cash is conserved: initial = remaining + invested."""
        stock_prices = {
            "AAPL": {"2024-01-01": 150.0},
            "MSFT": {"2024-01-01": 300.0},
        }
        
        initial_cash = 5000.0
        
        request = SimulatePortfolioInput(
            stock_prices=stock_prices,
            ticker_amounts={"AAPL": 2000.0, "MSFT": 2000.0},
            strategy="single_shot",
            available_cash=initial_cash,
        )
        
        result = simulate_portfolio(request)
        
        # Cash conservation: initial = remaining + total_invested
        total_cost = sum(t.cost for t in result.transaction_log)
        assert abs(result.remaining_cash + total_cost - initial_cash) < 0.01

    def test_insufficient_funds_handling(self):
        """Test handling when funds are insufficient."""
        stock_prices = {
            "AAPL": {"2024-01-01": 1000.0},  # Very expensive stock
        }
        
        request = SimulatePortfolioInput(
            stock_prices=stock_prices,
            ticker_amounts={"AAPL": 500.0},  # Not enough for even 1 share
            strategy="single_shot",
            available_cash=500.0,
        )
        
        result = simulate_portfolio(request)
        
        # Should have 0 shares since price > available amount
        assert result.holdings["AAPL"] == 0.0
        assert "AAPL" in result.insufficient_funds

    def test_whole_shares_only(self):
        """Test that only whole shares are purchased (integer division)."""
        stock_prices = {
            "AAPL": {"2024-01-01": 33.0},
        }
        
        request = SimulatePortfolioInput(
            stock_prices=stock_prices,
            ticker_amounts={"AAPL": 100.0},
            strategy="single_shot",
            available_cash=100.0,
        )
        
        result = simulate_portfolio(request)
        
        # 100 / 33 = 3.03, but should be 3 whole shares
        assert result.holdings["AAPL"] == 3.0
        assert result.transaction_log[0].shares == 3.0
        
        # Remaining cash should be 100 - (3 * 33) = 1
        assert result.remaining_cash == 1.0


class TestCalculateMetrics:
    """Tests for the calculate_metrics tool."""

    def test_portfolio_value_calculation(self):
        """Test that portfolio value is calculated correctly.
        
        Requirement 6.1: total_value = sum(holdings × current_price) + remaining_cash
        """
        request = CalculateMetricsInput(
            holdings={"AAPL": 10.0, "MSFT": 5.0},
            current_prices={"AAPL": 150.0, "MSFT": 300.0},
            invested_amounts={"AAPL": 1000.0, "MSFT": 1000.0},
            historical_prices={
                "AAPL": {"2024-01-01": 100.0, "2024-02-01": 150.0},
                "MSFT": {"2024-01-01": 200.0, "2024-02-01": 300.0},
            },
            spy_prices={"2024-01-01": 400.0, "2024-02-01": 450.0},
            remaining_cash=500.0,
        )
        
        result = calculate_metrics(request)
        
        # Expected: (10 * 150) + (5 * 300) + 500 = 1500 + 1500 + 500 = 3500
        assert result.total_value == 3500.0

    def test_absolute_returns_calculation(self):
        """Test that absolute returns are calculated correctly.
        
        Requirement 6.2: absolute_return = (holdings × current_price) - invested_amount
        """
        request = CalculateMetricsInput(
            holdings={"AAPL": 10.0},
            current_prices={"AAPL": 150.0},
            invested_amounts={"AAPL": 1000.0},
            historical_prices={"AAPL": {"2024-01-01": 100.0}},
            spy_prices={"2024-01-01": 400.0},
            remaining_cash=0.0,
        )
        
        result = calculate_metrics(request)
        
        # Expected: (10 * 150) - 1000 = 1500 - 1000 = 500
        assert result.returns["AAPL"] == 500.0

    def test_percent_returns_calculation(self):
        """Test that percentage returns are calculated correctly.
        
        Requirement 6.3: percent_return = ((holdings × current_price - invested) / invested) × 100
        """
        request = CalculateMetricsInput(
            holdings={"AAPL": 10.0},
            current_prices={"AAPL": 150.0},
            invested_amounts={"AAPL": 1000.0},
            historical_prices={"AAPL": {"2024-01-01": 100.0}},
            spy_prices={"2024-01-01": 400.0},
            remaining_cash=0.0,
        )
        
        result = calculate_metrics(request)
        
        # Expected: ((1500 - 1000) / 1000) * 100 = 50%
        assert result.percent_returns["AAPL"] == 50.0

    def test_allocation_calculation(self):
        """Test that allocation percentages are calculated correctly.
        
        Requirement 6.4: allocation = (invested_per_ticker / total_invested) × 100
        """
        request = CalculateMetricsInput(
            holdings={"AAPL": 10.0, "MSFT": 5.0},
            current_prices={"AAPL": 150.0, "MSFT": 300.0},
            invested_amounts={"AAPL": 2000.0, "MSFT": 3000.0},
            historical_prices={
                "AAPL": {"2024-01-01": 100.0},
                "MSFT": {"2024-01-01": 200.0},
            },
            spy_prices={"2024-01-01": 400.0},
            remaining_cash=0.0,
        )
        
        result = calculate_metrics(request)
        
        # Expected: AAPL = 2000/5000 * 100 = 40%, MSFT = 3000/5000 * 100 = 60%
        assert result.allocations["AAPL"] == 40.0
        assert result.allocations["MSFT"] == 60.0
        
        # Verify allocations sum to 100%
        total_allocation = sum(result.allocations.values())
        assert abs(total_allocation - 100.0) < 0.01

    def test_performance_data_generation(self):
        """Test that performance data is generated correctly.
        
        Requirements 6.5, 7.3: Generate time-series data with portfolio and SPY values
        """
        request = CalculateMetricsInput(
            holdings={"AAPL": 10.0},
            current_prices={"AAPL": 150.0},
            invested_amounts={"AAPL": 1000.0},
            historical_prices={
                "AAPL": {
                    "2024-01-01": 100.0,
                    "2024-02-01": 120.0,
                    "2024-03-01": 150.0,
                },
            },
            spy_prices={
                "2024-01-01": 400.0,
                "2024-02-01": 420.0,
                "2024-03-01": 450.0,
            },
            remaining_cash=0.0,
        )
        
        result = calculate_metrics(request)
        
        # Verify performance data has entries for all dates
        assert len(result.performance_data) == 3
        
        # Verify each point has date, portfolio, and spy values
        for point in result.performance_data:
            assert point.date is not None
            assert point.portfolio >= 0
            assert point.spy >= 0
        
        # Verify dates are in order
        dates = [p.date for p in result.performance_data]
        assert dates == sorted(dates)

    def test_zero_investment_handling(self):
        """Test handling of zero investment amounts."""
        request = CalculateMetricsInput(
            holdings={"AAPL": 0.0},
            current_prices={"AAPL": 150.0},
            invested_amounts={"AAPL": 0.0},
            historical_prices={"AAPL": {"2024-01-01": 100.0}},
            spy_prices={"2024-01-01": 400.0},
            remaining_cash=1000.0,
        )
        
        result = calculate_metrics(request)
        
        # Total value should just be remaining cash
        assert result.total_value == 1000.0
        
        # Returns should be 0
        assert result.returns["AAPL"] == 0.0
        
        # Percent return should be 0 (avoid division by zero)
        assert result.percent_returns["AAPL"] == 0.0
        
        # Allocation should be 0
        assert result.allocations["AAPL"] == 0.0

    def test_multiple_tickers_metrics(self):
        """Test metrics calculation with multiple tickers."""
        request = CalculateMetricsInput(
            holdings={"AAPL": 10.0, "GOOGL": 2.0, "MSFT": 5.0},
            current_prices={"AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 300.0},
            invested_amounts={"AAPL": 1000.0, "GOOGL": 5000.0, "MSFT": 1500.0},
            historical_prices={
                "AAPL": {"2024-01-01": 100.0},
                "GOOGL": {"2024-01-01": 2500.0},
                "MSFT": {"2024-01-01": 300.0},
            },
            spy_prices={"2024-01-01": 400.0},
            remaining_cash=0.0,
        )
        
        result = calculate_metrics(request)
        
        # Verify all tickers have metrics
        assert len(result.returns) == 3
        assert len(result.percent_returns) == 3
        assert len(result.allocations) == 3
        
        # Verify allocations sum to 100%
        total_allocation = sum(result.allocations.values())
        assert abs(total_allocation - 100.0) < 0.01


class TestSimulateSPYInvestment:
    """Tests for the simulate_spy_investment tool.
    
    Requirements: 7.1, 7.2
    """

    def test_single_shot_spy_investment(self):
        """Test single-shot SPY investment strategy.
        
        Requirement 7.1: Simulate investing the same total amount in SPY
        Requirement 7.2: Use the same investment strategy as the portfolio
        """
        spy_prices = {
            "2024-01-01": 400.0,
            "2024-02-01": 420.0,
            "2024-03-01": 450.0,
        }
        
        request = SimulateSPYInput(
            total_amount=10000.0,
            spy_prices=spy_prices,
            strategy="single_shot",
        )
        
        result = simulate_spy_investment(request)
        
        # Verify shares purchased at first price: 10000 / 400 = 25 shares
        assert result.spy_shares == 25.0
        
        # Verify total invested: 25 * 400 = 10000
        assert result.total_invested == 10000.0
        
        # Verify remaining cash: 10000 - 10000 = 0
        assert result.remaining_cash == 0.0
        
        # Verify transaction log
        assert len(result.transaction_log) == 1
        assert result.transaction_log[0].ticker == "SPY"
        assert result.transaction_log[0].shares == 25.0
        assert result.transaction_log[0].price == 400.0
        
        # Verify value over time
        assert len(result.value_over_time) == 3
        assert result.value_over_time["2024-01-01"] == 25 * 400.0  # 10000
        assert result.value_over_time["2024-02-01"] == 25 * 420.0  # 10500
        assert result.value_over_time["2024-03-01"] == 25 * 450.0  # 11250

    def test_dca_spy_investment(self):
        """Test DCA SPY investment strategy.
        
        Requirement 7.1: Simulate investing the same total amount in SPY
        Requirement 7.2: Use the same investment strategy as the portfolio
        """
        spy_prices = {
            "2024-01-01": 400.0,
            "2024-02-01": 420.0,
            "2024-03-01": 380.0,
            "2024-04-01": 450.0,
        }
        
        request = SimulateSPYInput(
            total_amount=4000.0,
            spy_prices=spy_prices,
            strategy="dca",
            dca_interval="monthly",
        )
        
        result = simulate_spy_investment(request)
        
        # Verify multiple transactions (DCA spreads purchases)
        assert len(result.transaction_log) > 1
        
        # Verify all transactions are for SPY
        for txn in result.transaction_log:
            assert txn.ticker == "SPY"
        
        # Verify cash conservation: total_amount = total_invested + remaining_cash
        assert abs(result.total_invested + result.remaining_cash - 4000.0) < 0.01
        
        # Verify value over time has entries for all dates
        assert len(result.value_over_time) == 4

    def test_spy_investment_matches_portfolio_amount(self):
        """Test that SPY investment uses the same total amount as portfolio.
        
        Requirement 7.1: THE Benchmark_Comparator SHALL simulate investing 
        the same total amount in SPY
        """
        # Simulate a portfolio investment
        portfolio_total = 5000.0
        
        spy_prices = {
            "2024-01-01": 450.0,
            "2024-02-01": 460.0,
        }
        
        request = SimulateSPYInput(
            total_amount=portfolio_total,
            spy_prices=spy_prices,
            strategy="single_shot",
        )
        
        result = simulate_spy_investment(request)
        
        # Verify total invested + remaining = original amount
        assert abs(result.total_invested + result.remaining_cash - portfolio_total) < 0.01

    def test_spy_investment_whole_shares_only(self):
        """Test that only whole shares are purchased for SPY.
        
        Same as portfolio simulation - integer division for realistic simulation.
        """
        spy_prices = {
            "2024-01-01": 333.0,
        }
        
        request = SimulateSPYInput(
            total_amount=1000.0,
            spy_prices=spy_prices,
            strategy="single_shot",
        )
        
        result = simulate_spy_investment(request)
        
        # 1000 / 333 = 3.003, but should be 3 whole shares
        assert result.spy_shares == 3.0
        
        # Remaining cash: 1000 - (3 * 333) = 1
        assert result.remaining_cash == 1.0

    def test_spy_dca_value_over_time_accumulates(self):
        """Test that DCA value over time correctly accumulates shares."""
        spy_prices = {
            "2024-01-01": 100.0,
            "2024-02-01": 100.0,
        }
        
        request = SimulateSPYInput(
            total_amount=200.0,
            spy_prices=spy_prices,
            strategy="dca",
            dca_interval="monthly",
        )
        
        result = simulate_spy_investment(request)
        
        # With DCA, we should buy 1 share at each date (100/100 = 1)
        # After first purchase: 1 share, 100 cash remaining
        # After second purchase: 2 shares, 0 cash remaining
        
        # Value at first date: 1 share * 100 + 100 cash = 200
        assert result.value_over_time["2024-01-01"] == 200.0
        
        # Value at second date: 2 shares * 100 + 0 cash = 200
        assert result.value_over_time["2024-02-01"] == 200.0
