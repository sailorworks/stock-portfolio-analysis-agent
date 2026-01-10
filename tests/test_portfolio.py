"""Tests for portfolio state management.

This module contains tests to verify portfolio management functionality:
- Adding investments (additive approach)
- Removing investments
- Portfolio data completeness

Requirements: 3.1, 3.2, 3.3
"""

import pytest

from agent.portfolio import (
    PortfolioHolding,
    Portfolio,
    PortfolioManager,
    get_portfolio_manager,
)


class TestPortfolioHolding:
    """Tests for the PortfolioHolding model."""

    def test_valid_holding_creation(self):
        """Test creating a valid portfolio holding."""
        holding = PortfolioHolding(
            ticker_symbol="AAPL",
            investment_amount=1000.0,
            purchase_date="2024-01-15",
        )
        
        assert holding.ticker_symbol == "AAPL"
        assert holding.investment_amount == 1000.0
        assert holding.purchase_date == "2024-01-15"

    def test_holding_requires_positive_amount(self):
        """Test that investment_amount must be positive."""
        with pytest.raises(ValueError):
            PortfolioHolding(
                ticker_symbol="AAPL",
                investment_amount=0.0,
                purchase_date="2024-01-15",
            )
        
        with pytest.raises(ValueError):
            PortfolioHolding(
                ticker_symbol="AAPL",
                investment_amount=-100.0,
                purchase_date="2024-01-15",
            )

    def test_holding_validates_date_format(self):
        """Test that purchase_date must be in YYYY-MM-DD format."""
        with pytest.raises(ValueError) as exc_info:
            PortfolioHolding(
                ticker_symbol="AAPL",
                investment_amount=1000.0,
                purchase_date="01-15-2024",  # Wrong format
            )
        
        assert "Invalid purchase_date format" in str(exc_info.value)

    def test_holding_data_completeness(self):
        """Test that holding contains all required fields.
        
        Property 4: Portfolio Data Completeness
        Validates: Requirements 3.2
        """
        holding = PortfolioHolding(
            ticker_symbol="MSFT",
            investment_amount=5000.0,
            purchase_date="2023-06-01",
        )
        
        # Verify all required fields are present and valid
        assert isinstance(holding.ticker_symbol, str)
        assert len(holding.ticker_symbol) > 0
        assert isinstance(holding.investment_amount, float)
        assert holding.investment_amount > 0
        assert isinstance(holding.purchase_date, str)
        # Date format is validated in model_post_init


class TestPortfolio:
    """Tests for the Portfolio model."""

    def test_empty_portfolio_creation(self):
        """Test creating an empty portfolio."""
        portfolio = Portfolio()
        
        assert portfolio.holdings == []
        assert portfolio.is_empty()

    def test_add_investment_additive(self):
        """Test that adding investments is additive.
        
        Property 3: Portfolio Additive Invariant
        Validates: Requirements 3.1
        """
        # Start with a portfolio containing one holding
        initial_holding = PortfolioHolding(
            ticker_symbol="AAPL",
            investment_amount=1000.0,
            purchase_date="2024-01-01",
        )
        portfolio = Portfolio(holdings=[initial_holding])
        
        # Add a new investment
        new_holding = PortfolioHolding(
            ticker_symbol="MSFT",
            investment_amount=2000.0,
            purchase_date="2024-02-01",
        )
        updated_portfolio = portfolio.add_investment(new_holding)
        
        # Verify additive behavior:
        # 1. Original holding is still present
        assert len(updated_portfolio.holdings) == 2
        
        # 2. Original holding is unchanged
        aapl_holdings = updated_portfolio.get_holdings_by_ticker("AAPL")
        assert len(aapl_holdings) == 1
        assert aapl_holdings[0].investment_amount == 1000.0
        
        # 3. New holding is added
        msft_holdings = updated_portfolio.get_holdings_by_ticker("MSFT")
        assert len(msft_holdings) == 1
        assert msft_holdings[0].investment_amount == 2000.0

    def test_add_same_ticker_multiple_times(self):
        """Test adding multiple investments in the same ticker.
        
        Property 3: Portfolio Additive Invariant
        Validates: Requirements 3.1
        """
        portfolio = Portfolio()
        
        # Add first AAPL investment
        holding1 = PortfolioHolding(
            ticker_symbol="AAPL",
            investment_amount=1000.0,
            purchase_date="2024-01-01",
        )
        portfolio = portfolio.add_investment(holding1)
        
        # Add second AAPL investment
        holding2 = PortfolioHolding(
            ticker_symbol="AAPL",
            investment_amount=500.0,
            purchase_date="2024-02-01",
        )
        portfolio = portfolio.add_investment(holding2)
        
        # Both holdings should be present (additive)
        aapl_holdings = portfolio.get_holdings_by_ticker("AAPL")
        assert len(aapl_holdings) == 2
        
        # Total invested in AAPL should be sum of both
        invested = portfolio.get_invested_by_ticker()
        assert invested["AAPL"] == 1500.0

    def test_remove_investment(self):
        """Test removing investments from portfolio.
        
        Property 5: Portfolio Removal Correctness
        Validates: Requirements 3.3
        """
        # Create portfolio with multiple tickers
        holdings = [
            PortfolioHolding(ticker_symbol="AAPL", investment_amount=1000.0, purchase_date="2024-01-01"),
            PortfolioHolding(ticker_symbol="MSFT", investment_amount=2000.0, purchase_date="2024-01-01"),
            PortfolioHolding(ticker_symbol="GOOGL", investment_amount=3000.0, purchase_date="2024-01-01"),
        ]
        portfolio = Portfolio(holdings=holdings)
        
        # Remove MSFT
        updated_portfolio = portfolio.remove_investment("MSFT")
        
        # Verify MSFT is removed
        assert len(updated_portfolio.get_holdings_by_ticker("MSFT")) == 0
        
        # Verify other holdings are unchanged
        assert len(updated_portfolio.get_holdings_by_ticker("AAPL")) == 1
        assert len(updated_portfolio.get_holdings_by_ticker("GOOGL")) == 1
        
        # Verify total holdings count
        assert len(updated_portfolio.holdings) == 2

    def test_remove_nonexistent_ticker(self):
        """Test removing a ticker that doesn't exist."""
        holding = PortfolioHolding(
            ticker_symbol="AAPL",
            investment_amount=1000.0,
            purchase_date="2024-01-01",
        )
        portfolio = Portfolio(holdings=[holding])
        
        # Remove a ticker that doesn't exist
        updated_portfolio = portfolio.remove_investment("MSFT")
        
        # Portfolio should be unchanged
        assert len(updated_portfolio.holdings) == 1
        assert updated_portfolio.holdings[0].ticker_symbol == "AAPL"

    def test_remove_all_holdings_of_ticker(self):
        """Test that remove_investment removes ALL holdings of a ticker.
        
        Property 5: Portfolio Removal Correctness
        Validates: Requirements 3.3
        """
        # Create portfolio with multiple holdings of same ticker
        holdings = [
            PortfolioHolding(ticker_symbol="AAPL", investment_amount=1000.0, purchase_date="2024-01-01"),
            PortfolioHolding(ticker_symbol="AAPL", investment_amount=500.0, purchase_date="2024-02-01"),
            PortfolioHolding(ticker_symbol="MSFT", investment_amount=2000.0, purchase_date="2024-01-01"),
        ]
        portfolio = Portfolio(holdings=holdings)
        
        # Remove AAPL
        updated_portfolio = portfolio.remove_investment("AAPL")
        
        # All AAPL holdings should be removed
        assert len(updated_portfolio.get_holdings_by_ticker("AAPL")) == 0
        
        # MSFT should remain
        assert len(updated_portfolio.get_holdings_by_ticker("MSFT")) == 1
        assert len(updated_portfolio.holdings) == 1

    def test_get_all_tickers(self):
        """Test getting all unique tickers in portfolio."""
        holdings = [
            PortfolioHolding(ticker_symbol="AAPL", investment_amount=1000.0, purchase_date="2024-01-01"),
            PortfolioHolding(ticker_symbol="AAPL", investment_amount=500.0, purchase_date="2024-02-01"),
            PortfolioHolding(ticker_symbol="MSFT", investment_amount=2000.0, purchase_date="2024-01-01"),
        ]
        portfolio = Portfolio(holdings=holdings)
        
        tickers = portfolio.get_all_tickers()
        
        assert len(tickers) == 2
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_get_total_invested(self):
        """Test calculating total invested amount."""
        holdings = [
            PortfolioHolding(ticker_symbol="AAPL", investment_amount=1000.0, purchase_date="2024-01-01"),
            PortfolioHolding(ticker_symbol="MSFT", investment_amount=2000.0, purchase_date="2024-01-01"),
            PortfolioHolding(ticker_symbol="GOOGL", investment_amount=3000.0, purchase_date="2024-01-01"),
        ]
        portfolio = Portfolio(holdings=holdings)
        
        total = portfolio.get_total_invested()
        
        assert total == 6000.0

    def test_immutability_of_add_investment(self):
        """Test that add_investment returns a new portfolio, not modifying original."""
        original_holding = PortfolioHolding(
            ticker_symbol="AAPL",
            investment_amount=1000.0,
            purchase_date="2024-01-01",
        )
        original_portfolio = Portfolio(holdings=[original_holding])
        original_count = len(original_portfolio.holdings)
        
        # Add new investment
        new_holding = PortfolioHolding(
            ticker_symbol="MSFT",
            investment_amount=2000.0,
            purchase_date="2024-02-01",
        )
        new_portfolio = original_portfolio.add_investment(new_holding)
        
        # Original should be unchanged
        assert len(original_portfolio.holdings) == original_count
        
        # New portfolio should have the addition
        assert len(new_portfolio.holdings) == original_count + 1


class TestPortfolioManager:
    """Tests for the PortfolioManager class."""

    def test_get_portfolio_creates_empty_if_not_exists(self):
        """Test that get_portfolio creates an empty portfolio for new users."""
        manager = PortfolioManager()
        
        portfolio = manager.get_portfolio("user123")
        
        assert portfolio is not None
        assert portfolio.is_empty()
        assert portfolio.user_id == "user123"

    def test_add_investment_via_manager(self):
        """Test adding investment through the manager."""
        manager = PortfolioManager()
        
        portfolio = manager.add_investment(
            user_id="user123",
            ticker_symbol="AAPL",
            investment_amount=1000.0,
            purchase_date="2024-01-15",
        )
        
        assert len(portfolio.holdings) == 1
        assert portfolio.holdings[0].ticker_symbol == "AAPL"
        assert portfolio.holdings[0].investment_amount == 1000.0

    def test_remove_investment_via_manager(self):
        """Test removing investment through the manager."""
        manager = PortfolioManager()
        
        # Add some investments
        manager.add_investment("user123", "AAPL", 1000.0, "2024-01-01")
        manager.add_investment("user123", "MSFT", 2000.0, "2024-01-01")
        
        # Remove AAPL
        portfolio = manager.remove_investment("user123", "AAPL")
        
        assert len(portfolio.holdings) == 1
        assert portfolio.holdings[0].ticker_symbol == "MSFT"

    def test_portfolio_persistence_across_operations(self):
        """Test that portfolio state persists across multiple operations."""
        manager = PortfolioManager()
        user_id = "user456"
        
        # Add first investment
        manager.add_investment(user_id, "AAPL", 1000.0, "2024-01-01")
        
        # Add second investment
        manager.add_investment(user_id, "MSFT", 2000.0, "2024-02-01")
        
        # Get portfolio and verify both are present
        portfolio = manager.get_portfolio(user_id)
        
        assert len(portfolio.holdings) == 2
        assert portfolio.get_total_invested() == 3000.0

    def test_separate_portfolios_per_user(self):
        """Test that different users have separate portfolios."""
        manager = PortfolioManager()
        
        # Add investment for user1
        manager.add_investment("user1", "AAPL", 1000.0, "2024-01-01")
        
        # Add investment for user2
        manager.add_investment("user2", "MSFT", 2000.0, "2024-01-01")
        
        # Verify portfolios are separate
        portfolio1 = manager.get_portfolio("user1")
        portfolio2 = manager.get_portfolio("user2")
        
        assert len(portfolio1.holdings) == 1
        assert portfolio1.holdings[0].ticker_symbol == "AAPL"
        
        assert len(portfolio2.holdings) == 1
        assert portfolio2.holdings[0].ticker_symbol == "MSFT"

    def test_clear_portfolio(self):
        """Test clearing all holdings from a portfolio."""
        manager = PortfolioManager()
        
        # Add some investments
        manager.add_investment("user123", "AAPL", 1000.0, "2024-01-01")
        manager.add_investment("user123", "MSFT", 2000.0, "2024-01-01")
        
        # Clear portfolio
        portfolio = manager.clear_portfolio("user123")
        
        assert portfolio.is_empty()
        assert len(portfolio.holdings) == 0

    def test_has_portfolio(self):
        """Test checking if a user has a portfolio."""
        manager = PortfolioManager()
        
        # Initially no portfolio
        assert not manager.has_portfolio("user123")
        
        # After getting portfolio, it exists
        manager.get_portfolio("user123")
        assert manager.has_portfolio("user123")


class TestGlobalPortfolioManager:
    """Tests for the global portfolio manager instance."""

    def test_get_portfolio_manager_returns_same_instance(self):
        """Test that get_portfolio_manager returns the same instance."""
        manager1 = get_portfolio_manager()
        manager2 = get_portfolio_manager()
        
        assert manager1 is manager2

    def test_global_manager_persists_state(self):
        """Test that global manager persists state across calls."""
        manager = get_portfolio_manager()
        
        # Add investment
        manager.add_investment("global_test_user", "AAPL", 1000.0, "2024-01-01")
        
        # Get manager again and verify state
        manager2 = get_portfolio_manager()
        portfolio = manager2.get_portfolio("global_test_user")
        
        assert len(portfolio.holdings) >= 1
