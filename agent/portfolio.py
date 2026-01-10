"""Portfolio state management for the Stock Portfolio Analysis Agent.

This module provides portfolio state management functionality including:
- Storing portfolio holdings in session state
- Adding new investments (additive approach)
- Removing investments

Requirements: 3.1, 3.2, 3.3
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class PortfolioHolding(BaseModel):
    """A single holding in the portfolio.
    
    Requirements: 3.2 - Track ticker symbols, investment amounts, and purchase dates
    """
    ticker_symbol: str = Field(
        ...,
        description="Stock ticker symbol (e.g., 'AAPL')"
    )
    investment_amount: float = Field(
        ...,
        gt=0,
        description="Amount invested in this holding (must be > 0)"
    )
    purchase_date: str = Field(
        ...,
        description="Date of purchase in YYYY-MM-DD format"
    )
    
    def model_post_init(self, __context) -> None:
        """Validate purchase_date format after initialization."""
        try:
            datetime.strptime(self.purchase_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid purchase_date format: {self.purchase_date}. Use YYYY-MM-DD.")


class Portfolio(BaseModel):
    """Portfolio containing multiple holdings.
    
    Requirements: 3.1, 3.2, 3.3
    """
    holdings: List[PortfolioHolding] = Field(
        default_factory=list,
        description="List of portfolio holdings"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User ID for session-scoped portfolio"
    )
    
    def add_investment(self, holding: PortfolioHolding) -> "Portfolio":
        """Add a new investment to the portfolio (additive approach).
        
        Requirements: 3.1 - WHEN a user adds a new investment, THE Portfolio_Manager 
        SHALL combine it with existing holdings (additive approach)
        
        Args:
            holding: The new holding to add
            
        Returns:
            Updated Portfolio with the new holding added
        """
        # Create a new list with existing holdings plus the new one
        new_holdings = list(self.holdings) + [holding]
        return Portfolio(holdings=new_holdings, user_id=self.user_id)
    
    def remove_investment(self, ticker_symbol: str) -> "Portfolio":
        """Remove all holdings for a specific ticker from the portfolio.
        
        Requirements: 3.3 - WHEN a user explicitly requests removal, 
        THE Portfolio_Manager SHALL remove specified holdings from the portfolio
        
        Args:
            ticker_symbol: The ticker symbol to remove
            
        Returns:
            Updated Portfolio with the ticker removed
        """
        # Filter out holdings with the specified ticker
        new_holdings = [h for h in self.holdings if h.ticker_symbol != ticker_symbol]
        return Portfolio(holdings=new_holdings, user_id=self.user_id)
    
    def get_holdings_by_ticker(self, ticker_symbol: str) -> List[PortfolioHolding]:
        """Get all holdings for a specific ticker.
        
        Args:
            ticker_symbol: The ticker symbol to look up
            
        Returns:
            List of holdings for the specified ticker
        """
        return [h for h in self.holdings if h.ticker_symbol == ticker_symbol]
    
    def get_all_tickers(self) -> List[str]:
        """Get a list of all unique ticker symbols in the portfolio.
        
        Returns:
            List of unique ticker symbols
        """
        return list(set(h.ticker_symbol for h in self.holdings))
    
    def get_total_invested(self) -> float:
        """Get the total amount invested across all holdings.
        
        Returns:
            Total investment amount
        """
        return sum(h.investment_amount for h in self.holdings)
    
    def get_invested_by_ticker(self) -> Dict[str, float]:
        """Get total invested amount per ticker.
        
        Returns:
            Dict mapping ticker symbols to total invested amounts
        """
        invested: Dict[str, float] = {}
        for holding in self.holdings:
            ticker = holding.ticker_symbol
            invested[ticker] = invested.get(ticker, 0.0) + holding.investment_amount
        return invested
    
    def is_empty(self) -> bool:
        """Check if the portfolio has no holdings.
        
        Returns:
            True if portfolio is empty, False otherwise
        """
        return len(self.holdings) == 0


class PortfolioManager:
    """Manages portfolio state for users.
    
    This class provides session-scoped portfolio management,
    storing portfolios in memory keyed by user_id.
    
    Requirements: 3.1, 3.2, 3.3, 3.4
    """
    
    def __init__(self):
        """Initialize the portfolio manager with empty state."""
        self._portfolios: Dict[str, Portfolio] = {}
    
    def get_portfolio(self, user_id: str) -> Portfolio:
        """Get the portfolio for a user, creating an empty one if it doesn't exist.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            The user's portfolio
        """
        if user_id not in self._portfolios:
            self._portfolios[user_id] = Portfolio(user_id=user_id)
        return self._portfolios[user_id]
    
    def set_portfolio(self, user_id: str, portfolio: Portfolio) -> None:
        """Set the portfolio for a user.
        
        Args:
            user_id: The user's unique identifier
            portfolio: The portfolio to store
        """
        self._portfolios[user_id] = portfolio
    
    def add_investment(
        self,
        user_id: str,
        ticker_symbol: str,
        investment_amount: float,
        purchase_date: str,
    ) -> Portfolio:
        """Add a new investment to a user's portfolio.
        
        Requirements: 3.1 - Additive approach for new investments
        Requirements: 3.2 - Track ticker, amount, and date
        
        Args:
            user_id: The user's unique identifier
            ticker_symbol: Stock ticker symbol
            investment_amount: Amount to invest
            purchase_date: Date of purchase (YYYY-MM-DD)
            
        Returns:
            Updated portfolio
        """
        holding = PortfolioHolding(
            ticker_symbol=ticker_symbol,
            investment_amount=investment_amount,
            purchase_date=purchase_date,
        )
        
        portfolio = self.get_portfolio(user_id)
        updated_portfolio = portfolio.add_investment(holding)
        self.set_portfolio(user_id, updated_portfolio)
        
        return updated_portfolio
    
    def remove_investment(self, user_id: str, ticker_symbol: str) -> Portfolio:
        """Remove all holdings for a ticker from a user's portfolio.
        
        Requirements: 3.3 - Remove specified holdings when explicitly requested
        
        Args:
            user_id: The user's unique identifier
            ticker_symbol: Stock ticker symbol to remove
            
        Returns:
            Updated portfolio
        """
        portfolio = self.get_portfolio(user_id)
        updated_portfolio = portfolio.remove_investment(ticker_symbol)
        self.set_portfolio(user_id, updated_portfolio)
        
        return updated_portfolio
    
    def clear_portfolio(self, user_id: str) -> Portfolio:
        """Clear all holdings from a user's portfolio.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            Empty portfolio
        """
        empty_portfolio = Portfolio(user_id=user_id)
        self.set_portfolio(user_id, empty_portfolio)
        return empty_portfolio
    
    def has_portfolio(self, user_id: str) -> bool:
        """Check if a user has a portfolio.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            True if user has a portfolio, False otherwise
        """
        return user_id in self._portfolios


# Global portfolio manager instance for session state
_portfolio_manager: Optional[PortfolioManager] = None


def get_portfolio_manager() -> PortfolioManager:
    """Get or create the global portfolio manager instance.
    
    Returns:
        The global PortfolioManager instance
    """
    global _portfolio_manager
    if _portfolio_manager is None:
        _portfolio_manager = PortfolioManager()
    return _portfolio_manager
