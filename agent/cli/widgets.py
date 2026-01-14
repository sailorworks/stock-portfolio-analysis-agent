"""Custom Textual widgets for the CLI frontend.

This module contains:
- StatusPanel: Real-time analysis progress display
- HoldingsTable: Portfolio holdings table (Task 4)
- PerformanceDisplay: Portfolio vs SPY comparison (Task 5)
- InsightsPanel: Bull and bear insights display (Task 6)
- TransactionLog: Scrollable transaction history (Task 7)
"""

from typing import Optional

from textual.widgets import Static
from rich.table import Table
from rich.text import Text

from agent.models import InvestmentSummary, Insights, Insight


class StatusPanel(Static):
    """Displays analysis progress with status indicators.
    
    Shows real-time progress of the analysis with:
    - Spinners (⏳) for running operations
    - Checkmarks (✓) for completed operations
    - Error indicators (✗) for failed operations
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
    """
    
    DEFAULT_CSS = """
    StatusPanel {
        height: auto;
        padding: 1;
        border: solid #58A6FF;
        margin-bottom: 1;
    }
    """
    
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the StatusPanel."""
        super().__init__(*args, **kwargs)
        self._status_lines: list[tuple[str, str, str]] = []  # (indicator, message, style)
    
    def _render_status(self) -> Text:
        """Render the current status lines as Rich Text."""
        text = Text()
        
        if not self._status_lines:
            text.append("Ready for analysis", style="dim")
            return text
        
        for i, (indicator, message, style) in enumerate(self._status_lines):
            if i > 0:
                text.append("\n")
            text.append(f"{indicator} ", style=style)
            text.append(message, style=style)
        
        return text
    
    def set_running(self, tool_name: str) -> None:
        """Show tool is running with spinner.
        
        Args:
            tool_name: Name of the tool currently running
            
        Requirements: 3.2
        """
        self._status_lines.append(("⏳", tool_name, "yellow"))
        self.update(self._render_status())
    
    def set_complete(self, tool_name: str) -> None:
        """Show tool completed with checkmark.
        
        Replaces the running status for this tool with a completed status.
        
        Args:
            tool_name: Name of the tool that completed
            
        Requirements: 3.3
        """
        # Find and replace the running status for this tool
        for i, (indicator, message, style) in enumerate(self._status_lines):
            if indicator == "⏳" and message == tool_name:
                self._status_lines[i] = ("✓", tool_name, "green")
                break
        else:
            # If not found, just add as complete
            self._status_lines.append(("✓", tool_name, "green"))
        
        self.update(self._render_status())
    
    def set_error(self, message: str) -> None:
        """Show error with indicator.
        
        Args:
            message: Error message to display
            
        Requirements: 3.4
        """
        self._status_lines.append(("✗", message, "red"))
        self.update(self._render_status())
    
    def set_starting(self) -> None:
        """Show analysis is starting.
        
        Requirements: 3.1
        """
        self._status_lines = [("⏳", "Starting analysis...", "yellow")]
        self.update(self._render_status())
    
    def set_analysis_complete(self) -> None:
        """Show analysis is complete.
        
        Requirements: 3.5
        """
        self._status_lines.append(("✓", "Analysis complete", "green bold"))
        self.update(self._render_status())
    
    def reset(self) -> None:
        """Reset status for new analysis."""
        self._status_lines = []
        self.update(self._render_status())


class HoldingsTable(Static):
    """Displays portfolio holdings as a formatted table.
    
    Shows a Rich table with columns:
    - Ticker, Shares, Price, Value, Return, Return %, Allocation %
    
    Color-codes returns: green for positive, red for negative.
    Displays total value and remaining cash below the table.
    
    Requirements: 4.1, 4.2, 4.3, 4.4
    """
    
    DEFAULT_CSS = """
    HoldingsTable {
        height: auto;
        padding: 1;
        border: solid #58A6FF;
        margin-bottom: 1;
    }
    """
    
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the HoldingsTable."""
        super().__init__(*args, **kwargs)
        self._summary: Optional[InvestmentSummary] = None
    
    def _get_return_style(self, value: float) -> str:
        """Get the style for a return value based on sign.
        
        Args:
            value: The return value (positive, negative, or zero)
            
        Returns:
            Style string: 'green' for positive, 'red' for negative, '' for zero
            
        Requirements: 4.2
        """
        if value > 0:
            return "green"
        elif value < 0:
            return "red"
        return ""
    
    def _format_currency(self, value: float) -> str:
        """Format a value as currency."""
        return f"${value:,.2f}"
    
    def _format_percent(self, value: float) -> str:
        """Format a value as percentage."""
        return f"{value:,.2f}%"
    
    def _render_table(self) -> Table | Text:
        """Render the holdings table.
        
        Returns:
            Rich Table with holdings data, or Text message if no holdings
            
        Requirements: 4.1, 4.2, 4.3, 4.4
        """
        if self._summary is None or not self._summary.holdings:
            return Text("No holdings to display", style="dim italic")
        
        # Create the table with columns
        table = Table(
            title="Portfolio Holdings",
            show_header=True,
            header_style="bold #58A6FF",
            border_style="#58A6FF",
        )
        
        # Add columns - Requirements 4.1
        table.add_column("Ticker", style="bold")
        table.add_column("Shares", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("Value", justify="right")
        table.add_column("Return", justify="right")
        table.add_column("Return %", justify="right")
        table.add_column("Allocation %", justify="right")
        
        # Add rows for each holding
        for ticker, shares in self._summary.holdings.items():
            price = self._summary.final_prices.get(ticker, 0.0)
            value = shares * price
            return_val = self._summary.returns.get(ticker, 0.0)
            return_pct = self._summary.percent_return.get(ticker, 0.0)
            allocation = self._summary.percent_allocation.get(ticker, 0.0)
            
            # Get style for return values - Requirements 4.2
            return_style = self._get_return_style(return_val)
            
            # Format return with sign
            return_str = self._format_currency(return_val)
            if return_val > 0:
                return_str = f"+{return_str}"
            
            return_pct_str = self._format_percent(return_pct)
            if return_pct > 0:
                return_pct_str = f"+{return_pct_str}"
            
            table.add_row(
                ticker,
                f"{shares:,.4f}",
                self._format_currency(price),
                self._format_currency(value),
                Text(return_str, style=return_style),
                Text(return_pct_str, style=return_style),
                self._format_percent(allocation),
            )
        
        return table
    
    def _render_summary(self) -> Text:
        """Render the summary line with total value and cash.
        
        Requirements: 4.3
        """
        text = Text()
        
        if self._summary is None:
            return text
        
        text.append("\n")
        text.append("Total Portfolio Value: ", style="bold")
        text.append(self._format_currency(self._summary.total_value), style="bold green")
        text.append("  |  ")
        text.append("Remaining Cash: ", style="bold")
        text.append(self._format_currency(self._summary.cash), style="bold")
        
        return text
    
    def update_holdings(self, summary: InvestmentSummary) -> None:
        """Update the table with new holdings data.
        
        Args:
            summary: The InvestmentSummary containing holdings data
        """
        self._summary = summary
        self.refresh_display()
    
    def refresh_display(self) -> None:
        """Refresh the display with current data."""
        from rich.console import Group
        
        table = self._render_table()
        summary = self._render_summary()
        
        if isinstance(table, Text):
            # No holdings case
            self.update(table)
        else:
            # Combine table and summary
            self.update(Group(table, summary))
    
    def clear(self) -> None:
        """Clear the holdings display."""
        self._summary = None
        self.update(Text("No holdings to display", style="dim italic"))


class InsightsPanel(Static):
    """Displays bull and bear investment insights.
    
    Shows:
    - Bull insights with 📈 emoji in green
    - Bear insights with ⚠️ emoji in yellow/orange
    - Title and description for each insight
    - Handles missing insights case
    
    Requirements: 6.1, 6.2, 6.3, 6.4
    """
    
    DEFAULT_CSS = """
    InsightsPanel {
        height: auto;
        padding: 1;
        border: solid #58A6FF;
        margin-bottom: 1;
    }
    """
    
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the InsightsPanel."""
        super().__init__(*args, **kwargs)
        self._insights: Optional["Insights"] = None
        self._loading: bool = False
    
    def _render_insight(self, insight: "Insight", style: str) -> Text:
        """Render a single insight with emoji, title, and description.
        
        Args:
            insight: The Insight object to render
            style: The style to apply (e.g., 'green' for bull, 'yellow' for bear)
            
        Returns:
            Rich Text with formatted insight
            
        Requirements: 6.3
        """
        text = Text()
        text.append(f"{insight.emoji} ", style=style)
        text.append(insight.title, style=f"{style} bold")
        text.append("\n   ")
        text.append(insight.description, style="dim")
        return text
    
    def _render_insights(self) -> Text:
        """Render all insights.
        
        Returns:
            Rich Text with all bull and bear insights
            
        Requirements: 6.1, 6.2, 6.3, 6.4
        """
        text = Text()
        
        # Handle loading state - Requirements 6.4
        if self._loading:
            text.append("⏳ Generating insights...", style="yellow italic")
            return text
        
        # Handle missing insights case - Requirements 6.4
        if self._insights is None:
            text.append("No insights available", style="dim italic")
            return text
        
        has_bull = bool(self._insights.bull_insights)
        has_bear = bool(self._insights.bear_insights)
        
        if not has_bull and not has_bear:
            text.append("No insights available", style="dim italic")
            return text
        
        # Render bull insights - Requirements 6.1
        if has_bull:
            text.append("📈 BULLISH INSIGHTS\n", style="green bold underline")
            for i, insight in enumerate(self._insights.bull_insights):
                if i > 0:
                    text.append("\n")
                insight_text = self._render_insight(insight, "green")
                text.append_text(insight_text)
        
        # Add separator between bull and bear
        if has_bull and has_bear:
            text.append("\n\n")
        
        # Render bear insights - Requirements 6.2
        if has_bear:
            text.append("⚠️  BEARISH INSIGHTS\n", style="yellow bold underline")
            for i, insight in enumerate(self._insights.bear_insights):
                if i > 0:
                    text.append("\n")
                insight_text = self._render_insight(insight, "yellow")
                text.append_text(insight_text)
        
        return text
    
    def update_insights(self, insights: "Insights") -> None:
        """Update insights display.
        
        Args:
            insights: The Insights object containing bull and bear insights
        """
        self._insights = insights
        self._loading = False
        self.update(self._render_insights())
    
    def set_loading(self) -> None:
        """Set the panel to loading state.
        
        Requirements: 6.4
        """
        self._loading = True
        self._insights = None
        self.update(self._render_insights())
    
    def clear(self) -> None:
        """Clear the insights display."""
        self._insights = None
        self._loading = False
        self.update(Text("No insights available", style="dim italic"))


class TransactionLog(Static):
    """Scrollable transaction history log.
    
    Displays all buy transactions from investment_log with:
    - Date, action (Bought), shares, ticker, price, and total cost
    - Scrollable for long lists
    - Handles empty transactions case
    
    Requirements: 7.1, 7.2, 7.3, 7.4
    """
    
    DEFAULT_CSS = """
    TransactionLog {
        height: auto;
        max-height: 15;
        padding: 1;
        border: solid #58A6FF;
        margin-bottom: 1;
        overflow-y: auto;
    }
    """
    
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the TransactionLog."""
        super().__init__(*args, **kwargs)
        self._transactions: list[str] = []
    
    def _render_transactions(self) -> Text:
        """Render the transaction log.
        
        Returns:
            Rich Text with all transactions formatted
            
        Requirements: 7.1, 7.2, 7.3, 7.4
        """
        text = Text()
        
        # Handle empty transactions case - Requirements 7.4
        if not self._transactions:
            text.append("No transactions", style="dim italic")
            return text
        
        # Add header
        text.append("📋 Transaction History\n", style="bold #58A6FF underline")
        text.append("\n")
        
        # Display all transactions - Requirements 7.1, 7.2
        for i, transaction in enumerate(self._transactions):
            if i > 0:
                text.append("\n")
            
            # Parse and style the transaction entry
            # Format expected: "[date] Bought X shares of TICKER @ $price ($total)"
            # or similar human-readable format
            text.append("• ", style="dim")
            text.append(transaction, style="")
        
        return text
    
    def update_transactions(self, investment_log: list[str]) -> None:
        """Update transaction log display.
        
        Args:
            investment_log: List of human-readable transaction strings
            
        Requirements: 7.1
        """
        self._transactions = investment_log or []
        self.update(self._render_transactions())
    
    def clear(self) -> None:
        """Clear the transaction log display."""
        self._transactions = []
        self.update(Text("No transactions", style="dim italic"))


class PerformanceDisplay(Static):
    """Displays portfolio performance vs SPY benchmark.
    
    Shows:
    - Portfolio total return percentage
    - SPY benchmark return percentage
    - Outperformance/underperformance indicator
    - Total portfolio value prominently
    
    Requirements: 5.1, 5.2, 5.3, 5.4
    """
    
    DEFAULT_CSS = """
    PerformanceDisplay {
        height: auto;
        padding: 1;
        border: solid #58A6FF;
        margin-bottom: 1;
    }
    """
    
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the PerformanceDisplay."""
        super().__init__(*args, **kwargs)
        self._summary: Optional[InvestmentSummary] = None
    
    def _calculate_portfolio_return(self) -> Optional[float]:
        """Calculate the total portfolio return percentage.
        
        Uses performance_data to calculate return from first to last point.
        
        Returns:
            Portfolio return percentage, or None if not calculable
        """
        if self._summary is None or not self._summary.performance_data:
            return None
        
        first_point = self._summary.performance_data[0]
        last_point = self._summary.performance_data[-1]
        
        if first_point.portfolio > 0:
            return ((last_point.portfolio - first_point.portfolio) / first_point.portfolio) * 100
        return None
    
    def _calculate_spy_return(self) -> Optional[float]:
        """Calculate the SPY benchmark return percentage.
        
        Uses performance_data to calculate return from first to last point.
        
        Returns:
            SPY return percentage, or None if not calculable
            
        Requirements: 5.2
        """
        if self._summary is None or not self._summary.performance_data:
            return None
        
        first_point = self._summary.performance_data[0]
        last_point = self._summary.performance_data[-1]
        
        if first_point.spy > 0:
            return ((last_point.spy - first_point.spy) / first_point.spy) * 100
        return None
    
    def _get_comparison_indicator(self, portfolio_return: float, spy_return: float) -> tuple[str, str, str]:
        """Get the comparison indicator text, emoji, and style.
        
        Args:
            portfolio_return: Portfolio return percentage
            spy_return: SPY return percentage
            
        Returns:
            Tuple of (indicator_text, emoji, style)
            
        Requirements: 5.3
        """
        diff = portfolio_return - spy_return
        
        if diff > 0:
            return (f"Outperforming SPY by {diff:.2f}%", "🚀", "green bold")
        elif diff < 0:
            return (f"Underperforming SPY by {abs(diff):.2f}%", "📉", "red")
        else:
            return ("Matching SPY performance", "➡️", "yellow")
    
    def _format_return(self, value: float) -> tuple[str, str]:
        """Format a return value with sign and style.
        
        Args:
            value: Return percentage
            
        Returns:
            Tuple of (formatted_string, style)
        """
        if value > 0:
            return (f"+{value:.2f}%", "green")
        elif value < 0:
            return (f"{value:.2f}%", "red")
        else:
            return ("0.00%", "")
    
    def _render_performance(self) -> Text:
        """Render the performance display.
        
        Returns:
            Rich Text with performance metrics
            
        Requirements: 5.1, 5.2, 5.3, 5.4
        """
        text = Text()
        
        if self._summary is None:
            text.append("No performance data available", style="dim italic")
            return text
        
        # Display total portfolio value prominently - Requirements 5.4
        text.append("💰 Total Portfolio Value: ", style="bold")
        text.append(f"${self._summary.total_value:,.2f}", style="bold green")
        text.append("\n\n")
        
        # Calculate returns
        portfolio_return = self._calculate_portfolio_return()
        spy_return = self._calculate_spy_return()
        
        # Display portfolio return - Requirements 5.1
        text.append("📊 Portfolio Return: ", style="bold")
        if portfolio_return is not None:
            return_str, return_style = self._format_return(portfolio_return)
            text.append(return_str, style=return_style)
        else:
            text.append("N/A", style="dim")
        text.append("\n")
        
        # Display SPY benchmark return - Requirements 5.2
        text.append("📈 SPY Benchmark Return: ", style="bold")
        if spy_return is not None:
            spy_str, spy_style = self._format_return(spy_return)
            text.append(spy_str, style=spy_style)
        else:
            text.append("N/A", style="dim")
        text.append("\n\n")
        
        # Display comparison indicator - Requirements 5.3
        if portfolio_return is not None and spy_return is not None:
            indicator_text, emoji, indicator_style = self._get_comparison_indicator(
                portfolio_return, spy_return
            )
            text.append(f"{emoji} ", style=indicator_style)
            text.append(indicator_text, style=indicator_style)
        elif portfolio_return is None and spy_return is None:
            text.append("⏳ ", style="dim")
            text.append("Performance comparison unavailable", style="dim italic")
        
        return text
    
    def update_performance(self, summary: InvestmentSummary) -> None:
        """Update performance metrics display.
        
        Args:
            summary: The InvestmentSummary containing performance data
        """
        self._summary = summary
        self.update(self._render_performance())
    
    def clear(self) -> None:
        """Clear the performance display."""
        self._summary = None
        self.update(Text("No performance data available", style="dim italic"))
