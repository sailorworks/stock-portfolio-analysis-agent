"""Main Textual application for the CLI frontend.

This module contains the PortfolioApp class that provides an interactive
terminal interface for the Stock Portfolio Analysis Agent.

Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5, 8.1, 8.2, 8.3, 9.1, 9.2, 9.3, 9.4, 10.1, 10.2, 10.3
"""

import asyncio
import os
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Input, Static, Label

from agent.cli.widgets import (
    StatusPanel,
    HoldingsTable,
    PerformanceDisplay,
    InsightsPanel,
    TransactionLog,
)
from agent.cli.client import AgentClient
from agent.errors import ErrorCode


class QueryDisplay(Static):
    """Displays the submitted query above results."""
    
    DEFAULT_CSS = """
    QueryDisplay {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        border: solid #58A6FF;
        display: none;
    }
    
    QueryDisplay.visible {
        display: block;
    }
    """
    
    def show_query(self, query: str) -> None:
        """Display the submitted query."""
        self.update(f"📝 Query: {query}")
        self.add_class("visible")
    
    def clear(self) -> None:
        """Clear the query display."""
        self.update("")
        self.remove_class("visible")


class WelcomeMessage(Static):
    """Displays the welcome message on startup."""
    
    DEFAULT_CSS = """
    WelcomeMessage {
        height: auto;
        padding: 2;
        margin: 1;
        text-align: center;
        border: double #58A6FF;
    }
    
    WelcomeMessage.hidden {
        display: none;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static(
            "🚀 [bold #58A6FF]Stock Portfolio Analysis Agent[/bold #58A6FF]\n\n"
            "Enter a natural language query to analyze hypothetical investments.\n\n"
            "[dim]Example: \"What if I invested $10k in AAPL since 2020?\"[/dim]",
            markup=True,
        )
    
    def hide(self) -> None:
        """Hide the welcome message."""
        self.add_class("hidden")
    
    def show(self) -> None:
        """Show the welcome message."""
        self.remove_class("hidden")


def check_api_keys() -> Optional[str]:
    """Check if required API keys are set in environment.
    
    Returns:
        Error message if keys are missing, None if all keys are present.
        
    Requirements: 9.3 - Handle missing API keys on startup
    """
    required_keys = ["COMPOSIO_API_KEY", "OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.environ.get(key)]
    
    if missing_keys:
        if len(missing_keys) == 1:
            return f"Missing API key. Please set {missing_keys[0]} in .env"
        else:
            keys_str = ", ".join(missing_keys[:-1]) + " and " + missing_keys[-1]
            return f"Missing API keys. Please set {keys_str} in .env"
    return None


def format_error_message(error: str) -> str:
    """Format an error message into a user-friendly display message.
    
    Args:
        error: The raw error message from the agent
        
    Returns:
        User-friendly error message
        
    Requirements: 9.1, 9.2 - Display user-friendly error messages
    """
    error_lower = error.lower()
    
    # Check for invalid ticker errors
    if "invalid ticker" in error_lower or ErrorCode.INVALID_TICKER.value.lower() in error_lower:
        # Try to extract ticker symbols from the error
        import re
        ticker_match = re.search(r'(?:invalid ticker[s]?:?\s*|tickers?:\s*)([A-Z,\s]+)', error, re.IGNORECASE)
        if ticker_match:
            tickers = ticker_match.group(1).strip()
            return f"Invalid ticker: {tickers}. Please check the symbol."
        return "Invalid ticker. Please check the symbol."
    
    # Check for no data available errors
    if "no data available" in error_lower or ErrorCode.NO_DATA_AVAILABLE.value.lower() in error_lower:
        return "No data available for the specified date range"
    
    # Check for API key errors
    if "api key" in error_lower or "api_key" in error_lower or "authentication" in error_lower:
        return "Missing API keys. Please set COMPOSIO_API_KEY and OPENAI_API_KEY in .env"
    
    # Check for network/connection errors
    if "network" in error_lower or "connection" in error_lower or "timeout" in error_lower:
        return "Network error. Please check your internet connection and try again."
    
    # Check for Yahoo Finance API errors
    if "yfinance" in error_lower or "yahoo" in error_lower:
        return "Failed to fetch stock data. Please try again later."
    
    # Check for validation errors
    if "validation" in error_lower or ErrorCode.VALIDATION_ERROR.value.lower() in error_lower:
        return f"Invalid input: {error}"
    
    # Default: return a cleaned up version of the error
    # Remove technical details but keep the core message
    if len(error) > 200:
        return f"Error: {error[:200]}..."
    return f"Error: {error}"


class PortfolioApp(App):
    """Main CLI application for portfolio analysis.
    
    This Textual application provides an interactive terminal interface
    for the Stock Portfolio Analysis Agent. It integrates with the existing
    Composio + OpenAI agent architecture by calling run_agent_analysis().
    
    Requirements:
        1.1 - Full-screen terminal interface with header, input, and result panels
        1.2 - Dark theme with #58A6FF highlights
        1.3 - Welcome message and focus on Query_Input on startup
        1.4 - Footer with keyboard shortcuts
        2.1 - Submit query to agent via run_agent_analysis()
        2.2 - Accept natural language queries
        2.3 - Display query text above results
        2.4 - Disable input during analysis
        2.5 - Reject empty queries
        8.1, 8.2, 8.3 - Integration with existing agent architecture
    """
    
    # Application title
    TITLE = "Stock Portfolio Analysis Agent"
    SUB_TITLE = "Powered by Composio + OpenAI"
    
    # Dark theme CSS with #58A6FF highlights - Requirements 1.2
    CSS = """
    Screen {
        background: #1a1a2e;
    }
    
    Header {
        background: #16213e;
        color: #58A6FF;
    }
    
    Footer {
        background: #16213e;
    }
    
    #main-container {
        width: 100%;
        height: 100%;
        padding: 1;
    }
    
    #input-container {
        height: auto;
        margin-bottom: 1;
    }
    
    #query-input {
        border: solid #58A6FF;
        background: #0f0f23;
        color: #e0e0e0;
        padding: 1;
    }
    
    #query-input:focus {
        border: solid #7ec8ff;
    }
    
    #query-input.-disabled {
        background: #2a2a3e;
        color: #666;
    }
    
    #results-container {
        height: 1fr;
        overflow-y: auto;
    }
    
    #left-panel {
        width: 60%;
        padding-right: 1;
    }
    
    #right-panel {
        width: 40%;
    }
    
    .panel-title {
        text-style: bold;
        color: #58A6FF;
        margin-bottom: 1;
    }
    """
    
    # Keyboard bindings - Requirements 1.4
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+c", "cancel", "Cancel/Clear", show=True),
    ]
    
    def __init__(self):
        """Initialize the PortfolioApp."""
        super().__init__()
        self._agent_client = AgentClient()
        self._analysis_running = False
        self._api_key_error: Optional[str] = check_api_keys()
        self._current_analysis_task: Optional[asyncio.Task] = None
    
    def compose(self) -> ComposeResult:
        """Create the application layout.
        
        Requirements: 1.1 - Full-screen terminal interface
        """
        yield Header()
        
        with Container(id="main-container"):
            # Input area
            with Container(id="input-container"):
                yield Input(
                    placeholder="Ask anything... (e.g., 'What if I invested $10k in AAPL since 2020?')",
                    id="query-input",
                )
            
            # Welcome message - Requirements 1.3
            yield WelcomeMessage(id="welcome-message")
            
            # Query display (hidden initially)
            yield QueryDisplay(id="query-display")
            
            # Status panel
            yield StatusPanel(id="status-panel")
            
            # Results area with two columns
            with ScrollableContainer(id="results-container"):
                with Horizontal():
                    # Left panel: Holdings and Performance
                    with Vertical(id="left-panel"):
                        yield HoldingsTable(id="holdings-table")
                        yield PerformanceDisplay(id="performance-display")
                    
                    # Right panel: Insights and Transactions
                    with Vertical(id="right-panel"):
                        yield InsightsPanel(id="insights-panel")
                        yield TransactionLog(id="transaction-log")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Handle application mount event.
        
        Requirements: 1.3 - Focus Query_Input on startup
        Requirements: 9.3 - Handle missing API keys on startup
        """
        # Focus the query input on startup
        query_input = self.query_one("#query-input", Input)
        query_input.focus()
        
        # Check for API key errors and display warning
        if self._api_key_error:
            status_panel = self.query_one("#status-panel", StatusPanel)
            status_panel.set_error(self._api_key_error)
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle query submission.
        
        Requirements:
            2.1 - Submit query to agent
            2.2 - Accept natural language queries
            2.3 - Display query text above results
            2.4 - Disable input during analysis
            2.5 - Reject empty queries
        """
        query = event.value.strip()
        
        # Requirement 2.5 - Reject empty/whitespace-only queries
        if not query:
            status_panel = self.query_one("#status-panel", StatusPanel)
            status_panel.set_error("Please enter a query")
            return
        
        # Prevent concurrent analysis
        if self._analysis_running:
            return
        
        # Run the analysis as a tracked task for cancellation support
        self._current_analysis_task = asyncio.create_task(self.run_analysis(query))
        try:
            await self._current_analysis_task
        except asyncio.CancelledError:
            # Task was cancelled, cleanup is handled in action_cancel
            pass
        finally:
            self._current_analysis_task = None
    
    async def run_analysis(self, query: str) -> None:
        """Execute portfolio analysis via existing agent and update UI.
        
        Requirements:
            8.1 - Call existing run_agent_analysis()
            8.2 - Pass user query and receive InvestmentSummary
            8.3 - Use existing agent orchestrator
            9.1, 9.2 - Display user-friendly error messages
            9.4 - Re-enable input after errors
        """
        # Get widget references
        query_input = self.query_one("#query-input", Input)
        welcome_message = self.query_one("#welcome-message", WelcomeMessage)
        query_display = self.query_one("#query-display", QueryDisplay)
        status_panel = self.query_one("#status-panel", StatusPanel)
        holdings_table = self.query_one("#holdings-table", HoldingsTable)
        performance_display = self.query_one("#performance-display", PerformanceDisplay)
        insights_panel = self.query_one("#insights-panel", InsightsPanel)
        transaction_log = self.query_one("#transaction-log", TransactionLog)
        
        # Requirement 2.4 - Disable input during analysis
        self._analysis_running = True
        query_input.disabled = True
        query_input.value = ""
        
        # Hide welcome message and show query
        welcome_message.hide()
        
        # Requirement 2.3 - Display query above results
        query_display.show_query(query)
        
        # Clear previous results
        holdings_table.clear()
        performance_display.clear()
        insights_panel.clear()
        transaction_log.clear()
        
        # Start analysis status
        status_panel.set_starting()
        status_panel.set_running("Analyzing query with AI agent...")
        
        try:
            # Call the agent - Requirements 8.1, 8.2, 8.3
            result = await self._agent_client.analyze(query)
            
            if result.success and result.summary:
                # Update status
                status_panel.set_complete("Analyzing query with AI agent...")
                status_panel.set_analysis_complete()
                
                # Update all widgets with the summary
                holdings_table.update_holdings(result.summary)
                performance_display.update_performance(result.summary)
                
                # Update insights if available
                if result.summary.insights:
                    insights_panel.update_insights(result.summary.insights)
                else:
                    insights_panel.clear()
                
                # Update transaction log
                transaction_log.update_transactions(result.summary.investment_log)
            else:
                # Handle error - Requirements 9.1, 9.2
                error_msg = result.error or "Unknown error occurred"
                user_friendly_error = format_error_message(error_msg)
                status_panel.set_error(user_friendly_error)
        
        except Exception as e:
            # Handle unexpected errors - Requirements 9.1, 9.2
            user_friendly_error = format_error_message(str(e))
            status_panel.set_error(user_friendly_error)
        
        finally:
            # Re-enable input - Requirements 2.4, 9.4
            self._analysis_running = False
            query_input.disabled = False
            query_input.focus()
    
    def action_quit(self) -> None:
        """Handle Ctrl+Q to quit the application.
        
        Requirements: 10.1 - Graceful exit
        Requirements: 10.3 - Cleanup background tasks on exit
        """
        # Cancel any running analysis task before exiting
        self._cleanup_tasks()
        self.exit()
    
    def action_cancel(self) -> None:
        """Handle Ctrl+C to cancel/clear.
        
        Requirements: 10.2 - Cancel current analysis and re-enable input
        Requirements: 9.4 - Re-enable input after errors
        """
        if self._analysis_running and self._current_analysis_task:
            # Cancel the running analysis task
            self._current_analysis_task.cancel()
            
            # Reset the UI state and re-enable input
            self._analysis_running = False
            query_input = self.query_one("#query-input", Input)
            query_input.disabled = False
            query_input.focus()
            
            status_panel = self.query_one("#status-panel", StatusPanel)
            status_panel.set_error("Analysis cancelled by user")
        else:
            # Clear the input
            query_input = self.query_one("#query-input", Input)
            query_input.value = ""
            query_input.focus()
    
    def _cleanup_tasks(self) -> None:
        """Cleanup any background tasks.
        
        Requirements: 10.3 - Cleanup background tasks on exit
        """
        if self._current_analysis_task and not self._current_analysis_task.done():
            self._current_analysis_task.cancel()
        self._current_analysis_task = None
        self._analysis_running = False
    
    async def on_unmount(self) -> None:
        """Handle application unmount event.
        
        Requirements: 10.3 - Cleanup background tasks on exit
        """
        self._cleanup_tasks()


def run_cli() -> None:
    """Run the CLI application.
    
    This is the main entry point for the CLI.
    It loads environment variables from .env and initializes the app.
    
    Requirements: 1.1, 8.3, 9.3 - Initialize agent components on startup
    """
    import sys
    import logging
    
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configure logging for CLI (less verbose than API)
    log_level = os.environ.get("LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.WARNING),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # Set specific log levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    
    # Initialize agent components on startup (same as FastAPI startup)
    # Requirements: 8.3 - Use existing agent orchestrator and Composio tool registration
    try:
        from main import initialize_components
        initialize_components()
    except Exception as e:
        # If initialization fails, the app will still run but show API key errors
        logging.getLogger(__name__).warning(f"Failed to initialize components: {e}")
    
    app = PortfolioApp()
    app.run()


# =============================================================================
# Exported Classes and Functions
# =============================================================================

__all__ = [
    "PortfolioApp",
    "run_cli",
    "check_api_keys",
    "format_error_message",
]
