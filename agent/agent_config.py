"""Agent configuration for the Stock Portfolio Analysis Agent.

This module provides agent setup and configuration including:
- Setting up the agent with system prompt
- Attaching all custom tools
- Configuring the model (gpt-4o-mini)

Requirements: 9.4
"""

from typing import Any, Dict, List, Optional

from agents import Agent

from agent.portfolio import Portfolio, get_portfolio_manager
from agent.prompts import get_query_parser_prompt, get_insights_prompt
from agent.session import SessionManager, get_session_manager


# Default model for the agent
DEFAULT_MODEL = "gpt-4o-mini"


def create_portfolio_agent(
    user_id: str,
    session_manager: Optional[SessionManager] = None,
    model: str = DEFAULT_MODEL,
    portfolio: Optional[Portfolio] = None,
) -> Agent:
    """Create a Stock Portfolio Analysis Agent for a user.
    
    Requirements: 9.4 - Tool Router handles LLM interactions for query parsing
                        and insights generation
    
    This function creates an OpenAI Agent configured with:
    - System prompt for query parsing with portfolio context
    - All custom tools registered with Composio
    - The specified model (default: gpt-4o-mini)
    
    Args:
        user_id: Unique identifier for the user
        session_manager: Optional SessionManager instance. If not provided,
                        uses the global session manager.
        model: The model to use for the agent (default: gpt-4o-mini)
        portfolio: Optional Portfolio object for context. If not provided,
                  retrieves from PortfolioManager.
        
    Returns:
        Configured Agent instance ready for use
    """
    # Get or create session manager
    if session_manager is None:
        session_manager = get_session_manager()
    
    # Get portfolio context for the system prompt
    if portfolio is None:
        portfolio_manager = get_portfolio_manager()
        portfolio = portfolio_manager.get_portfolio(user_id)
    
    # Format portfolio holdings for the prompt
    holdings_list = [
        {
            "ticker_symbol": h.ticker_symbol,
            "investment_amount": h.investment_amount,
            "purchase_date": h.purchase_date,
        }
        for h in portfolio.holdings
    ]
    total_invested = portfolio.get_total_invested()
    
    # Get the system prompt with portfolio context
    system_prompt = get_query_parser_prompt(
        holdings=holdings_list,
        total_invested=total_invested,
    )
    
    # Get tools from the session
    tools = session_manager.get_tools(user_id)
    
    # Create the agent
    agent = Agent(
        name="Stock Portfolio Analyst",
        instructions=system_prompt,
        model=model,
        tools=tools,
    )
    
    return agent


def create_insights_agent(
    user_id: str,
    session_manager: Optional[SessionManager] = None,
    model: str = DEFAULT_MODEL,
) -> Agent:
    """Create an Insights Generator Agent for a user.
    
    Requirements: 9.4 - Tool Router handles LLM interactions for insights generation
    
    This function creates an OpenAI Agent configured with:
    - System prompt for insights generation
    - The specified model (default: gpt-4o-mini)
    
    Note: This agent doesn't need tools as it only generates insights
    based on provided portfolio data.
    
    Args:
        user_id: Unique identifier for the user
        session_manager: Optional SessionManager instance
        model: The model to use for the agent (default: gpt-4o-mini)
        
    Returns:
        Configured Agent instance for insights generation
    """
    # Get the insights system prompt
    system_prompt = get_insights_prompt()
    
    # Create the agent (no tools needed for insights generation)
    agent = Agent(
        name="Investment Insights Generator",
        instructions=system_prompt,
        model=model,
        tools=[],  # No tools needed for insights generation
    )
    
    return agent


class AgentOrchestrator:
    """Orchestrates the Stock Portfolio Analysis workflow.
    
    This class manages the complete workflow:
    1. Query parsing → data fetching → calculation → insights
    
    Requirements: 9.1, 9.4
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        """Initialize the agent orchestrator.
        
        Args:
            api_key: Optional Composio API key
            model: The model to use for agents (default: gpt-4o-mini)
        """
        self._session_manager = get_session_manager(api_key)
        self._model = model
        self._agents: Dict[str, Agent] = {}
    
    def get_portfolio_agent(
        self,
        user_id: str,
        portfolio: Optional[Portfolio] = None,
    ) -> Agent:
        """Get or create a portfolio analysis agent for a user.
        
        Args:
            user_id: Unique identifier for the user
            portfolio: Optional Portfolio object for context
            
        Returns:
            Configured Agent instance
        """
        # Create a new agent with current portfolio context
        # We always create a new agent to ensure the system prompt
        # has the latest portfolio context
        agent = create_portfolio_agent(
            user_id=user_id,
            session_manager=self._session_manager,
            model=self._model,
            portfolio=portfolio,
        )
        self._agents[user_id] = agent
        return agent
    
    def get_insights_agent(self, user_id: str) -> Agent:
        """Get or create an insights generator agent for a user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Configured Agent instance for insights generation
        """
        insights_key = f"{user_id}_insights"
        if insights_key not in self._agents:
            self._agents[insights_key] = create_insights_agent(
                user_id=user_id,
                session_manager=self._session_manager,
                model=self._model,
            )
        return self._agents[insights_key]
    
    def get_session_manager(self) -> SessionManager:
        """Get the session manager.
        
        Returns:
            The SessionManager instance
        """
        return self._session_manager


# Global orchestrator instance
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator(
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> AgentOrchestrator:
    """Get or create the global agent orchestrator instance.
    
    Args:
        api_key: Optional Composio API key
        model: The model to use for agents
        
    Returns:
        The global AgentOrchestrator instance
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator(api_key=api_key, model=model)
    return _orchestrator
