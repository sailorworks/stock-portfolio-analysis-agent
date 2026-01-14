"""CLI frontend for Stock Portfolio Analysis Agent.

This module provides a Textual-based terminal interface that integrates
with the existing Composio + OpenAI agent architecture.
"""

from agent.cli.app import PortfolioApp
from agent.cli.client import AgentClient, AnalysisResult

__all__ = ["PortfolioApp", "AgentClient", "AnalysisResult"]
