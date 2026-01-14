"""Agent client wrapper for CLI integration.

This module provides a thin wrapper around the existing agent architecture,
calling run_agent_analysis() from agent/api.py.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import logging
from dataclasses import dataclass
from typing import Optional

from agent.api import run_agent_analysis, build_summary_from_tool_results
from agent.insights import generate_insights_from_summary
from agent.models import InvestmentSummary


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of portfolio analysis from agent.
    
    Attributes:
        success: Whether the analysis was successful
        summary: The InvestmentSummary if successful, None otherwise
        error: Error message if analysis failed, None otherwise
    """
    success: bool
    summary: Optional[InvestmentSummary]
    error: Optional[str]


class AgentClient:
    """Client that calls existing Composio + OpenAI agent.
    
    This is a thin wrapper around the existing agent architecture.
    It calls run_agent_analysis() from agent/api.py and returns
    the resulting InvestmentSummary.
    
    Requirements:
        8.1 - Call existing run_agent_analysis() from agent/api.py
        8.2 - Pass user query and receive InvestmentSummary response
        8.3 - Use existing agent orchestrator and Composio tool registration
        8.4 - Leverage OpenAI for natural language understanding
        8.5 - Propagate errors from agent to CLI_App
    """
    
    def __init__(self):
        """Initialize the AgentClient."""
        pass
    
    async def analyze(
        self,
        query: str,
        user_id: str = "cli_user",
    ) -> AnalysisResult:
        """Run analysis using existing run_agent_analysis().
        
        This calls the same code path as the FastAPI /analyze endpoint,
        ensuring Composio + OpenAI handle all the heavy lifting.
        
        Args:
            query: Natural language investment query
            user_id: User ID for session management (default: "cli_user")
            
        Returns:
            AnalysisResult with success status, summary, and error info
        """
        try:
            logger.info(f"Starting analysis for query: {query}")
            
            # Call the existing run_agent_analysis function
            # Requirements: 8.1, 8.3, 8.4
            result = await run_agent_analysis(
                query=query,
                user_id=user_id,
            )
            
            # Check if the agent execution was successful
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error occurred")
                logger.error(f"Agent analysis failed: {error_msg}")
                # Requirement 8.5 - Propagate errors
                return AnalysisResult(
                    success=False,
                    summary=None,
                    error=error_msg,
                )
            
            # Build summary from tool results
            # Requirement 8.2 - Receive InvestmentSummary response
            tool_results = result.get("tool_results", {})
            summary = build_summary_from_tool_results(tool_results)
            
            if summary is None:
                logger.warning("Could not build summary from tool results")
                return AnalysisResult(
                    success=False,
                    summary=None,
                    error="Failed to build investment summary from analysis results",
                )
            
            # Generate insights for the summary
            try:
                insights = await generate_insights_from_summary(summary)
                if insights:
                    summary.insights = insights
                    logger.info("Successfully generated insights")
            except Exception as e:
                # Insights generation failure is non-fatal
                logger.warning(f"Failed to generate insights: {e}")
            
            logger.info(f"Analysis complete. Total value: {summary.total_value}")
            
            return AnalysisResult(
                success=True,
                summary=summary,
                error=None,
            )
            
        except Exception as e:
            logger.exception(f"Error during analysis: {e}")
            # Requirement 8.5 - Propagate errors
            return AnalysisResult(
                success=False,
                summary=None,
                error=str(e),
            )


# =============================================================================
# Exported Classes
# =============================================================================

__all__ = [
    "AgentClient",
    "AnalysisResult",
]
