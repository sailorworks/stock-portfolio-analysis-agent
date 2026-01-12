"""Insights generation for the Stock Portfolio Analysis Agent.

This module provides LLM-powered insights generation functionality including:
- Creating an insights agent with portfolio context
- Running the agent to generate bull/bear insights
- Parsing LLM responses into structured Insights model

Requirements: 4.1, 4.2, 4.3, 4.4
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from agents import Agent, Runner

from agent.models import Insight, Insights
from agent.prompts import get_insights_prompt_with_context


# Configure logging
logger = logging.getLogger(__name__)

# Default model for insights generation
DEFAULT_MODEL = "gpt-4o-mini"


def parse_insights_from_response(response_text: str) -> Optional[Insights]:
    """Parse LLM response text into an Insights model.
    
    Requirements: 4.4 - Each insight SHALL contain title, description, and emoji fields
    
    Args:
        response_text: Raw text response from the LLM
        
    Returns:
        Insights model if parsing succeeds, None otherwise
    """
    try:
        # Try to find JSON in the response
        # Look for JSON block that contains bull_insights and bear_insights
        json_match = re.search(r'\{[\s\S]*"bull_insights"[\s\S]*"bear_insights"[\s\S]*\}', response_text)
        
        if not json_match:
            # Try alternative pattern - JSON might be in a code block
            code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response_text)
            if code_block_match:
                json_str = code_block_match.group(1)
            else:
                # Try to find any JSON object
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    json_str = json_match.group()
                else:
                    logger.warning("No JSON found in insights response")
                    return None
        else:
            json_str = json_match.group()
        
        # Parse the JSON
        data = json.loads(json_str)
        
        # Extract bull insights
        bull_insights = []
        for item in data.get("bull_insights", []):
            if isinstance(item, dict):
                # Validate required fields (Requirement 4.4)
                title = item.get("title", "").strip()
                description = item.get("description", "").strip()
                emoji = item.get("emoji", "📈").strip()
                
                if title and description:
                    bull_insights.append(Insight(
                        title=title,
                        description=description,
                        emoji=emoji if emoji else "📈",
                    ))
        
        # Extract bear insights
        bear_insights = []
        for item in data.get("bear_insights", []):
            if isinstance(item, dict):
                # Validate required fields (Requirement 4.4)
                title = item.get("title", "").strip()
                description = item.get("description", "").strip()
                emoji = item.get("emoji", "⚠️").strip()
                
                if title and description:
                    bear_insights.append(Insight(
                        title=title,
                        description=description,
                        emoji=emoji if emoji else "⚠️",
                    ))
        
        # Requirement 4.1, 4.3 - Must have at least one bull and one bear insight
        if not bull_insights or not bear_insights:
            logger.warning(f"Incomplete insights: {len(bull_insights)} bull, {len(bear_insights)} bear")
            # Still return what we have if we got at least something
            if bull_insights or bear_insights:
                return Insights(
                    bull_insights=bull_insights,
                    bear_insights=bear_insights,
                )
            return None
        
        return Insights(
            bull_insights=bull_insights,
            bear_insights=bear_insights,
        )
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse insights JSON: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error parsing insights: {e}")
        return None


async def generate_insights(
    tickers: List[str],
    holdings: Dict[str, float],
    current_prices: Dict[str, float],
    invested_amounts: Dict[str, float],
    returns: Dict[str, float],
    percent_returns: Dict[str, float],
    allocations: Dict[str, float],
    total_value: float,
    spy_return: Optional[float] = None,
    model: str = DEFAULT_MODEL,
) -> Optional[Insights]:
    """Generate bull and bear insights for a portfolio using LLM.
    
    Requirements: 4.1, 4.2 - Generate bullish and bearish insights using
                            the insights prompt with portfolio data injected
    
    This function:
    1. Creates an insights agent with the insights prompt containing portfolio data
    2. Calls Runner.run to generate insights
    3. Parses the LLM response into an Insights model
    
    Args:
        tickers: List of ticker symbols in the portfolio
        holdings: Shares held per ticker
        current_prices: Current price per ticker
        invested_amounts: Amount invested per ticker
        returns: Absolute returns per ticker
        percent_returns: Percentage returns per ticker
        allocations: Allocation percentage per ticker
        total_value: Total portfolio value
        spy_return: SPY benchmark return percentage (optional)
        model: The model to use for the agent (default: gpt-4o-mini)
        
    Returns:
        Insights model with bull and bear insights, or None if generation fails
    """
    try:
        # Get the insights prompt with portfolio data context (Requirement 4.2)
        system_prompt = get_insights_prompt_with_context(
            tickers=tickers,
            holdings=holdings,
            current_prices=current_prices,
            invested_amounts=invested_amounts,
            returns=returns,
            percent_returns=percent_returns,
            allocations=allocations,
            total_value=total_value,
            spy_return=spy_return,
        )
        
        # Create the insights agent (Requirement 4.1)
        insights_agent = Agent(
            name="Investment Insights Generator",
            instructions=system_prompt,
            model=model,
            tools=[],  # No tools needed for insights generation
        )
        
        # Run the agent to generate insights
        result = await Runner.run(
            insights_agent,
            "Generate bull and bear insights for this portfolio based on the data provided."
        )
        
        # Get the final output
        final_output = result.final_output if hasattr(result, 'final_output') else str(result)
        
        logger.info(f"Insights agent response length: {len(final_output)}")
        
        # Parse the response into Insights model
        insights = parse_insights_from_response(final_output)
        
        if insights:
            logger.info(f"Generated {len(insights.bull_insights)} bull and {len(insights.bear_insights)} bear insights")
        else:
            logger.warning("Failed to parse insights from agent response")
        
        return insights
        
    except Exception as e:
        logger.exception(f"Error generating insights: {e}")
        return None


async def generate_insights_from_summary(
    summary: Any,
    spy_return: Optional[float] = None,
    model: str = DEFAULT_MODEL,
) -> Optional[Insights]:
    """Generate insights from an InvestmentSummary object.
    
    Requirements: 4.1, 4.2, 4.3 - Generate insights after portfolio analysis
    
    This is a convenience function that extracts the necessary data from
    an InvestmentSummary and calls generate_insights.
    
    Args:
        summary: InvestmentSummary object with portfolio data
        spy_return: Optional SPY benchmark return percentage
        model: The model to use for the agent
        
    Returns:
        Insights model with bull and bear insights, or None if generation fails
    """
    try:
        # Extract tickers from holdings
        tickers = list(summary.holdings.keys())
        
        if not tickers:
            logger.warning("No tickers in summary, cannot generate insights")
            return None
        
        # Calculate invested amounts from returns and percent returns
        invested_amounts = {}
        for ticker in tickers:
            ret = summary.returns.get(ticker, 0)
            pct_ret = summary.percent_return.get(ticker, 0)
            if pct_ret != 0:
                # invested = return / (percent_return / 100)
                invested_amounts[ticker] = ret / (pct_ret / 100) if pct_ret != 0 else 0
            else:
                # If no return, estimate from current value
                shares = summary.holdings.get(ticker, 0)
                price = summary.final_prices.get(ticker, 0)
                invested_amounts[ticker] = shares * price - ret
        
        # Calculate SPY return if we have performance data
        calculated_spy_return = spy_return
        if calculated_spy_return is None and summary.performance_data:
            # Calculate SPY return from first to last performance point
            first_point = summary.performance_data[0]
            last_point = summary.performance_data[-1]
            if first_point.spy > 0:
                calculated_spy_return = ((last_point.spy - first_point.spy) / first_point.spy) * 100
        
        return await generate_insights(
            tickers=tickers,
            holdings=summary.holdings,
            current_prices=summary.final_prices,
            invested_amounts=invested_amounts,
            returns=summary.returns,
            percent_returns=summary.percent_return,
            allocations=summary.percent_allocation,
            total_value=summary.total_value,
            spy_return=calculated_spy_return,
            model=model,
        )
        
    except Exception as e:
        logger.exception(f"Error generating insights from summary: {e}")
        return None


# =============================================================================
# Exported Functions
# =============================================================================

__all__ = [
    "generate_insights",
    "generate_insights_from_summary",
    "parse_insights_from_response",
]
