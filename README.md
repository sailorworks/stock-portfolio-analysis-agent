# Composio Stock Portfolio Analysis Agent

A stock portfolio analysis agent built with Composio Tool Router. The agent analyzes investment opportunities, tracks stock performance, simulates portfolio allocations, and generates insights.

## Features

- Natural language investment query parsing
- Historical stock data fetching via yfinance
- Portfolio simulation (single-shot and DCA strategies)
- Performance metrics calculation
- S&P 500 benchmark comparison
- Bull/bear insights generation
- Real-time streaming API with SSE

## Setup

1. Create a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

4. Run the server:
   ```bash
   uv run uvicorn agent.main:app --reload
   ```

## Environment Variables

- `COMPOSIO_API_KEY`: Your Composio API key
- `OPENAI_API_KEY`: Your OpenAI API key

## Project Structure

```
stock-portfolio-analysis-agent/
├── agent/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── models.py         # Pydantic data models
│   ├── tools.py          # Custom Composio tools
│   ├── session.py        # Session management
│   └── prompts.py        # System prompts
├── tests/
│   ├── __init__.py
│   └── test_*.py         # Property and unit tests
├── pyproject.toml
├── .env.example
└── README.md
```
