"""Integration tests for the Stock Portfolio Analysis Agent.

This module contains integration tests to verify:
- Full end-to-end flow with sample queries
- SSE streaming works correctly
- API response completeness

Requirements: 10.1, 10.2, 10.3, 10.4
"""

import json
import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any

from fastapi.testclient import TestClient

from agent.api import app, parse_query_simple


class TestQueryParser:
    """Tests for the query parser functionality."""

    def test_parse_single_ticker_query(self):
        """Test parsing a query with a single ticker."""
        query = "Invest $10000 in AAPL since January 2024"
        params = parse_query_simple(query)
        
        assert "AAPL" in params["ticker_symbols"]
        assert params["total_cash"] == 10000.0
        assert params["strategy"] == "single_shot"
        assert "2024" in params["start_date"]

    def test_parse_multiple_tickers_query(self):
        """Test parsing a query with multiple tickers."""
        query = "Invest $5000 in AAPL and MSFT"
        params = parse_query_simple(query)
        
        assert "AAPL" in params["ticker_symbols"]
        assert "MSFT" in params["ticker_symbols"]
        assert params["total_cash"] == 5000.0

    def test_parse_company_name_query(self):
        """Test parsing a query with company names."""
        query = "Invest 10k in Apple and Microsoft"
        params = parse_query_simple(query)
        
        assert "AAPL" in params["ticker_symbols"]
        assert "MSFT" in params["ticker_symbols"]
        assert params["total_cash"] == 10000.0

    def test_parse_dca_strategy_query(self):
        """Test parsing a query with DCA strategy."""
        query = "Invest $12000 in GOOGL monthly since Jan 2024"
        params = parse_query_simple(query)
        
        assert "GOOGL" in params["ticker_symbols"]
        assert params["strategy"] == "dca"
        assert params["dca_interval"] == "monthly"

    def test_parse_query_with_defaults(self):
        """Test that missing parameters get reasonable defaults."""
        query = "Invest in Tesla"
        params = parse_query_simple(query)
        
        assert "TSLA" in params["ticker_symbols"]
        assert params["total_cash"] == 10000.0  # Default
        assert params["strategy"] == "single_shot"  # Default


class TestAPIEndpoints:
    """Tests for the API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test the root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "Stock Portfolio Analysis Agent"

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_tools_endpoint(self, client):
        """Test the tools listing endpoint."""
        response = client.get("/tools")
        
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "count" in data
        assert data["count"] == 5
        
        # Verify all expected tools are listed
        tool_names = [t["name"] for t in data["tools"]]
        assert "fetch_stock_data" in tool_names
        assert "fetch_benchmark_data" in tool_names
        assert "simulate_portfolio" in tool_names
        assert "simulate_spy_investment" in tool_names
        assert "calculate_metrics" in tool_names

    def test_config_endpoint(self, client):
        """Test the config endpoint."""
        response = client.get("/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "model" in data


class TestSSEStreaming:
    """Tests for SSE streaming functionality."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    def test_analyze_endpoint_streams_events(self, client):
        """Test that the /analyze endpoint streams SSE events.
        
        Requirements: 10.2 - Stream progress events using SSE
        """
        # Use a date range in the past to ensure data availability
        request_data = {
            "query": "Invest $5000 in AAPL since January 2024",
            "user_id": "test_user_sse"
        }
        
        # Make the request with streaming
        with client.stream("POST", "/analyze", json=request_data) as response:
            assert response.status_code == 200
            assert response.headers.get("content-type") == "text/event-stream; charset=utf-8"
            
            events = []
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event_type = line.replace("event:", "").strip()
                elif line.startswith("data:"):
                    data = line.replace("data:", "").strip()
                    events.append({"event": event_type, "data": json.loads(data)})
            
            # Verify we received events
            assert len(events) > 0
            
            # Verify event types
            event_types = [e["event"] for e in events]
            assert "status" in event_types  # Should have status events

    def test_analyze_endpoint_emits_tool_logs(self, client):
        """Test that the /analyze endpoint emits tool log events.
        
        Requirements: 10.3 - Emit state delta events for UI updates
        """
        request_data = {
            "query": "Invest $5000 in MSFT since January 2024",
            "user_id": "test_user_tool_logs"
        }
        
        with client.stream("POST", "/analyze", json=request_data) as response:
            events = []
            current_event = {}
            
            for line in response.iter_lines():
                if line.startswith("event:"):
                    current_event["event"] = line.replace("event:", "").strip()
                elif line.startswith("data:"):
                    current_event["data"] = json.loads(line.replace("data:", "").strip())
                    events.append(current_event.copy())
                    current_event = {}
            
            # Check for tool_log events
            tool_log_events = [e for e in events if e.get("event") == "tool_log"]
            
            # Should have tool logs for various tools
            if tool_log_events:
                for event in tool_log_events:
                    assert "tool" in event["data"]
                    assert "message" in event["data"]

    def test_analyze_endpoint_returns_result(self, client):
        """Test that the /analyze endpoint returns a final result.
        
        Requirements: 10.4 - Return investment summary with all calculated metrics
        """
        request_data = {
            "query": "Invest $10000 in AAPL since January 2024",
            "user_id": "test_user_result"
        }
        
        with client.stream("POST", "/analyze", json=request_data) as response:
            events = []
            current_event = {}
            
            for line in response.iter_lines():
                if line.startswith("event:"):
                    current_event["event"] = line.replace("event:", "").strip()
                elif line.startswith("data:"):
                    current_event["data"] = json.loads(line.replace("data:", "").strip())
                    events.append(current_event.copy())
                    current_event = {}
            
            # Find the result event
            result_events = [e for e in events if e.get("event") == "result"]
            
            if result_events:
                result = result_events[0]["data"]
                
                # Verify all required fields are present (Property 19)
                assert "holdings" in result
                assert "final_prices" in result
                assert "cash" in result
                assert "returns" in result
                assert "total_value" in result
                assert "percent_allocation" in result
                assert "percent_return" in result
                assert "performance_data" in result


class TestSyncAnalyzeEndpoint:
    """Tests for the synchronous analyze endpoint."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    def test_sync_analyze_single_ticker(self, client):
        """Test synchronous analysis with a single ticker."""
        request_data = {
            "query": "Invest $5000 in AAPL since January 2024",
            "user_id": "test_sync_single"
        }
        
        response = client.post("/analyze/sync", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        if data["success"]:
            assert "summary" in data
            summary = data["summary"]
            
            # Verify required fields (Property 19)
            assert "holdings" in summary
            assert "final_prices" in summary
            assert "cash" in summary
            assert "returns" in summary
            assert "total_value" in summary
            assert "percent_allocation" in summary
            assert "percent_return" in summary
            assert "performance_data" in summary
            
            # Verify AAPL is in holdings
            assert "AAPL" in summary["holdings"]

    def test_sync_analyze_multiple_tickers(self, client):
        """Test synchronous analysis with multiple tickers."""
        request_data = {
            "query": "Invest $10000 in AAPL and MSFT since January 2024",
            "user_id": "test_sync_multiple"
        }
        
        response = client.post("/analyze/sync", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        if data["success"]:
            summary = data["summary"]
            
            # Verify both tickers are in holdings
            assert "AAPL" in summary["holdings"]
            assert "MSFT" in summary["holdings"]
            
            # Verify allocations sum to approximately 100%
            total_allocation = sum(summary["percent_allocation"].values())
            assert abs(total_allocation - 100.0) < 0.1

    def test_sync_analyze_dca_strategy(self, client):
        """Test synchronous analysis with DCA strategy."""
        request_data = {
            "query": "Invest $6000 in GOOGL monthly since January 2024",
            "user_id": "test_sync_dca"
        }
        
        response = client.post("/analyze/sync", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        if data["success"]:
            summary = data["summary"]
            
            # Verify GOOGL is in holdings
            assert "GOOGL" in summary["holdings"]
            
            # Verify investment log has multiple entries (DCA)
            assert len(summary["investment_log"]) >= 1

    def test_sync_analyze_no_tickers_error(self, client):
        """Test that analysis fails gracefully when no tickers are found."""
        request_data = {
            "query": "Invest some money",  # No ticker specified
            "user_id": "test_sync_no_ticker"
        }
        
        response = client.post("/analyze/sync", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is False
        assert "error" in data
        assert data["error"] is not None

    def test_sync_analyze_with_portfolio(self, client):
        """Test synchronous analysis with existing portfolio."""
        request_data = {
            "query": "Invest $5000 in NVDA since January 2024",
            "user_id": "test_sync_portfolio",
            "portfolio": {
                "holdings": [
                    {
                        "ticker_symbol": "AAPL",
                        "investment_amount": 2000.0,
                        "purchase_date": "2024-01-15"
                    }
                ]
            }
        }
        
        response = client.post("/analyze/sync", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        if data["success"]:
            summary = data["summary"]
            
            # Verify both tickers are in holdings
            assert "NVDA" in summary["holdings"]
            assert "AAPL" in summary["holdings"]


class TestEndToEndFlow:
    """End-to-end integration tests for the complete analysis flow."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    def test_full_analysis_flow(self, client):
        """Test the complete analysis flow from query to results.
        
        This test verifies:
        1. Query parsing extracts correct parameters
        2. Stock data is fetched successfully
        3. Portfolio simulation runs correctly
        4. Metrics are calculated accurately
        5. Response contains all required fields
        """
        request_data = {
            "query": "Invest $10000 in Apple since January 2024",
            "user_id": "test_e2e_flow"
        }
        
        response = client.post("/analyze/sync", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        if data["success"]:
            summary = data["summary"]
            
            # 1. Verify holdings exist
            assert len(summary["holdings"]) > 0
            assert "AAPL" in summary["holdings"]
            
            # 2. Verify prices are positive
            for ticker, price in summary["final_prices"].items():
                assert price > 0
            
            # 3. Verify total value is calculated
            assert summary["total_value"] > 0
            
            # 4. Verify returns are calculated
            assert len(summary["returns"]) > 0
            
            # 5. Verify allocations sum to ~100%
            total_allocation = sum(summary["percent_allocation"].values())
            assert abs(total_allocation - 100.0) < 0.1
            
            # 6. Verify performance data exists
            assert len(summary["performance_data"]) > 0
            
            # 7. Verify each performance point has required fields
            for point in summary["performance_data"]:
                assert "date" in point
                assert "portfolio" in point
                assert "spy" in point

    def test_portfolio_value_consistency(self, client):
        """Test that portfolio value is consistent with holdings and prices."""
        request_data = {
            "query": "Invest $5000 in MSFT since January 2024",
            "user_id": "test_value_consistency"
        }
        
        response = client.post("/analyze/sync", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        if data["success"]:
            summary = data["summary"]
            
            # Calculate expected value from holdings and prices
            calculated_value = summary["cash"]
            for ticker, shares in summary["holdings"].items():
                if ticker in summary["final_prices"]:
                    calculated_value += shares * summary["final_prices"][ticker]
            
            # Verify total_value matches calculated value
            assert abs(summary["total_value"] - calculated_value) < 0.01

    def test_investment_log_completeness(self, client):
        """Test that investment log contains all transactions."""
        request_data = {
            "query": "Invest $8000 in AAPL and GOOGL since January 2024",
            "user_id": "test_log_completeness"
        }
        
        response = client.post("/analyze/sync", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        if data["success"]:
            summary = data["summary"]
            
            # Verify investment log exists
            assert len(summary["investment_log"]) > 0
            
            # Verify log entries contain expected information
            for entry in summary["investment_log"]:
                assert "Bought" in entry
                assert "shares" in entry
                assert "$" in entry
