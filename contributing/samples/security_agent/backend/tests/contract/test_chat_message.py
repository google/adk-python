"""
Contract test for POST /api/v1/chat/message endpoint.
This test MUST FAIL until the implementation is fixed.
"""

import pytest
import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"
ENDPOINT = "/api/v1/chat/message"


class TestChatMessageContract:
    """Test the chat message endpoint contract."""

    def test_endpoint_exists(self):
        """Test that the endpoint exists and responds."""
        response = requests.post(
            f"{BASE_URL}{ENDPOINT}",
            json={"message": "test"},
            timeout=5
        )
        # Should not return 404
        assert response.status_code != 404, "Endpoint does not exist"

    def test_request_schema_validation(self):
        """Test request schema validation."""
        # Missing required field
        response = requests.post(
            f"{BASE_URL}{ENDPOINT}",
            json={},
            timeout=5
        )
        assert response.status_code == 400, "Should reject empty request"

        # Invalid message type
        response = requests.post(
            f"{BASE_URL}{ENDPOINT}",
            json={"message": 123},  # Should be string
            timeout=5
        )
        assert response.status_code == 400, "Should reject non-string message"

    def test_successful_chat_message(self):
        """Test successful chat message with database query."""
        payload = {
            "message": "Show me high severity security findings",
            "session_id": "test-session-123",
            "user_id": "test-user"
        }

        response = requests.post(
            f"{BASE_URL}{ENDPOINT}",
            json=payload,
            timeout=30
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()

        # Validate response schema
        assert "response" in data, "Response must contain 'response' field"
        assert "success" in data, "Response must contain 'success' field"
        assert isinstance(data["success"], bool), "'success' must be boolean"
        assert isinstance(data["response"], str), "'response' must be string"

        # Validate that actual data is returned (not empty or error)
        assert data["success"] is True, "Query should succeed"
        assert len(data["response"]) > 0, "Response should not be empty"
        assert "high" in data["response"].lower() or "critical" in data["response"].lower(), \
            "Response should contain security findings"

    def test_response_includes_metadata(self):
        """Test that response includes metadata about execution."""
        payload = {
            "message": "List all assets",
            "session_id": "test-meta-session"
        }

        response = requests.post(
            f"{BASE_URL}{ENDPOINT}",
            json=payload,
            timeout=30
        )

        assert response.status_code == 200
        data = response.json()

        # Optional but recommended fields
        if "execution_time" in data:
            assert isinstance(data["execution_time"], (int, float)), \
                "'execution_time' must be numeric"
            assert data["execution_time"] > 0, "Execution time must be positive"

        if "agent_used" in data:
            assert isinstance(data["agent_used"], bool), "'agent_used' must be boolean"

        if "model" in data:
            assert isinstance(data["model"], str), "'model' must be string"

    def test_session_context_preserved(self):
        """Test that session context is maintained across requests."""
        session_id = "context-test-session"

        # First request
        response1 = requests.post(
            f"{BASE_URL}{ENDPOINT}",
            json={
                "message": "Show me compute instances",
                "session_id": session_id
            },
            timeout=30
        )
        assert response1.status_code == 200

        # Second request in same session
        response2 = requests.post(
            f"{BASE_URL}{ENDPOINT}",
            json={
                "message": "Show only the running ones",  # Refers to previous context
                "session_id": session_id
            },
            timeout=30
        )
        assert response2.status_code == 200

        data2 = response2.json()
        assert data2["success"] is True
        # Response should understand context from first request
        assert "running" in data2["response"].lower() or "status" in data2["response"].lower()

    def test_error_handling(self):
        """Test error response format."""
        # Send a query that might cause an error
        response = requests.post(
            f"{BASE_URL}{ENDPOINT}",
            json={
                "message": "INVALID_QUERY_TYPE_12345",  # Unlikely to match any handler
                "session_id": "error-test"
            },
            timeout=30
        )

        # Even errors should return structured response
        assert response.status_code in [200, 400, 500]
        data = response.json()

        if response.status_code != 200:
            assert "error" in data or "message" in data, \
                "Error response must contain error details"

    def test_database_connection_works(self):
        """Test that database queries actually return data."""
        queries = [
            "Show security findings",
            "List storage buckets",
            "Show IAM accounts",
            "List GKE clusters"
        ]

        for query in queries:
            response = requests.post(
                f"{BASE_URL}{ENDPOINT}",
                json={"message": query},
                timeout=30
            )

            assert response.status_code == 200, f"Query '{query}' failed"
            data = response.json()
            assert data["success"] is True, f"Query '{query}' was not successful"
            assert len(data["response"]) > 0, f"Query '{query}' returned empty response"
            # Should NOT contain common error messages
            assert "error" not in data["response"].lower(), \
                f"Query '{query}' returned error in response"
            assert "not found" not in data["response"].lower(), \
                f"Query '{query}' indicates missing data"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])