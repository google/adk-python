import os
from unittest.mock import patch, MagicMock
import pytest
from frontend.unified_api_client import UnifiedAPIClient

# Set up the API client for testing
@pytest.fixture(scope="module")
def api_client():
    """Fixture to provide a configured API client."""
    client = UnifiedAPIClient()
    client.backend_url = "http://127.0.0.1:8000"
    return client

@patch('requests.Session.request')
def test_chat_with_agent_success(mock_request, api_client):
    """Test successful chat with agent."""
    # Mock the response from the backend
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Test chat response"}
    mock_request.return_value = mock_response

    # Call the chat method
    response = api_client.chat_with_agent("hello")

    # Assertions
    assert response["response"] == "Test chat response"
    mock_request.assert_called_with(
        method='POST',
        url=f"{api_client.backend_url}/api/v1/agent/chat",
        json={'message': 'hello', 'project_id': os.getenv('GOOGLE_CLOUD_PROJECT')},
        params=None,
        timeout=30
    )
