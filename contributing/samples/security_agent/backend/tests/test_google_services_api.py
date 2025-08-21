"""
Tests for the Google Services API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# Add the backend path to sys.path to allow for absolute imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.main import app 
from backend.services.google_service_analyzer import ServiceProfile

@pytest.fixture
def client():
    """Create a FastAPI TestClient instance."""
    return TestClient(app)

@patch('backend.services.google_service_analyzer.GoogleServiceAnalyzer._fetch_real_service_data')
def test_evaluate_new_service_endpoint(mock_fetch_real_data, client):
    """
    Test the /evaluate endpoint with mocked GCP calls.
    This test is fast and reliable.
    """
    # Arrange: Configure the mock to return predictable data instantly
    mock_fetch_real_data.return_value = {
        "is_enabled": True,
        "iam_permissions": ["test.permission.get", "test.permission.list"]
    }

    service_name = "test-service.googleapis.com"
    request_payload = {
        "service_name": service_name,
        "project_id": "test-project"
    }

    # Act: Call the API endpoint using the TestClient
    response = client.post("/api/v1/google-services/evaluate", json=request_payload)

    # Assert: Verify the response and behavior
    assert response.status_code == 200
    profile = ServiceProfile(**response.json())
    
    assert profile.service_name == service_name
    assert profile.is_enabled is True
    assert "test.permission.get" in profile.security_assessment.iam_permissions
    
    # Verify that our mock was called, confirming we didn't make a real API call
    mock_fetch_real_data.assert_called_once_with(service_name, "test-project")

def test_list_evaluations_endpoint(client):
    """
    Test the /evaluations/list endpoint.
    This test uses the real (but local) SQLite database.
    """
    # Arrange: First, add an evaluation to the database
    with patch('backend.services.google_service_analyzer.GoogleServiceAnalyzer._fetch_real_service_data') as mock_fetch:
        mock_fetch.return_value = {"is_enabled": True, "iam_permissions": []}
        client.post("/api/v1/google-services/evaluate", json={"service_name": "service-to-list.googleapis.com", "project_id": "test-project"})

    # Act: Call the list endpoint
    response = client.get("/api/v1/google-services/evaluations/list")

    # Assert
    assert response.status_code == 200
    evaluations = [ServiceProfile(**item) for item in response.json()]
    assert any(e.service_name == "service-to-list.googleapis.com" for e in evaluations)
