import requests
from unittest.mock import patch

BASE_URL = "http://127.0.0.1:8000"

def test_get_asset_summary_fallback():
    """Tests the /api/v1/assets/summary endpoint fallback mechanism."""
    response = requests.get(f"{BASE_URL}/api/v1/assets/summary", params={"project_id": "test-project"})
    assert response.status_code == 200
    assert response.json()["source"] == "sample_data"

def test_agent_chat():
    """Tests the /api/v1/agent/chat endpoint."""
    response = requests.post(f"{BASE_URL}/api/v1/agent/chat", json={"message": "hello"})
    assert response.status_code == 200

def test_get_recommendations_fallback():
    """Tests the /api/v1/recommendations endpoint fallback mechanism."""
    response = requests.post(f"{BASE_URL}/live", json={"project_id": "test-project"})
    assert response.status_code == 200
    assert response.json()["source"] == "google_cloud_recommender"