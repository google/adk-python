import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_discover_assets_with_valid_request():
    response = client.post("/api/v1/assets/discover", json={"query": "discover all assets"})
    assert response.status_code == 200
    assert "data" in response.json()

def test_discover_assets_with_invalid_request():
    response = client.post("/api/v1/assets/discover", json={})
    assert response.status_code == 422

def test_search_assets_with_valid_request():
    response = client.post("/api/v1/assets/search", json={"asset_name": "test", "asset_type": "test"})
    assert response.status_code == 200
    assert "data" in response.json()

def test_search_assets_with_invalid_request():
    response = client.post("/api/v1/assets/search", json={})
    assert response.status_code == 422