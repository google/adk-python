import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_chat_with_valid_request():
    response = client.post("/api/v1/agent/chat", json={"query": "Hello", "session_id": "123"})
    assert response.status_code == 200
    assert "response" in response.json()

def test_chat_with_invalid_request():
    response = client.post("/api/v1/agent/chat", json={"query": "Hello"})
    assert response.status_code == 422