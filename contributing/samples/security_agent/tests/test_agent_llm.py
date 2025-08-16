import pytest
from fastapi.testclient import TestClient
from backend.api.agent_llm import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)

def test_get_agent_info():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "agent_info" in data

def test_list_available_tools():
    response = client.get("/tools")
    assert response.status_code == 200
    data = response.json()
    assert "agents_and_tools" in data
    assert "ADK Built-in Tools" in data["agents_and_tools"]

def test_create_session():
    payload = {"user_id": "test_user", "project_id": "test_project"}
    response = client.post("/sessions/create", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "session_id" in data

def test_chat_with_llm_agent():
    # Create session first
    payload = {"user_id": "test_user", "project_id": "test_project"}
    session_resp = client.post("/sessions/create", json=payload)
    session_id = session_resp.json()["session_id"]
    chat_payload = {
        "query": "Show me storage security analysis",
        "user_id": "test_user",
        "project_id": "test_project",
        "session_id": session_id
    }
    response = client.post("/chat", json=chat_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "response" in data
    assert "agent_used" in data

def test_get_session_messages():
    payload = {"user_id": "test_user", "project_id": "test_project"}
    session_resp = client.post("/sessions/create", json=payload)
    session_id = session_resp.json()["session_id"]
    response = client.get(f"/sessions/{session_id}/messages")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "messages" in data

def test_get_session_status():
    payload = {"user_id": "test_user", "project_id": "test_project"}
    session_resp = client.post("/sessions/create", json=payload)
    session_id = session_resp.json()["session_id"]
    response = client.get(f"/sessions/{session_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "analytics" in data

def test_get_adk_status():
    response = client.get("/adk/status")
    assert response.status_code == 200
    data = response.json()
    assert "adk_available" in data
