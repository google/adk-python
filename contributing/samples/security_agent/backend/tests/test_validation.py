from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_chat_message_validation():
    # Test with a valid request
    response = client.post("/api/v1/chat/message", json={"query": "hello", "session_id": "123", "user_id": "test"})
    assert response.status_code == 200

    # Test with an invalid request (missing query)
    response = client.post("/api/v1/chat/message", json={"session_id": "123", "user_id": "test"})
    assert response.status_code == 422

    # Test with an invalid request (query too long)
    response = client.post("/api/v1/chat/message", json={"query": "a" * 501, "session_id": "123", "user_id": "test"})
    assert response.status_code == 422
