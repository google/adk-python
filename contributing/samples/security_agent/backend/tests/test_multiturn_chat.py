import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from backend.main import app

client = TestClient(app)

def test_multi_turn_conversation():
    """Test that the agent can handle a multi-turn conversation and maintain context."""
    # Start a new session
    from unittest.mock import patch
    import uuid

    # Mock the session creation endpoint
    with patch('backend.api.sessions.session_manager.create_session') as mock_create_session:
        mock_session_id = str(uuid.uuid4())
        mock_create_session.return_value.id = mock_session_id
        mock_create_session.return_value.user_id = "test_user"
        mock_create_session.return_value.created_at = "2025-01-01T00:00:00Z"
        mock_create_session.return_value.updated_at = "2025-01-01T00:00:00Z"
        mock_create_session.return_value.expires_at = "2025-01-02T00:00:00Z"
        mock_create_session.return_value.is_active = True
        mock_create_session.return_value.context = {}
        mock_create_session.return_value.metadata = {}

        from unittest.mock import patch
        import uuid
        from backend.services.session_manager import Session, Message
    
        # Mock the session creation endpoint
        with patch('backend.api.sessions.session_manager.create_session') as mock_create_session:
            mock_session_id = str(uuid.uuid4())
            mock_session = Session(
                id=mock_session_id,
                user_id="test_user",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
                expires_at="2025-01-02T00:00:00Z",
                is_active=True,
                context={},
                metadata={}
            )
            mock_create_session.return_value = mock_session
    
            response = client.post("/api/v1/sessions/create", json={"user_id": "test_user"})
            assert response.status_code == 200
            session_id = response.json()["id"]
            assert session_id == mock_session_id
        assert session_id == mock_session_id

    # First turn
    response = client.post(
        "/api/v1/chat/message",
        json={"query": "list my storage buckets", "session_id": session_id, "user_id": "test_user"},
    )
    assert response.status_code == 200
    first_response = response.json()["response"]
    assert "bucket" in first_response.lower()

    # Second turn
    response = client.post(
        "/api/v1/chat/message",
        json={"query": "which of those are public?", "session_id": session_id, "user_id": "test_user"},
    )
    assert response.status_code == 200
    second_response = response.json()["response"]
    assert "public" in second_response.lower()