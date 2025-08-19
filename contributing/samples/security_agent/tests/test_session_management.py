"""
Test suite for Session Management Service (STORY-013)
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.session_manager import SessionManager, Session, Message


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_path = tmp.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def session_manager(temp_db):
    """Create a SessionManager instance with temporary database"""
    return SessionManager(db_path=temp_db, session_ttl_hours=24)


class TestSessionManager:
    """Test SessionManager functionality"""
    
    def test_create_session(self, session_manager):
        """Test session creation"""
        session = session_manager.create_session(user_id="test_user")
        
        assert session.id is not None
        assert session.user_id == "test_user"
        assert session.is_active is True
        assert session.expires_at > datetime.utcnow()
    
    def test_get_session(self, session_manager):
        """Test retrieving a session"""
        created_session = session_manager.create_session(user_id="test_user")
        retrieved_session = session_manager.get_session(created_session.id)
        
        assert retrieved_session is not None
        assert retrieved_session.id == created_session.id
        assert retrieved_session.user_id == "test_user"
    
    def test_update_session_context(self, session_manager):
        """Test updating session context"""
        session = session_manager.create_session(user_id="test_user")
        
        context = {"last_action": "security_scan", "findings": 5}
        success = session_manager.update_session(session.id, context=context)
        
        assert success is True
        
        updated_session = session_manager.get_session(session.id)
        assert updated_session.context["last_action"] == "security_scan"
        assert updated_session.context["findings"] == 5
    
    def test_add_message(self, session_manager):
        """Test adding messages to a session"""
        session = session_manager.create_session(user_id="test_user")
        
        message1 = session_manager.add_message(
            session.id, 
            "user", 
            "What are my security vulnerabilities?"
        )
        
        message2 = session_manager.add_message(
            session.id,
            "assistant",
            "I found 3 critical vulnerabilities in your GCP project."
        )
        
        assert message1.role == "user"
        assert message2.role == "assistant"
        assert message1.session_id == session.id
        assert message2.session_id == session.id
    
    def test_get_conversation_history(self, session_manager):
        """Test retrieving conversation history"""
        session = session_manager.create_session(user_id="test_user")
        
        # Add some messages
        session_manager.add_message(session.id, "user", "Hello")
        session_manager.add_message(session.id, "assistant", "Hi there!")
        session_manager.add_message(session.id, "user", "Check security")
        session_manager.add_message(session.id, "assistant", "Running scan...")
        
        # Get history
        history = session_manager.get_conversation_history(session.id)
        
        assert len(history) == 4
        assert history[0].content == "Hello"
        assert history[1].content == "Hi there!"
        assert history[2].content == "Check security"
        assert history[3].content == "Running scan..."
    
    def test_conversation_history_limit(self, session_manager):
        """Test limiting conversation history"""
        session = session_manager.create_session(user_id="test_user")
        
        # Add 10 messages
        for i in range(10):
            session_manager.add_message(session.id, "user", f"Message {i}")
        
        # Get limited history
        history = session_manager.get_conversation_history(session.id, limit=5)
        
        assert len(history) == 5
        assert history[0].content == "Message 0"
        assert history[4].content == "Message 4"
    
    def test_get_user_sessions(self, session_manager):
        """Test retrieving all sessions for a user"""
        # Create multiple sessions for the same user
        session1 = session_manager.create_session(user_id="test_user")
        session2 = session_manager.create_session(user_id="test_user")
        session3 = session_manager.create_session(user_id="other_user")
        
        user_sessions = session_manager.get_user_sessions("test_user")
        
        assert len(user_sessions) == 2
        session_ids = [s.id for s in user_sessions]
        assert session1.id in session_ids
        assert session2.id in session_ids
        assert session3.id not in session_ids
    
    def test_expire_session(self, session_manager):
        """Test expiring a session"""
        session = session_manager.create_session(user_id="test_user")
        
        # Expire the session
        success = session_manager.expire_session(session.id)
        assert success is True
        
        # Try to retrieve expired session
        expired_session = session_manager.get_session(session.id)
        assert expired_session is None
    
    def test_session_expiry_cleanup(self, session_manager):
        """Test automatic cleanup of expired sessions"""
        # Create a session with immediate expiry
        session = session_manager.create_session(user_id="test_user")
        
        # Manually set expiry to past
        from backend.services.session_manager import datetime
        with session_manager._get_connection() as conn:
            cursor = conn.cursor()
            past_time = datetime.utcnow() - timedelta(hours=1)
            cursor.execute(
                "UPDATE sessions SET expires_at = ? WHERE id = ?",
                (past_time, session.id)
            )
        
        # Run cleanup
        expired_count = session_manager.cleanup_expired_sessions()
        assert expired_count == 1
        
        # Verify session is marked inactive
        retrieved = session_manager.get_session(session.id)
        assert retrieved is None
    
    def test_search_messages(self, session_manager):
        """Test searching messages across sessions"""
        session1 = session_manager.create_session(user_id="test_user")
        session2 = session_manager.create_session(user_id="test_user")
        
        # Add messages to sessions
        session_manager.add_message(session1.id, "user", "Find security vulnerabilities")
        session_manager.add_message(session1.id, "assistant", "Found 3 vulnerabilities")
        session_manager.add_message(session2.id, "user", "Check IAM permissions")
        session_manager.add_message(session2.id, "assistant", "IAM analysis complete")
        
        # Search for "vulnerabilities"
        results = session_manager.search_messages("vulnerabilities")
        
        assert len(results) == 2
        assert any("Find security vulnerabilities" in r["content"] for r in results)
        assert any("Found 3 vulnerabilities" in r["content"] for r in results)
    
    def test_session_summary(self, session_manager):
        """Test getting session summary"""
        session = session_manager.create_session(user_id="test_user")
        
        # Add some messages
        session_manager.add_message(session.id, "user", "Hello")
        session_manager.add_message(session.id, "assistant", "Hi!")
        session_manager.add_message(session.id, "user", "Run scan")
        
        summary = session_manager.get_session_summary(session.id)
        
        assert summary["session_id"] == session.id
        assert summary["user_id"] == "test_user"
        assert summary["message_count"] == 3
        assert summary["user_messages"] == 2
        assert summary["assistant_messages"] == 1
        assert summary["is_active"] is True
    
    def test_message_metadata(self, session_manager):
        """Test message metadata storage"""
        session = session_manager.create_session(user_id="test_user")
        
        metadata = {
            "tool_used": "security_scan",
            "execution_time": 2.5,
            "findings_count": 10
        }
        
        message = session_manager.add_message(
            session.id,
            "assistant",
            "Security scan complete",
            metadata=metadata
        )
        
        # Retrieve and verify
        history = session_manager.get_conversation_history(session.id)
        assert len(history) == 1
        assert history[0].metadata["tool_used"] == "security_scan"
        assert history[0].metadata["execution_time"] == 2.5
        assert history[0].metadata["findings_count"] == 10
    
    def test_concurrent_sessions(self, session_manager):
        """Test handling multiple concurrent sessions"""
        sessions = []
        for i in range(5):
            session = session_manager.create_session(user_id=f"user_{i}")
            sessions.append(session)
            
            # Add messages to each session
            for j in range(3):
                session_manager.add_message(
                    session.id,
                    "user" if j % 2 == 0 else "assistant",
                    f"Message {j} for session {i}"
                )
        
        # Verify all sessions and messages
        for i, session in enumerate(sessions):
            history = session_manager.get_conversation_history(session.id)
            assert len(history) == 3
            assert f"session {i}" in history[0].content


class TestSessionAPI:
    """Test Session API endpoints (requires running backend)"""
    
    @pytest.mark.skipif(
        not os.getenv("TEST_API", False),
        reason="API tests require running backend"
    )
    def test_api_create_session(self):
        """Test session creation via API"""
        import httpx
        
        with httpx.Client() as client:
            response = client.post(
                "http://localhost:8000/api/v1/sessions/create",
                json={"user_id": "api_test_user"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "id" in data
            assert data["user_id"] == "api_test_user"
    
    @pytest.mark.skipif(
        not os.getenv("TEST_API", False),
        reason="API tests require running backend"
    )
    def test_api_add_message(self):
        """Test adding message via API"""
        import httpx
        
        with httpx.Client() as client:
            # Create session first
            create_response = client.post(
                "http://localhost:8000/api/v1/sessions/create",
                json={"user_id": "api_test_user"}
            )
            session_id = create_response.json()["id"]
            
            # Add message
            message_response = client.post(
                f"http://localhost:8000/api/v1/sessions/{session_id}/messages",
                json={
                    "role": "user",
                    "content": "Test message via API"
                }
            )
            
            assert message_response.status_code == 200
            data = message_response.json()
            assert data["content"] == "Test message via API"
            assert data["role"] == "user"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])