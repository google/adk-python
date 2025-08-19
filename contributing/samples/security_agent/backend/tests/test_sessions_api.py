"""
Comprehensive test suite for Sessions API endpoints - TASK-003.

Tests session management, conversation history, persistence, CRUD operations,
and session lifecycle management with SQLite backend.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json
import sqlite3

# Import the sessions module and related components
from backend.api.sessions import router
from backend.main import app

client = TestClient(app)

# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def temp_db():
    """Create temporary SQLite database for testing."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    # Initialize database schema
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    
    # Create sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            title TEXT,
            context TEXT,
            metadata TEXT
        )
    ''')
    
    # Create messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()
    
    yield path
    
    # Cleanup
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass

@pytest.fixture
def mock_session_manager(temp_db):
    """Mock session manager with temporary database."""
    with patch('backend.api.sessions.SessionManager') as mock_manager:
        # Create real instance with temp db
        from backend.services.session_manager import SessionManager
        real_manager = SessionManager(db_path=temp_db)
        mock_manager.return_value = real_manager
        yield real_manager

@pytest.fixture
def sample_session_data():
    """Sample session data for testing."""
    return {
        "session_id": "test-session-123",
        "user_id": "test-user@example.com",
        "title": "Test Security Analysis Session",
        "context": "Discussion about GCP security best practices",
        "metadata": {"tags": ["security", "gcp"], "priority": "high"}
    }

@pytest.fixture
def sample_message_data():
    """Sample message data for testing."""
    return {
        "role": "user",
        "content": "What are the security vulnerabilities in my GCP project?",
        "metadata": {"timestamp": datetime.now().isoformat(), "source": "web"}
    }

# ============================================================================
# SESSION CRUD OPERATIONS TESTS
# ============================================================================

class TestSessionCRUDOperations:
    """Test session CRUD operations."""

    def test_create_session_success(self, mock_session_manager, sample_session_data):
        """Test successful session creation."""
        response = client.post("/api/v1/sessions", json=sample_session_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["session_id"] == sample_session_data["session_id"]
        assert "created_at" in data

    def test_create_session_minimal_data(self, mock_session_manager):
        """Test session creation with minimal required data."""
        minimal_data = {
            "session_id": "minimal-session",
            "user_id": "minimal-user@example.com"
        }
        
        response = client.post("/api/v1/sessions", json=minimal_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["session_id"] == minimal_data["session_id"]

    def test_create_session_duplicate_id(self, mock_session_manager, sample_session_data):
        """Test creating session with duplicate ID."""
        # Create first session
        client.post("/api/v1/sessions", json=sample_session_data)
        
        # Try to create duplicate
        response = client.post("/api/v1/sessions", json=sample_session_data)
        
        assert response.status_code == 409  # Conflict
        data = response.json()
        assert "already exists" in data["detail"]

    def test_get_session_success(self, mock_session_manager, sample_session_data):
        """Test successful session retrieval."""
        # Create session first
        client.post("/api/v1/sessions", json=sample_session_data)
        
        # Retrieve session
        response = client.get(f"/api/v1/sessions/{sample_session_data['session_id']}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["session"]["session_id"] == sample_session_data["session_id"]
        assert data["session"]["user_id"] == sample_session_data["user_id"]

    def test_get_session_not_found(self, mock_session_manager):
        """Test retrieving non-existent session."""
        response = client.get("/api/v1/sessions/nonexistent-session")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_update_session_success(self, mock_session_manager, sample_session_data):
        """Test successful session update."""
        # Create session first
        client.post("/api/v1/sessions", json=sample_session_data)
        
        # Update session
        update_data = {
            "title": "Updated Security Analysis Session",
            "context": "Updated discussion about advanced GCP security",
            "metadata": {"tags": ["security", "gcp", "advanced"], "priority": "critical"}
        }
        
        response = client.put(f"/api/v1/sessions/{sample_session_data['session_id']}", json=update_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Updated Security Analysis" in data["title"]

    def test_delete_session_success(self, mock_session_manager, sample_session_data):
        """Test successful session deletion."""
        # Create session first
        client.post("/api/v1/sessions", json=sample_session_data)
        
        # Delete session
        response = client.delete(f"/api/v1/sessions/{sample_session_data['session_id']}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Verify deletion
        get_response = client.get(f"/api/v1/sessions/{sample_session_data['session_id']}")
        assert get_response.status_code == 404

    def test_list_sessions_by_user(self, mock_session_manager, sample_session_data):
        """Test listing sessions by user."""
        # Create multiple sessions for the same user
        for i in range(3):
            session_data = sample_session_data.copy()
            session_data["session_id"] = f"session-{i}"
            session_data["title"] = f"Session {i}"
            client.post("/api/v1/sessions", json=session_data)
        
        # List sessions
        response = client.get(f"/api/v1/sessions/user/{sample_session_data['user_id']}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["sessions"]) == 3
        assert data["total_count"] == 3

    def test_list_sessions_with_pagination(self, mock_session_manager, sample_session_data):
        """Test session listing with pagination."""
        # Create multiple sessions
        for i in range(5):
            session_data = sample_session_data.copy()
            session_data["session_id"] = f"session-{i}"
            client.post("/api/v1/sessions", json=session_data)
        
        # Test pagination
        response = client.get(f"/api/v1/sessions/user/{sample_session_data['user_id']}?page=1&page_size=2")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["sessions"]) == 2
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert data["total_count"] == 5

# ============================================================================
# MESSAGE MANAGEMENT TESTS
# ============================================================================

class TestMessageManagement:
    """Test message management within sessions."""

    def test_add_message_success(self, mock_session_manager, sample_session_data, sample_message_data):
        """Test successful message addition."""
        # Create session first
        client.post("/api/v1/sessions", json=sample_session_data)
        
        # Add message
        response = client.post(
            f"/api/v1/sessions/{sample_session_data['session_id']}/messages",
            json=sample_message_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "message_id" in data

    def test_add_multiple_messages(self, mock_session_manager, sample_session_data):
        """Test adding multiple messages to a session."""
        # Create session first
        client.post("/api/v1/sessions", json=sample_session_data)
        
        # Add multiple messages
        messages = [
            {"role": "user", "content": "What are my GCP security risks?"},
            {"role": "assistant", "content": "I'll analyze your GCP security posture..."},
            {"role": "user", "content": "Focus on IAM permissions"},
            {"role": "assistant", "content": "Here are the IAM security findings..."}
        ]
        
        message_ids = []
        for message in messages:
            response = client.post(
                f"/api/v1/sessions/{sample_session_data['session_id']}/messages",
                json=message
            )
            assert response.status_code == 200
            message_ids.append(response.json()["message_id"])
        
        assert len(message_ids) == 4

    def test_get_session_messages(self, mock_session_manager, sample_session_data):
        """Test retrieving messages from a session."""
        # Create session and add messages
        client.post("/api/v1/sessions", json=sample_session_data)
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "Check my security"}
        ]
        
        for message in messages:
            client.post(
                f"/api/v1/sessions/{sample_session_data['session_id']}/messages",
                json=message
            )
        
        # Retrieve messages
        response = client.get(f"/api/v1/sessions/{sample_session_data['session_id']}/messages")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["messages"]) == 3
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hello"

    def test_get_messages_with_pagination(self, mock_session_manager, sample_session_data):
        """Test message retrieval with pagination."""
        # Create session and add many messages
        client.post("/api/v1/sessions", json=sample_session_data)
        
        for i in range(10):
            message = {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            client.post(
                f"/api/v1/sessions/{sample_session_data['session_id']}/messages",
                json=message
            )
        
        # Test pagination
        response = client.get(
            f"/api/v1/sessions/{sample_session_data['session_id']}/messages?page=1&page_size=3"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["messages"]) == 3
        assert data["total_messages"] == 10

    def test_search_messages(self, mock_session_manager, sample_session_data):
        """Test searching messages within sessions."""
        # Create session and add messages
        client.post("/api/v1/sessions", json=sample_session_data)
        
        messages = [
            {"role": "user", "content": "What are my security vulnerabilities?"},
            {"role": "assistant", "content": "I found several security issues in your GCP project"},
            {"role": "user", "content": "Show me the IAM analysis"},
            {"role": "assistant", "content": "Here are the IAM security findings"}
        ]
        
        for message in messages:
            client.post(
                f"/api/v1/sessions/{sample_session_data['session_id']}/messages",
                json=message
            )
        
        # Search for security-related messages
        response = client.get(
            f"/api/v1/sessions/{sample_session_data['session_id']}/messages/search?query=security"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["messages"]) >= 2  # Should find security-related messages

# ============================================================================
# SESSION CONTEXT MANAGEMENT TESTS
# ============================================================================

class TestSessionContextManagement:
    """Test session context management."""

    def test_update_session_context(self, mock_session_manager, sample_session_data):
        """Test updating session context."""
        # Create session
        client.post("/api/v1/sessions", json=sample_session_data)
        
        # Update context
        context_update = {
            "context": "Extended discussion about GCP security best practices and compliance",
            "metadata": {
                "topics": ["security", "compliance", "IAM", "storage"],
                "priority": "high",
                "last_activity": datetime.now().isoformat()
            }
        }
        
        response = client.put(
            f"/api/v1/sessions/{sample_session_data['session_id']}/context",
            json=context_update
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_get_session_context(self, mock_session_manager, sample_session_data):
        """Test retrieving session context."""
        # Create session
        client.post("/api/v1/sessions", json=sample_session_data)
        
        # Get context
        response = client.get(f"/api/v1/sessions/{sample_session_data['session_id']}/context")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "context" in data
        assert "metadata" in data

    def test_session_summary_generation(self, mock_session_manager, sample_session_data):
        """Test automatic session summary generation."""
        # Create session and add messages
        client.post("/api/v1/sessions", json=sample_session_data)
        
        messages = [
            {"role": "user", "content": "Analyze my GCP security"},
            {"role": "assistant", "content": "I found 5 critical security issues"},
            {"role": "user", "content": "What about IAM permissions?"},
            {"role": "assistant", "content": "3 overprivileged accounts detected"}
        ]
        
        for message in messages:
            client.post(
                f"/api/v1/sessions/{sample_session_data['session_id']}/messages",
                json=message
            )
        
        # Generate summary
        response = client.post(f"/api/v1/sessions/{sample_session_data['session_id']}/summary")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "summary" in data
        assert "key_topics" in data
        assert "action_items" in data

# ============================================================================
# SESSION ANALYTICS TESTS
# ============================================================================

class TestSessionAnalytics:
    """Test session analytics and statistics."""

    def test_session_statistics(self, mock_session_manager, sample_session_data):
        """Test session statistics calculation."""
        # Create session and add messages
        client.post("/api/v1/sessions", json=sample_session_data)
        
        for i in range(10):
            message = {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            client.post(
                f"/api/v1/sessions/{sample_session_data['session_id']}/messages",
                json=message
            )
        
        # Get statistics
        response = client.get(f"/api/v1/sessions/{sample_session_data['session_id']}/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "message_count" in data
        assert "user_messages" in data
        assert "assistant_messages" in data
        assert data["message_count"] == 10

    def test_user_session_analytics(self, mock_session_manager, sample_session_data):
        """Test user-level session analytics."""
        # Create multiple sessions for user
        for i in range(3):
            session_data = sample_session_data.copy()
            session_data["session_id"] = f"session-{i}"
            client.post("/api/v1/sessions", json=session_data)
        
        # Get user analytics
        response = client.get(f"/api/v1/sessions/user/{sample_session_data['user_id']}/analytics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "total_sessions" in data
        assert "total_messages" in data
        assert "average_session_length" in data
        assert data["total_sessions"] == 3

# ============================================================================
# SESSION LIFECYCLE TESTS
# ============================================================================

class TestSessionLifecycle:
    """Test session lifecycle management."""

    def test_session_expiry_check(self, mock_session_manager, sample_session_data):
        """Test session expiry checking."""
        # Create session
        client.post("/api/v1/sessions", json=sample_session_data)
        
        # Check expiry
        response = client.get(f"/api/v1/sessions/{sample_session_data['session_id']}/expiry")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "is_expired" in data
        assert "expires_at" in data

    def test_session_cleanup(self, mock_session_manager):
        """Test session cleanup for expired sessions."""
        response = client.post("/api/v1/sessions/cleanup", json={
            "max_age_days": 30,
            "dry_run": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "cleanup_summary" in data

    def test_session_archival(self, mock_session_manager, sample_session_data):
        """Test session archival."""
        # Create session
        client.post("/api/v1/sessions", json=sample_session_data)
        
        # Archive session
        response = client.post(f"/api/v1/sessions/{sample_session_data['session_id']}/archive")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "archived_at" in data

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestSessionErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_session_id_format(self, mock_session_manager):
        """Test handling of invalid session ID format."""
        response = client.get("/api/v1/sessions/invalid@session@id")
        
        assert response.status_code in [400, 404]

    def test_create_session_missing_required_fields(self, mock_session_manager):
        """Test session creation with missing required fields."""
        incomplete_data = {"session_id": "incomplete"}  # Missing user_id
        
        response = client.post("/api/v1/sessions", json=incomplete_data)
        
        assert response.status_code == 422  # Validation error

    def test_add_message_to_nonexistent_session(self, mock_session_manager, sample_message_data):
        """Test adding message to non-existent session."""
        response = client.post("/api/v1/sessions/nonexistent/messages", json=sample_message_data)
        
        assert response.status_code == 404

    def test_database_connection_error(self, mock_session_manager):
        """Test handling of database connection errors."""
        with patch.object(mock_session_manager, 'get_session', side_effect=Exception("Database error")):
            response = client.get("/api/v1/sessions/test-session")
            
            assert response.status_code == 500

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestSessionPerformance:
    """Test session management performance."""

    def test_bulk_session_creation_performance(self, mock_session_manager):
        """Test performance with bulk session creation."""
        import time
        
        start_time = time.time()
        
        # Create many sessions
        for i in range(50):
            session_data = {
                "session_id": f"bulk-session-{i}",
                "user_id": f"user-{i}@example.com",
                "title": f"Bulk Session {i}"
            }
            response = client.post("/api/v1/sessions", json=session_data)
            assert response.status_code == 200
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 10.0  # 10 seconds for 50 sessions

    def test_large_session_message_retrieval(self, mock_session_manager, sample_session_data):
        """Test performance with large number of messages."""
        # Create session
        client.post("/api/v1/sessions", json=sample_session_data)
        
        # Add many messages
        for i in range(100):
            message = {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            client.post(
                f"/api/v1/sessions/{sample_session_data['session_id']}/messages",
                json=message
            )
        
        import time
        start_time = time.time()
        
        # Retrieve messages
        response = client.get(f"/api/v1/sessions/{sample_session_data['session_id']}/messages")
        
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 2.0  # Should be fast

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSessionIntegration:
    """Test session integration with other components."""

    def test_session_with_security_analysis_context(self, mock_session_manager):
        """Test session integration with security analysis."""
        # Create session with security context
        session_data = {
            "session_id": "security-session",
            "user_id": "security-analyst@example.com",
            "title": "GCP Security Analysis Session",
            "context": "Comprehensive security assessment for production environment",
            "metadata": {
                "type": "security_analysis",
                "project_id": "prod-project",
                "compliance_frameworks": ["SOC2", "HIPAA"]
            }
        }
        
        response = client.post("/api/v1/sessions", json=session_data)
        assert response.status_code == 200
        
        # Add security-related messages
        security_messages = [
            {"role": "user", "content": "Start comprehensive security scan"},
            {"role": "assistant", "content": "Initiating GCP security assessment..."},
            {"role": "system", "content": "Found 12 critical vulnerabilities", 
             "metadata": {"scan_results": {"critical": 12, "high": 8, "medium": 15}}}
        ]
        
        for message in security_messages:
            response = client.post("/api/v1/sessions/security-session/messages", json=message)
            assert response.status_code == 200

    def test_session_export_import(self, mock_session_manager, sample_session_data):
        """Test session export and import functionality."""
        # Create session with messages
        client.post("/api/v1/sessions", json=sample_session_data)
        
        messages = [
            {"role": "user", "content": "Export test message 1"},
            {"role": "assistant", "content": "Export test response 1"}
        ]
        
        for message in messages:
            client.post(
                f"/api/v1/sessions/{sample_session_data['session_id']}/messages",
                json=message
            )
        
        # Export session
        response = client.get(f"/api/v1/sessions/{sample_session_data['session_id']}/export")
        
        assert response.status_code == 200
        export_data = response.json()
        assert export_data["success"] is True
        assert "session_data" in export_data
        assert "messages" in export_data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])