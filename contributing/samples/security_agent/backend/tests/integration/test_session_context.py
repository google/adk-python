"""
Integration tests for session persistence and context management
Tests that session state is properly maintained across requests.
These tests should FAIL initially as part of TDD approach.
"""

import pytest
import json
import tempfile
import os
from typing import Dict, Any, Optional
import uuid


class TestSessionContext:
    """Test session persistence and context management"""

    @pytest.fixture
    def session_manager(self):
        """Initialize session manager - this doesn't exist yet"""
        # This will fail initially - the actual session manager doesn't exist
        from backend.services.session_manager import SessionManager
        return SessionManager()

    @pytest.fixture
    def adk_agent_with_session(self):
        """Initialize ADK agent with session support"""
        # This will fail initially - proper session integration doesn't exist
        from agents.adk_agent import ADKAgent

        agent = ADKAgent()
        agent.enable_session_persistence()
        return agent

    @pytest.fixture
    def temp_session_storage(self):
        """Create temporary session storage"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_session_creation(self, session_manager):
        """Test creating new session"""
        session_id = session_manager.create_session()

        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 0

        # Should be a valid UUID format
        try:
            uuid.UUID(session_id)
        except ValueError:
            pytest.fail("Session ID should be a valid UUID")

    def test_session_exists(self, session_manager):
        """Test checking if session exists"""
        # Create a session
        session_id = session_manager.create_session()

        # Should exist
        assert session_manager.session_exists(session_id)

        # Non-existent session should return False
        fake_session_id = str(uuid.uuid4())
        assert not session_manager.session_exists(fake_session_id)

    def test_session_context_storage(self, session_manager):
        """Test storing and retrieving session context"""
        session_id = session_manager.create_session()

        # Store context
        context_data = {
            "user_preferences": {"theme": "dark", "notifications": True},
            "conversation_history": ["Hello", "How can I help?"],
            "current_project": "project-123"
        }

        session_manager.store_context(session_id, context_data)

        # Retrieve context
        retrieved_context = session_manager.get_context(session_id)

        assert retrieved_context is not None
        assert retrieved_context["user_preferences"]["theme"] == "dark"
        assert retrieved_context["current_project"] == "project-123"
        assert len(retrieved_context["conversation_history"]) == 2

    def test_session_context_update(self, session_manager):
        """Test updating session context"""
        session_id = session_manager.create_session()

        # Initial context
        initial_context = {"messages": ["Hello"]}
        session_manager.store_context(session_id, initial_context)

        # Update context
        updated_context = {"messages": ["Hello", "How are you?"]}
        session_manager.update_context(session_id, updated_context)

        # Retrieve updated context
        context = session_manager.get_context(session_id)
        assert len(context["messages"]) == 2

    def test_session_context_merge(self, session_manager):
        """Test merging context data"""
        session_id = session_manager.create_session()

        # Set initial context
        session_manager.store_context(session_id, {
            "user_info": {"name": "John"},
            "preferences": {"lang": "en"}
        })

        # Merge additional context
        session_manager.merge_context(session_id, {
            "user_info": {"email": "john@example.com"},
            "current_task": "security_audit"
        })

        # Should merge without overwriting existing keys
        context = session_manager.get_context(session_id)
        assert context["user_info"]["name"] == "John"
        assert context["user_info"]["email"] == "john@example.com"
        assert context["preferences"]["lang"] == "en"
        assert context["current_task"] == "security_audit"

    def test_session_expiration(self, session_manager):
        """Test session expiration"""
        # Create session with short expiration
        session_id = session_manager.create_session(expires_in_minutes=1)

        # Should exist initially
        assert session_manager.session_exists(session_id)

        # Simulate time passing (this would need time manipulation in real implementation)
        session_manager.simulate_time_passage(minutes=2)

        # Should be expired
        assert session_manager.is_session_expired(session_id)
        assert not session_manager.session_exists(session_id)

    def test_session_cleanup(self, session_manager):
        """Test session cleanup functionality"""
        # Create multiple sessions
        session_ids = []
        for i in range(5):
            session_id = session_manager.create_session()
            session_ids.append(session_id)

        # All should exist
        for sid in session_ids:
            assert session_manager.session_exists(sid)

        # Clean up expired sessions
        session_manager.cleanup_expired_sessions()

        # Active sessions should still exist
        for sid in session_ids:
            assert session_manager.session_exists(sid)

    def test_session_persistence_across_restarts(self, session_manager, temp_session_storage):
        """Test that sessions persist across service restarts"""
        # Configure session manager to use temp storage
        session_manager.configure_storage(temp_session_storage)

        # Create session and store context
        session_id = session_manager.create_session()
        context_data = {"persistent_data": "should_survive_restart"}
        session_manager.store_context(session_id, context_data)

        # Simulate service restart by creating new session manager
        from backend.services.session_manager import SessionManager
        new_session_manager = SessionManager()
        new_session_manager.configure_storage(temp_session_storage)

        # Session should still exist and context should be retrievable
        assert new_session_manager.session_exists(session_id)
        retrieved_context = new_session_manager.get_context(session_id)
        assert retrieved_context["persistent_data"] == "should_survive_restart"

    @pytest.mark.asyncio
    async def test_agent_session_integration(self, adk_agent_with_session):
        """Test ADK agent integration with sessions"""
        session_id = str(uuid.uuid4())

        # First query - establish context
        response1 = await adk_agent_with_session.process_query(
            "My name is Alice and I work on project-alpha",
            session_id=session_id
        )

        assert response1 is not None

        # Second query - should remember context
        response2 = await adk_agent_with_session.process_query(
            "What is my name?",
            session_id=session_id
        )

        assert "Alice" in response2

        # Third query - should remember project
        response3 = await adk_agent_with_session.process_query(
            "What project do I work on?",
            session_id=session_id
        )

        assert "project-alpha" in response3

    @pytest.mark.asyncio
    async def test_multiple_session_isolation(self, adk_agent_with_session):
        """Test that multiple sessions are properly isolated"""
        session_1 = str(uuid.uuid4())
        session_2 = str(uuid.uuid4())

        # Set different context in each session
        await adk_agent_with_session.process_query(
            "My name is Alice",
            session_id=session_1
        )

        await adk_agent_with_session.process_query(
            "My name is Bob",
            session_id=session_2
        )

        # Verify isolation
        response_1 = await adk_agent_with_session.process_query(
            "What is my name?",
            session_id=session_1
        )

        response_2 = await adk_agent_with_session.process_query(
            "What is my name?",
            session_id=session_2
        )

        assert "Alice" in response_1
        assert "Bob" in response_2
        assert "Alice" not in response_2
        assert "Bob" not in response_1

    def test_session_metadata(self, session_manager):
        """Test session metadata tracking"""
        session_id = session_manager.create_session()

        # Should track metadata
        metadata = session_manager.get_session_metadata(session_id)

        assert metadata is not None
        assert "created_at" in metadata
        assert "last_accessed" in metadata
        assert "access_count" in metadata
        assert "user_agent" in metadata

        # Access the session multiple times
        for _ in range(3):
            session_manager.get_context(session_id)

        # Access count should increase
        updated_metadata = session_manager.get_session_metadata(session_id)
        assert updated_metadata["access_count"] > metadata["access_count"]

    def test_session_size_limits(self, session_manager):
        """Test session size limitations"""
        session_id = session_manager.create_session()

        # Store large context data
        large_data = {"large_field": "x" * 10000}  # 10KB of data

        # Should handle reasonably sized data
        session_manager.store_context(session_id, large_data)
        retrieved = session_manager.get_context(session_id)
        assert len(retrieved["large_field"]) == 10000

        # Test very large data (should be rejected or truncated)
        very_large_data = {"huge_field": "x" * 1000000}  # 1MB of data

        with pytest.raises(ValueError) as exc_info:
            session_manager.store_context(session_id, very_large_data)

        assert "too large" in str(exc_info.value).lower()

    def test_concurrent_session_access(self, session_manager):
        """Test concurrent access to the same session"""
        import threading
        import time

        session_id = session_manager.create_session()
        session_manager.store_context(session_id, {"counter": 0})

        def increment_counter():
            for _ in range(10):
                context = session_manager.get_context(session_id)
                context["counter"] += 1
                session_manager.store_context(session_id, context)
                time.sleep(0.01)  # Small delay to increase chance of race conditions

        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=increment_counter)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Final counter should be 30 (3 threads * 10 increments each)
        final_context = session_manager.get_context(session_id)
        assert final_context["counter"] == 30, "Concurrent access should be thread-safe"

    def test_session_backup_and_restore(self, session_manager, temp_session_storage):
        """Test session backup and restore functionality"""
        session_manager.configure_storage(temp_session_storage)

        # Create sessions with context
        session_ids = []
        for i in range(3):
            session_id = session_manager.create_session()
            session_ids.append(session_id)
            session_manager.store_context(session_id, {"user": f"user_{i}"})

        # Create backup
        backup_file = os.path.join(temp_session_storage, "session_backup.json")
        session_manager.backup_sessions(backup_file)

        assert os.path.exists(backup_file)

        # Clear all sessions
        session_manager.clear_all_sessions()

        # Verify sessions are gone
        for session_id in session_ids:
            assert not session_manager.session_exists(session_id)

        # Restore from backup
        session_manager.restore_sessions(backup_file)

        # Verify sessions are restored
        for i, session_id in enumerate(session_ids):
            assert session_manager.session_exists(session_id)
            context = session_manager.get_context(session_id)
            assert context["user"] == f"user_{i}"

    def test_session_analytics(self, session_manager):
        """Test session analytics and metrics"""
        # Create multiple sessions with different patterns
        session_ids = []
        for i in range(5):
            session_id = session_manager.create_session()
            session_ids.append(session_id)

            # Simulate different usage patterns
            for j in range(i + 1):
                session_manager.get_context(session_id)

        # Get analytics
        analytics = session_manager.get_analytics()

        assert analytics["total_sessions"] == 5
        assert analytics["active_sessions"] == 5
        assert analytics["total_access_count"] == 15  # 1+2+3+4+5
        assert "average_session_size" in analytics
        assert "most_active_session" in analytics