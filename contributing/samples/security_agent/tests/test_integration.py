#!/usr/bin/env python3
"""
Integration tests for backend-to-agent communication and session persistence
"""

import pytest
import asyncio
import json
import time
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'))

from backend.chat_manager import chat_manager, ChatMessage, MessageType

class TestSessionPersistence:
    """Test session persistence across requests."""
    
    def test_session_creates_and_persists(self):
        """Test that sessions persist across multiple interactions."""
        manager = chat_manager
        
        # Create session
        session_id = manager.create_session("test_user", {"project": "test"})
        assert session_id is not None
        
        # Add first message
        asyncio.run(manager.add_message(
            session_id, "What are my storage buckets?", "user"
        ))
        
        # Add response
        asyncio.run(manager.add_message(
            session_id, "You have 5 storage buckets...", "assistant",
            agent_used="StorageAgent"
        ))
        
        # Verify persistence
        history = manager.get_conversation_history(session_id)
        assert len(history) == 2
        assert history[0].content == "What are my storage buckets?"
        assert history[1].agent_used == "StorageAgent"
        
        # Add follow-up
        asyncio.run(manager.add_message(
            session_id, "Which ones are public?", "user"
        ))
        
        # Verify continuity
        history = manager.get_conversation_history(session_id)
        assert len(history) == 3
        assert history[2].content == "Which ones are public?"
    
    def test_multiple_users_isolated(self):
        """Test that different users have isolated sessions."""
        manager = chat_manager
        
        # Create sessions for different users
        session1 = manager.create_session("user1", {})
        session2 = manager.create_session("user2", {})
        
        # Add messages to each session
        asyncio.run(manager.add_message(session1, "User 1 message", "user"))
        asyncio.run(manager.add_message(session2, "User 2 message", "user"))
        
        # Verify isolation
        history1 = manager.get_conversation_history(session1)
        history2 = manager.get_conversation_history(session2)
        
        assert len(history1) == 1
        assert len(history2) == 1
        assert history1[0].content == "User 1 message"
        assert history2[0].content == "User 2 message"
    
    def test_session_analytics(self):
        """Test session analytics tracking."""
        manager = chat_manager
        
        session_id = manager.create_session("test_user", {})
        
        # Add multiple messages
        for i in range(5):
            asyncio.run(manager.add_message(
                session_id, f"Message {i}", "user"
            ))
            asyncio.run(manager.add_message(
                session_id, f"Response {i}", "assistant",
                agent_used="TestAgent"
            ))
        
        # Check analytics
        analytics = manager.get_session_analytics(session_id)
        assert analytics["message_count"] == 10
        assert analytics["status"] == "active"
        assert "TestAgent" in analytics.get("agents_used", [])

class TestAgentDelegation:
    """Test agent delegation and routing."""
    
    @pytest.mark.asyncio
    async def test_correct_agent_selection(self):
        """Test that queries route to correct agents."""
        from backend.api.agent_llm import process_with_llm_agent
        
        test_cases = [
            ("show my buckets", "StorageSecurityAgent"),
            ("check IAM users", "IAMSecurityAgent"),
            ("analyze firewall rules", "NetworkSecurityAgent"),
            ("check compliance status", "ComplianceAgent"),
            ("analyze costs", "CostOptimizationAgent"),
            ("general security question", "CoordinatorAgent")
        ]
        
        for query, expected_agent in test_cases:
            with patch('backend.api.agent_llm.AGENTS_AVAILABLE', False):
                with patch('backend.api.agent_llm.generate_response_with_real_data',
                          return_value="Mock response"):
                    
                    _, agent = await process_with_llm_agent(
                        query, "test-project", None, "test_req"
                    )
                    
                    assert agent == expected_agent, f"Query '{query}' routed to {agent}, expected {expected_agent}"
    
    @pytest.mark.asyncio
    async def test_delegation_path_tracking(self):
        """Test that delegation path is tracked correctly."""
        from backend.api.agent_llm import ChatRequest, chat_with_llm_agent
        
        request = ChatRequest(
            query="analyze my storage security",
            user_id="test_user",
            project_id="test-project"
        )
        
        mock_manager = Mock()
        mock_manager.create_session = Mock(return_value="test_session")
        mock_manager.get_session = Mock(return_value=Mock(conversations={"main": []}))
        mock_manager.add_message = AsyncMock()
        mock_manager.get_contextual_suggestions = Mock(return_value=[])
        
        with patch('backend.api.agent_llm.chat_manager', mock_manager):
            with patch('backend.api.agent_llm.CHAT_MANAGER_AVAILABLE', True):
                with patch('backend.api.agent_llm.process_with_llm_agent',
                          return_value=("Response", "StorageSecurityAgent")):
                    
                    response = await chat_with_llm_agent(request)
                    
                    assert response.delegation_path == ["SecurityAgent", "StorageSecurityAgent"]

class TestErrorRecovery:
    """Test error recovery and fallback mechanisms."""
    
    @pytest.mark.asyncio
    async def test_fallback_when_agent_unavailable(self):
        """Test fallback to mock data when agents unavailable."""
        from backend.api.agent_llm import process_with_llm_agent
        
        with patch('backend.api.agent_llm.AGENTS_AVAILABLE', False):
            response, agent = await process_with_llm_agent(
                "check my storage", "test-project", None, "test_req"
            )
            
            assert response is not None
            assert agent == "StorageSecurityAgent"
            assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_session_recovery(self):
        """Test session can be recovered after error."""
        manager = chat_manager
        
        session_id = manager.create_session("test_user", {})
        
        # Add message
        await manager.add_message(session_id, "First message", "user")
        
        # Simulate error (would normally raise)
        with patch.object(manager, 'add_message', side_effect=Exception("Test error")):
            try:
                await manager.add_message(session_id, "Error message", "user")
            except:
                pass
        
        # Verify session still accessible
        history = manager.get_conversation_history(session_id)
        assert len(history) == 1
        assert history[0].content == "First message"
        
        # Verify can still add messages
        await manager.add_message(session_id, "Recovery message", "user")
        history = manager.get_conversation_history(session_id)
        assert len(history) == 2

class TestPerformanceMetrics:
    """Test performance tracking and metrics."""
    
    @pytest.mark.asyncio
    async def test_response_time_tracking(self):
        """Test that response times are tracked."""
        from backend.api.agent_llm import ChatRequest, chat_with_llm_agent
        
        request = ChatRequest(
            query="test query",
            user_id="test_user",
            project_id="test-project"
        )
        
        mock_manager = Mock()
        mock_manager.create_session = Mock(return_value="test_session")
        mock_manager.get_session = Mock(return_value=Mock(conversations={"main": []}))
        mock_manager.add_message = AsyncMock()
        mock_manager.get_contextual_suggestions = Mock(return_value=[])
        
        with patch('backend.api.agent_llm.chat_manager', mock_manager):
            with patch('backend.api.agent_llm.CHAT_MANAGER_AVAILABLE', True):
                with patch('backend.api.agent_llm.process_with_llm_agent',
                          return_value=("Response", "TestAgent")):
                    
                    response = await chat_with_llm_agent(request)
                    
                    assert response.performance_metrics is not None
                    assert "response_time_ms" in response.performance_metrics
                    assert response.performance_metrics["response_time_ms"] > 0
    
    def test_concurrent_session_handling(self):
        """Test handling multiple concurrent sessions."""
        manager = chat_manager
        
        # Create multiple sessions
        sessions = []
        for i in range(10):
            session_id = manager.create_session(f"user_{i}", {})
            sessions.append(session_id)
        
        # Add messages concurrently
        async def add_messages():
            tasks = []
            for session_id in sessions:
                tasks.append(manager.add_message(
                    session_id, f"Message for {session_id}", "user"
                ))
            await asyncio.gather(*tasks)
        
        asyncio.run(add_messages())
        
        # Verify all messages were added
        for session_id in sessions:
            history = manager.get_conversation_history(session_id)
            assert len(history) == 1
            assert session_id in history[0].content

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])