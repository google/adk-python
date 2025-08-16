#!/usr/bin/env python3
"""
Test suite for critical API endpoints
Tests the main chat, session, and routing functionality
"""

import pytest
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'))

# Import API components
from backend.api.agent_llm import (
    ChatRequest, ChatResponse, 
    chat_with_llm_agent, process_with_llm_agent,
    generate_response_with_real_data
)
from backend.api.sessions import (
    SessionCreateRequest, SessionResponse,
    create_session, get_session, get_session_messages
)

@pytest.fixture
def mock_chat_manager():
    """Mock chat manager for testing."""
    manager = Mock()
    manager.create_session_async = AsyncMock(return_value="test_session_123")
    manager.create_session = Mock(return_value="test_session_123")
    manager.get_session = Mock(return_value=Mock(conversations={"main": []}))
    manager.add_message = AsyncMock()
    manager.get_conversation_history = Mock(return_value=[])
    manager.get_session_analytics = Mock(return_value={"status": "active", "message_count": 0})
    manager.get_contextual_suggestions = Mock(return_value=["Test suggestion 1", "Test suggestion 2"])
    return manager

@pytest.fixture
def chat_request():
    """Sample chat request."""
    return ChatRequest(
        query="Show me the storage buckets in my project",
        user_id="test_user",
        project_id="test-project",
        session_id="test_session_123"
    )

@pytest.fixture
def session_request():
    """Sample session creation request."""
    return SessionCreateRequest(
        user_id="test_user",
        project_id="test-project",
        metadata={"source": "test"}
    )

class TestChatEndpoint:
    """Test the main chat endpoint."""
    
    @pytest.mark.asyncio
    async def test_chat_creates_session(self, mock_chat_manager, chat_request):
        """Test that chat creates a new session when none provided."""
        with patch('backend.api.agent_llm.chat_manager', mock_chat_manager):
            with patch('backend.api.agent_llm.CHAT_MANAGER_AVAILABLE', True):
                with patch('backend.api.agent_llm.process_with_llm_agent', 
                          return_value=("Test response", "TestAgent")):
                    
                    response = await chat_with_llm_agent(chat_request)
                    
                    assert response.success == True
                    assert response.session_id == "test_session_123"
                    assert response.response == "Test response"
                    assert response.agent_used == "TestAgent"
                    mock_chat_manager.create_session_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_uses_existing_session(self, mock_chat_manager, chat_request):
        """Test that chat uses existing session when provided."""
        chat_request.session_id = "existing_session_456"
        
        with patch('backend.api.agent_llm.chat_manager', mock_chat_manager):
            with patch('backend.api.agent_llm.CHAT_MANAGER_AVAILABLE', True):
                with patch('backend.api.agent_llm.process_with_llm_agent',
                          return_value=("Test response", "TestAgent")):
                    
                    response = await chat_with_llm_agent(chat_request)
                    
                    assert response.session_id == "existing_session_456"
                    mock_chat_manager.create_session.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_chat_routing_storage(self, mock_chat_manager):
        """Test routing to storage agent based on keywords."""
        request = ChatRequest(
            query="Show me the buckets in my project",
            user_id="test_user",
            project_id="test-project"
        )
        
        with patch('backend.api.agent_llm.AGENTS_AVAILABLE', False):
            with patch('backend.api.agent_llm.generate_response_with_real_data',
                      return_value="Storage response"):
                
                response, agent = await process_with_llm_agent(
                    request.query, 
                    request.project_id,
                    None,
                    "test_req"
                )
                
                assert agent == "StorageSecurityAgent"
                assert "Storage" in response or response == "Storage response"
    
    @pytest.mark.asyncio
    async def test_chat_routing_iam(self, mock_chat_manager):
        """Test routing to IAM agent based on keywords."""
        request = ChatRequest(
            query="Show me users with high permissions",
            user_id="test_user",
            project_id="test-project"
        )
        
        with patch('backend.api.agent_llm.AGENTS_AVAILABLE', False):
            with patch('backend.api.agent_llm.generate_response_with_real_data',
                      return_value="IAM response"):
                
                response, agent = await process_with_llm_agent(
                    request.query,
                    request.project_id,
                    None,
                    "test_req"
                )
                
                assert agent == "IAMSecurityAgent"

class TestSessionEndpoints:
    """Test session management endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_session(self, mock_chat_manager, session_request):
        """Test session creation endpoint."""
        with patch('backend.api.sessions.chat_manager', mock_chat_manager):
            with patch('backend.api.sessions.CHAT_MANAGER_AVAILABLE', True):
                
                response = await create_session(session_request)
                
                assert response.success == True
                assert response.session_id == "test_session_123"
                assert response.user_id == "test_user"
                mock_chat_manager.create_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_session(self, mock_chat_manager):
        """Test getting session details."""
        mock_session = Mock()
        mock_session.user_id = "test_user"
        mock_session.conversations = {"main": []}
        mock_session.created_at = Mock(isoformat=Mock(return_value="2025-08-01T00:00:00"))
        mock_chat_manager.get_session.return_value = mock_session
        
        with patch('backend.api.sessions.chat_manager', mock_chat_manager):
            with patch('backend.api.sessions.CHAT_MANAGER_AVAILABLE', True):
                
                response = await get_session("test_session_123")
                
                assert response.success == True
                assert response.session_id == "test_session_123"
                assert response.user_id == "test_user"
    
    @pytest.mark.asyncio
    async def test_get_session_messages(self, mock_chat_manager):
        """Test getting messages from a session."""
        mock_message = Mock()
        mock_message.sender_type = "user"
        mock_message.content = "Test message"
        mock_message.timestamp = Mock(isoformat=Mock(return_value="2025-08-01T00:00:00"))
        mock_message.agent_used = None
        
        mock_chat_manager.get_conversation_history.return_value = [mock_message]
        
        with patch('backend.api.sessions.chat_manager', mock_chat_manager):
            with patch('backend.api.sessions.CHAT_MANAGER_AVAILABLE', True):
                
                response = await get_session_messages("test_session_123")
                
                assert response.success == True
                assert len(response.messages) == 1
                assert response.messages[0]["content"] == "Test message"

class TestRoutingLogic:
    """Test agent routing decision logic."""
    
    @pytest.mark.asyncio
    async def test_storage_keywords(self):
        """Test storage keyword detection."""
        queries = [
            "show me my buckets",
            "check storage security",
            "analyze backup policies",
            "review archive settings"
        ]
        
        for query in queries:
            with patch('backend.api.agent_llm.AGENTS_AVAILABLE', False):
                with patch('backend.api.agent_llm.generate_response_with_real_data',
                          return_value="Response"):
                    
                    _, agent = await process_with_llm_agent(
                        query, "test-project", None, "test_req"
                    )
                    
                    assert agent == "StorageSecurityAgent", f"Failed for query: {query}"
    
    @pytest.mark.asyncio
    async def test_network_keywords(self):
        """Test network keyword detection."""
        queries = [
            "check firewall rules",
            "analyze network security",
            "review vpc configuration",
            "check open ports"
        ]
        
        for query in queries:
            with patch('backend.api.agent_llm.AGENTS_AVAILABLE', False):
                with patch('backend.api.agent_llm.generate_response_with_real_data',
                          return_value="Response"):
                    
                    _, agent = await process_with_llm_agent(
                        query, "test-project", None, "test_req"
                    )
                    
                    assert agent == "NetworkSecurityAgent", f"Failed for query: {query}"
    
    @pytest.mark.asyncio
    async def test_cost_keywords(self):
        """Test cost keyword detection."""
        queries = [
            "analyze my costs",
            "show spending report",
            "check budget status",
            "find cost savings"
        ]
        
        for query in queries:
            with patch('backend.api.agent_llm.AGENTS_AVAILABLE', False):
                with patch('backend.api.agent_llm.generate_response_with_real_data',
                          return_value="Response"):
                    
                    _, agent = await process_with_llm_agent(
                        query, "test-project", None, "test_req"
                    )
                    
                    assert agent == "CostOptimizationAgent", f"Failed for query: {query}"

class TestErrorHandling:
    """Test error handling in API endpoints."""
    
    @pytest.mark.asyncio
    async def test_chat_without_manager(self, chat_request):
        """Test chat fails gracefully without chat manager."""
        with patch('backend.api.agent_llm.CHAT_MANAGER_AVAILABLE', False):
            with pytest.raises(Exception):  # HTTPException in real scenario
                await chat_with_llm_agent(chat_request)
    
    @pytest.mark.asyncio
    async def test_session_not_found(self, mock_chat_manager):
        """Test handling of non-existent session."""
        mock_chat_manager.get_session.return_value = None
        
        with patch('backend.api.sessions.chat_manager', mock_chat_manager):
            with patch('backend.api.sessions.CHAT_MANAGER_AVAILABLE', True):
                with pytest.raises(Exception):  # HTTPException in real scenario
                    await get_session("nonexistent_session")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])