# Phase 2: Conversational ADK Testing Strategy

## Overview

This document outlines the comprehensive testing strategy for Phase 2: Conversational ADK Orchestration features. The testing approach ensures reliability, performance, and maintainability of conversation-aware ADK agents while maintaining backward compatibility.

**Technical Lead Review Date:** August 12, 2025  
**Version:** 1.0  
**Scope:** All Phase 2 conversation features and integrations

## 1. Testing Pyramid Strategy

### 1.1 Test Distribution

```
                    E2E Tests (5%)
                 Integration Tests (15%)
              Component Tests (30%)
           Unit Tests (50%)
```

**Rationale:** Heavy emphasis on unit and component tests for fast feedback, with focused integration and E2E tests for critical conversation flows.

### 1.2 Test Categories

| Test Level | Coverage | Tools | Execution Time |
|------------|----------|-------|----------------|
| **Unit Tests** | Individual functions/classes | pytest, pytest-asyncio | <5 minutes |
| **Component Tests** | Agent behavior, memory management | pytest, testcontainers | <15 minutes |
| **Integration Tests** | API endpoints, database operations | pytest, httpx, asyncpg | <30 minutes |
| **E2E Tests** | Full conversation flows | playwright, websocket client | <60 minutes |
| **Performance Tests** | Load, stress, memory usage | locust, pytest-benchmark | Variable |

## 2. Unit Testing Framework

### 2.1 Conversation Context Testing

```python
# test_conversation_context.py
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

from src.conversation.context import ConversationContext, ConversationContextCache
from src.conversation.models import ConversationMessage

class TestConversationContext:
    """Unit tests for conversation context management."""
    
    def test_conversation_context_creation(self):
        """Test conversation context object creation and validation."""
        context = ConversationContext(
            conversation_id="test_conv_123",
            session_id="test_session_456", 
            user_id="test_user_789",
            current_topic="security_analysis",
            domain_focus="security"
        )
        
        assert context.conversation_id == "test_conv_123"
        assert context.domain_focus == "security"
        assert context.complexity_level == "medium"  # default value
        assert isinstance(context.user_preferences, dict)
    
    def test_conversation_context_validation(self):
        """Test conversation context input validation."""
        # Test required fields
        with pytest.raises(ValueError):
            ConversationContext(conversation_id="", session_id="test", user_id="test")
        
        # Test domain focus validation
        with pytest.raises(ValueError):
            ConversationContext(
                conversation_id="test", 
                session_id="test", 
                user_id="test",
                domain_focus="invalid_domain"
            )
    
    @pytest.mark.asyncio
    async def test_conversation_context_cache(self):
        """Test conversation context caching mechanisms."""
        cache = ConversationContextCache(max_size=10, ttl_seconds=60)
        
        context = ConversationContext(
            conversation_id="cache_test",
            session_id="test",
            user_id="test"
        )
        
        # Test cache miss
        result = await cache.get("cache_test")
        assert result is None
        
        # Test cache store and hit
        await cache.put("cache_test", context)
        result = await cache.get("cache_test")
        assert result is not None
        assert result.conversation_id == "cache_test"
    
    @pytest.mark.asyncio 
    async def test_conversation_context_ttl(self):
        """Test TTL-based cache expiration."""
        cache = ConversationContextCache(max_size=10, ttl_seconds=0.1)  # 100ms TTL
        
        context = ConversationContext(
            conversation_id="ttl_test",
            session_id="test", 
            user_id="test"
        )
        
        await cache.put("ttl_test", context)
        
        # Should be cached immediately
        result = await cache.get("ttl_test")
        assert result is not None
        
        # Wait for TTL expiration
        await asyncio.sleep(0.2)
        
        # Should be expired
        result = await cache.get("ttl_test")
        assert result is None

class TestConversationMessage:
    """Unit tests for conversation message handling."""
    
    def test_message_creation_with_metadata(self):
        """Test conversation message creation with metadata."""
        message = ConversationMessage(
            id="msg_123",
            conversation_id="conv_456",
            session_id="session_789",
            role="user",
            content="What's my security score?",
            timestamp=datetime.now(),
            metadata={
                "user_agent": "test_client",
                "ip_address": "127.0.0.1"
            }
        )
        
        assert message.role == "user"
        assert "security score" in message.content
        assert message.metadata["user_agent"] == "test_client"
    
    def test_message_role_validation(self):
        """Test message role validation."""
        valid_roles = ["user", "assistant", "system"]
        
        for role in valid_roles:
            message = ConversationMessage(
                id="test",
                conversation_id="test",
                session_id="test",
                role=role,
                content="test content",
                timestamp=datetime.now()
            )
            assert message.role == role
        
        # Test invalid role
        with pytest.raises(ValueError):
            ConversationMessage(
                id="test",
                conversation_id="test", 
                session_id="test",
                role="invalid_role",
                content="test",
                timestamp=datetime.now()
            )
```

### 2.2 Agent Delegation Testing

```python
# test_agent_delegation.py
import pytest
from unittest.mock import AsyncMock, Mock, patch

from src.agents.coordinator import ConversationAwareCoordinator
from src.agents.delegation import AgentDelegationManager
from src.conversation.context import ConversationContext

class TestAgentDelegation:
    """Unit tests for conversation-aware agent delegation."""
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        return {
            "direct_agent": AsyncMock(),
            "hybrid_agent": AsyncMock(), 
            "security_agent": AsyncMock()
        }
    
    @pytest.fixture
    def delegation_manager(self, mock_agents):
        """Create delegation manager with mock agents."""
        return AgentDelegationManager(agents=mock_agents)
    
    @pytest.mark.asyncio
    async def test_simple_query_delegation(self, delegation_manager):
        """Test delegation of simple queries to direct agent."""
        context = ConversationContext(
            conversation_id="test",
            session_id="test",
            user_id="test",
            complexity_level="simple"
        )
        
        query = "What's my security score?"
        
        # Mock direct agent response
        delegation_manager.agents["direct_agent"].process_query.return_value = {
            "content": "Your security score is 85/100",
            "agent_used": "direct_agent"
        }
        
        result = await delegation_manager.delegate_query(query, context)
        
        assert result["agent_used"] == "direct_agent"
        delegation_manager.agents["direct_agent"].process_query.assert_called_once()
        delegation_manager.agents["hybrid_agent"].process_query.assert_not_called()
        delegation_manager.agents["security_agent"].process_query.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_complex_query_delegation(self, delegation_manager):
        """Test delegation of complex queries to security agent."""
        context = ConversationContext(
            conversation_id="test",
            session_id="test",
            user_id="test",
            domain_focus="security",
            complexity_level="complex"
        )
        
        query = "Perform comprehensive security audit with compliance analysis"
        
        # Mock security agent response
        delegation_manager.agents["security_agent"].process_query.return_value = {
            "content": "Comprehensive security audit completed...",
            "agent_used": "security_agent"
        }
        
        result = await delegation_manager.delegate_query(query, context)
        
        assert result["agent_used"] == "security_agent"
        delegation_manager.agents["security_agent"].process_query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delegation_fallback_on_failure(self, delegation_manager):
        """Test fallback delegation when primary agent fails."""
        context = ConversationContext(
            conversation_id="test",
            session_id="test", 
            user_id="test",
            complexity_level="medium"
        )
        
        query = "Security analysis"
        
        # Primary agent fails
        delegation_manager.agents["hybrid_agent"].process_query.side_effect = Exception("Agent unavailable")
        
        # Fallback agent succeeds
        delegation_manager.agents["direct_agent"].process_query.return_value = {
            "content": "Fallback security analysis",
            "agent_used": "direct_agent"
        }
        
        result = await delegation_manager.delegate_query(query, context)
        
        assert result["agent_used"] == "direct_agent"
        # Verify fallback was used
        delegation_manager.agents["hybrid_agent"].process_query.assert_called_once()
        delegation_manager.agents["direct_agent"].process_query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_conversation_context_influences_delegation(self, delegation_manager):
        """Test that conversation history influences agent selection."""
        # Create context indicating user prefers detailed analysis
        context = ConversationContext(
            conversation_id="test",
            session_id="test",
            user_id="test",
            user_preferences={"analysis_depth": "detailed"},
            successful_delegation_patterns=["security_agent"]
        )
        
        query = "Check my permissions"
        
        # Mock security agent response  
        delegation_manager.agents["security_agent"].process_query.return_value = {
            "content": "Detailed permission analysis...",
            "agent_used": "security_agent"
        }
        
        result = await delegation_manager.delegate_query(query, context)
        
        # Should delegate to security agent based on user preferences
        assert result["agent_used"] == "security_agent"
```

## 3. Component Testing Framework

### 3.1 Memory Management Testing

```python
# test_conversation_memory.py
import pytest
import asyncio
from datetime import datetime, timedelta
from testcontainers.redis import RedisContainer

from src.conversation.memory import ConversationMemoryManager
from src.conversation.context import ConversationContext

class TestConversationMemoryManager:
    """Component tests for conversation memory management."""
    
    @pytest.fixture(scope="session")
    def redis_container(self):
        """Start Redis container for testing."""
        with RedisContainer("redis:7-alpine") as redis:
            yield redis
    
    @pytest.fixture
    async def memory_manager(self, redis_container):
        """Create memory manager with test Redis instance."""
        redis_url = redis_container.get_connection_url()
        manager = ConversationMemoryManager(redis_url=redis_url)
        await manager.initialize()
        yield manager
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_context(self, memory_manager):
        """Test storing and retrieving conversation context."""
        context = ConversationContext(
            conversation_id="memory_test_1",
            session_id="session_1",
            user_id="user_1",
            current_topic="security",
            user_preferences={"analysis_level": "detailed"}
        )
        
        # Store context
        success = await memory_manager.store_conversation_context(context)
        assert success is True
        
        # Retrieve context
        retrieved = await memory_manager.get_conversation_context("memory_test_1")
        assert retrieved is not None
        assert retrieved.conversation_id == "memory_test_1"
        assert retrieved.current_topic == "security"
        assert retrieved.user_preferences["analysis_level"] == "detailed"
    
    @pytest.mark.asyncio
    async def test_conversation_pattern_learning(self, memory_manager):
        """Test conversation pattern learning and retrieval."""
        # Store multiple conversation patterns
        patterns = [
            {"query_type": "security_scan", "agent_used": "security_agent", "success": True},
            {"query_type": "iam_analysis", "agent_used": "direct_agent", "success": True},
            {"query_type": "security_scan", "agent_used": "hybrid_agent", "success": False}
        ]
        
        for pattern in patterns:
            await memory_manager.store_delegation_pattern(
                user_id="user_1",
                pattern=pattern
            )
        
        # Retrieve learned patterns
        learned_patterns = await memory_manager.get_user_delegation_patterns("user_1")
        
        assert len(learned_patterns) > 0
        # Should prefer security_agent for security_scan (higher success rate)
        security_patterns = [p for p in learned_patterns if p["query_type"] == "security_scan"]
        successful_patterns = [p for p in security_patterns if p["success"]]
        assert len(successful_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_and_ttl(self, memory_manager):
        """Test memory cleanup and TTL functionality."""
        # Store context with short TTL
        context = ConversationContext(
            conversation_id="ttl_test",
            session_id="session_ttl",
            user_id="user_ttl"
        )
        
        await memory_manager.store_conversation_context(context, ttl_seconds=1)
        
        # Should be available immediately
        retrieved = await memory_manager.get_conversation_context("ttl_test")
        assert retrieved is not None
        
        # Wait for TTL expiration
        await asyncio.sleep(2)
        
        # Should be expired
        retrieved = await memory_manager.get_conversation_context("ttl_test")
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_concurrent_memory_operations(self, memory_manager):
        """Test concurrent memory operations for race conditions."""
        async def store_context(conv_id: str):
            context = ConversationContext(
                conversation_id=conv_id,
                session_id=f"session_{conv_id}",
                user_id="concurrent_user"
            )
            return await memory_manager.store_conversation_context(context)
        
        # Run concurrent operations
        conv_ids = [f"concurrent_{i}" for i in range(50)]
        results = await asyncio.gather(*[store_context(conv_id) for conv_id in conv_ids])
        
        # All operations should succeed
        assert all(results)
        
        # All contexts should be retrievable
        for conv_id in conv_ids:
            context = await memory_manager.get_conversation_context(conv_id)
            assert context is not None
            assert context.conversation_id == conv_id
```

### 3.2 WebSocket Communication Testing

```python
# test_websocket_manager.py
import pytest
import asyncio
import json
from unittest.mock import AsyncMock, Mock
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

from src.api.websocket import ConversationWebSocketManager
from src.conversation.models import ConversationMessage

class TestConversationWebSocketManager:
    """Component tests for WebSocket conversation management."""
    
    @pytest.fixture
    def websocket_manager(self):
        """Create WebSocket manager for testing."""
        return ConversationWebSocketManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket connection."""
        ws = Mock(spec=WebSocket)
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()
        ws.receive_text = AsyncMock()
        return ws
    
    @pytest.mark.asyncio
    async def test_websocket_connection_management(self, websocket_manager, mock_websocket):
        """Test WebSocket connection establishment and tracking."""
        # Test connection
        await websocket_manager.connect(mock_websocket, "user123", "conv456")
        
        # Verify connection tracking
        assert len(websocket_manager.connections) == 1
        assert "user123" in websocket_manager.user_connections
        
        # Test connection metadata
        connection_id = list(websocket_manager.connections.keys())[0]
        metadata = websocket_manager.connection_metadata[connection_id]
        assert metadata["user_id"] == "user123"
        assert metadata["conversation_id"] == "conv456"
    
    @pytest.mark.asyncio
    async def test_websocket_message_broadcasting(self, websocket_manager, mock_websocket):
        """Test message broadcasting to relevant connections."""
        # Connect multiple WebSockets for same conversation
        await websocket_manager.connect(mock_websocket, "user1", "conv123")
        
        mock_websocket2 = Mock(spec=WebSocket)
        mock_websocket2.accept = AsyncMock()
        mock_websocket2.send_text = AsyncMock()
        await websocket_manager.connect(mock_websocket2, "user2", "conv123")
        
        # Broadcast message to conversation
        test_message = {
            "type": "message_response",
            "content": "Test response",
            "conversation_id": "conv123"
        }
        
        await websocket_manager.broadcast_conversation_update("conv123", test_message)
        
        # Both connections should receive the message
        mock_websocket.send_text.assert_called_once()
        mock_websocket2.send_text.assert_called_once()
        
        # Verify message content
        sent_message = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_message["type"] == "message_response"
        assert sent_message["content"] == "Test response"
    
    @pytest.mark.asyncio
    async def test_websocket_disconnection_cleanup(self, websocket_manager, mock_websocket):
        """Test proper cleanup on WebSocket disconnection."""
        # Connect WebSocket
        await websocket_manager.connect(mock_websocket, "user123", "conv456")
        connection_id = list(websocket_manager.connections.keys())[0]
        
        # Verify connection exists
        assert connection_id in websocket_manager.connections
        assert "user123" in websocket_manager.user_connections
        
        # Disconnect
        await websocket_manager.disconnect(connection_id)
        
        # Verify cleanup
        assert connection_id not in websocket_manager.connections
        assert connection_id not in websocket_manager.connection_metadata
        assert "user123" not in websocket_manager.user_connections
    
    @pytest.mark.asyncio
    async def test_websocket_error_handling(self, websocket_manager):
        """Test WebSocket error handling and recovery."""
        mock_websocket = Mock(spec=WebSocket)
        mock_websocket.accept = AsyncMock(side_effect=Exception("Connection failed"))
        
        # Connection should fail gracefully
        await websocket_manager.connect(mock_websocket, "user123", "conv456")
        
        # Should not have created connection entry
        assert len(websocket_manager.connections) == 0
        assert "user123" not in websocket_manager.user_connections
```

## 4. Integration Testing Framework

### 4.1 API Endpoint Testing

```python
# test_api_integration.py
import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient

from src.main import app
from src.conversation.models import ChatRequest, ChatResponse

class TestConversationAPIIntegration:
    """Integration tests for conversation API endpoints."""
    
    @pytest.fixture
    async def test_client(self):
        """Create test client for API testing."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_chat_endpoint_with_context(self, test_client):
        """Test chat endpoint with conversation context."""
        # First message - establishes context
        first_request = {
            "query": "What's my GCP security score?",
            "user_id": "test_user",
            "message_type": "chat"
        }
        
        response = await test_client.post("/api/v1/agent/chat", json=first_request)
        assert response.status_code == 200
        
        first_data = response.json()
        assert first_data["success"] is True
        assert "security" in first_data["response"].lower()
        
        session_id = first_data["session_id"]
        conversation_id = first_data["conversation_id"]
        
        # Follow-up message - should use context
        followup_request = {
            "query": "How can I improve it?",
            "user_id": "test_user",
            "session_id": session_id,
            "conversation_id": conversation_id,
            "message_type": "follow_up"
        }
        
        response = await test_client.post("/api/v1/agent/chat", json=followup_request)
        assert response.status_code == 200
        
        followup_data = response.json()
        assert followup_data["success"] is True
        # Response should reference improving security score
        assert any(keyword in followup_data["response"].lower() 
                  for keyword in ["improve", "security", "score", "recommendation"])
    
    @pytest.mark.asyncio
    async def test_session_management_endpoints(self, test_client):
        """Test session management API endpoints."""
        # Create new session
        session_request = {
            "user_id": "test_user",
            "session_type": "chat"
        }
        
        response = await test_client.post("/api/v1/agent/sessions", json=session_request)
        assert response.status_code == 200
        
        session_data = response.json()
        assert session_data["success"] is True
        session_id = session_data["session_id"]
        
        # Get user sessions
        response = await test_client.get(f"/api/v1/agent/sessions/test_user")
        assert response.status_code == 200
        
        sessions_data = response.json()
        assert sessions_data["success"] is True
        assert len(sessions_data["sessions"]) >= 1
        
        # Close session
        response = await test_client.delete(f"/api/v1/agent/sessions/{session_id}")
        assert response.status_code == 200
        
        close_data = response.json()
        assert close_data["success"] is True
    
    @pytest.mark.asyncio
    async def test_conversation_history_endpoint(self, test_client):
        """Test conversation history retrieval."""
        # Create conversation with messages
        chat_request = {
            "query": "Test conversation history",
            "user_id": "history_test_user"
        }
        
        response = await test_client.post("/api/v1/agent/chat", json=chat_request)
        conversation_data = response.json()
        conversation_id = conversation_data["conversation_id"]
        
        # Get conversation history
        response = await test_client.get(f"/api/v1/agent/conversations/{conversation_id}/history")
        assert response.status_code == 200
        
        history_data = response.json()
        assert history_data["success"] is True
        assert history_data["message_count"] >= 2  # User message + assistant response
        
        # Verify message structure
        messages = history_data["history"]
        assert any(msg["role"] == "user" for msg in messages)
        assert any(msg["role"] == "assistant" for msg in messages)
```

### 4.2 Database Integration Testing

```python
# test_database_integration.py
import pytest
import asyncio
from datetime import datetime
from testcontainers.postgres import PostgresContainer

from src.database.conversation_repository import ConversationRepository
from src.conversation.models import Conversation, ConversationMessage, ConversationContext

class TestDatabaseIntegration:
    """Integration tests for conversation database operations."""
    
    @pytest.fixture(scope="session")
    def postgres_container(self):
        """Start PostgreSQL container for testing."""
        with PostgresContainer("postgres:15-alpine") as postgres:
            yield postgres
    
    @pytest.fixture
    async def repository(self, postgres_container):
        """Create repository with test database."""
        db_url = postgres_container.get_connection_url()
        repo = ConversationRepository(db_url)
        await repo.initialize_schema()
        yield repo
        await repo.cleanup()
    
    @pytest.mark.asyncio
    async def test_conversation_crud_operations(self, repository):
        """Test basic CRUD operations for conversations."""
        # Create conversation
        conversation = Conversation(
            id="test_conv_123",
            session_id="test_session_456",
            user_id="test_user_789",
            topic="Database Test",
            domain_focus="testing"
        )
        
        created = await repository.create_conversation(conversation)
        assert created is True
        
        # Read conversation
        retrieved = await repository.get_conversation("test_conv_123")
        assert retrieved is not None
        assert retrieved.id == "test_conv_123"
        assert retrieved.topic == "Database Test"
        
        # Update conversation
        retrieved.topic = "Updated Database Test"
        updated = await repository.update_conversation(retrieved)
        assert updated is True
        
        # Verify update
        updated_conv = await repository.get_conversation("test_conv_123")
        assert updated_conv.topic == "Updated Database Test"
        
        # Delete conversation
        deleted = await repository.delete_conversation("test_conv_123")
        assert deleted is True
        
        # Verify deletion
        deleted_conv = await repository.get_conversation("test_conv_123")
        assert deleted_conv is None
    
    @pytest.mark.asyncio
    async def test_message_batch_operations(self, repository):
        """Test batch message operations for performance."""
        # Create test conversation
        conversation = Conversation(
            id="batch_test_conv",
            session_id="batch_session", 
            user_id="batch_user",
            topic="Batch Test"
        )
        await repository.create_conversation(conversation)
        
        # Create batch of messages
        messages = [
            ConversationMessage(
                id=f"msg_{i}",
                conversation_id="batch_test_conv",
                role="user" if i % 2 == 0 else "assistant",
                content=f"Test message {i}",
                timestamp=datetime.now()
            )
            for i in range(100)
        ]
        
        # Batch insert
        start_time = asyncio.get_event_loop().time()
        results = await repository.batch_insert_messages(messages)
        end_time = asyncio.get_event_loop().time()
        
        # Verify batch insert succeeded
        assert all(results)
        assert len(results) == 100
        
        # Verify performance (should be faster than individual inserts)
        batch_time = end_time - start_time
        assert batch_time < 5.0  # Should complete in under 5 seconds
        
        # Verify messages were stored correctly
        retrieved_messages = await repository.get_conversation_messages("batch_test_conv")
        assert len(retrieved_messages) == 100
    
    @pytest.mark.asyncio
    async def test_conversation_context_persistence(self, repository):
        """Test conversation context storage and retrieval."""
        # Create conversation with context
        conversation = Conversation(
            id="context_test_conv",
            session_id="context_session",
            user_id="context_user",
            topic="Context Test"
        )
        await repository.create_conversation(conversation)
        
        context = ConversationContext(
            conversation_id="context_test_conv",
            session_id="context_session",
            user_id="context_user",
            current_topic="security_analysis",
            domain_focus="security",
            user_preferences={"analysis_depth": "detailed", "format": "structured"},
            learned_patterns=["security_agent_preference", "detailed_analysis"]
        )
        
        # Store context
        stored = await repository.store_conversation_context(context)
        assert stored is True
        
        # Retrieve context
        retrieved_context = await repository.get_conversation_context("context_test_conv")
        assert retrieved_context is not None
        assert retrieved_context.current_topic == "security_analysis"
        assert retrieved_context.user_preferences["analysis_depth"] == "detailed"
        assert "security_agent_preference" in retrieved_context.learned_patterns
    
    @pytest.mark.asyncio
    async def test_database_transaction_rollback(self, repository):
        """Test database transaction rollback on errors."""
        # Start transaction that should fail
        with pytest.raises(Exception):
            async with repository.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Create valid conversation
                    await conn.execute(
                        "INSERT INTO conversations (id, session_id, user_id) VALUES ($1, $2, $3)",
                        "rollback_test", "session", "user"
                    )
                    
                    # Cause constraint violation to trigger rollback
                    await conn.execute(
                        "INSERT INTO conversations (id, session_id, user_id) VALUES ($1, $2, $3)",
                        "rollback_test", "session", "user"  # Duplicate ID
                    )
        
        # Verify transaction was rolled back
        conversation = await repository.get_conversation("rollback_test")
        assert conversation is None
```

## 5. End-to-End Testing Framework

### 5.1 Full Conversation Flow Testing

```python
# test_e2e_conversation_flows.py
import pytest
import asyncio
import json
from playwright.async_api import async_playwright
from websockets import connect

class TestE2EConversationFlows:
    """End-to-end tests for complete conversation flows."""
    
    @pytest.fixture(scope="session")
    async def browser_context(self):
        """Setup browser context for E2E testing."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            yield context
            await browser.close()
    
    @pytest.mark.asyncio
    async def test_complete_conversation_flow_via_ui(self, browser_context):
        """Test complete conversation flow through UI."""
        page = await browser_context.new_page()
        
        # Navigate to chat interface
        await page.goto("http://localhost:8501")
        
        # Wait for page load
        await page.wait_for_selector('[data-testid="stChatInput"]')
        
        # Send initial message
        chat_input = page.locator('[data-testid="stChatInput"]')
        await chat_input.fill("What's my GCP security posture?")
        await chat_input.press("Enter")
        
        # Wait for response
        await page.wait_for_selector('[data-testid="stChatMessage"]')
        
        # Verify response appears
        messages = await page.locator('[data-testid="stChatMessage"]').all()
        assert len(messages) >= 2  # User message + assistant response
        
        # Send follow-up message
        await chat_input.fill("What are the main security risks?")
        await chat_input.press("Enter")
        
        # Wait for context-aware response
        await page.wait_for_selector('[data-testid="stChatMessage"]', state="visible")
        
        # Verify conversation continuity
        all_messages = await page.locator('[data-testid="stChatMessage"]').all()
        assert len(all_messages) >= 4  # Two exchanges
        
        # Verify context preservation by checking response content
        last_response = await all_messages[-1].inner_text()
        assert any(keyword in last_response.lower() 
                  for keyword in ["risk", "security", "vulnerability"])
    
    @pytest.mark.asyncio
    async def test_websocket_real_time_updates(self):
        """Test real-time conversation updates via WebSocket."""
        websocket_url = "ws://localhost:8000/api/v1/agent/ws?user_id=e2e_test_user"
        
        async with connect(websocket_url) as websocket:
            # Send initial connection message
            await websocket.send(json.dumps({
                "type": "chat",
                "message": "Test real-time conversation",
                "conversation_id": "e2e_test_conv"
            }))
            
            # Receive acknowledgment
            ack_response = await websocket.recv()
            ack_data = json.loads(ack_response)
            assert ack_data["type"] == "message_received"
            
            # Receive agent response
            response_message = await websocket.recv()
            response_data = json.loads(response_message)
            assert response_data["type"] == "message_response"
            assert "response" in response_data
            
            # Send follow-up message
            await websocket.send(json.dumps({
                "type": "chat", 
                "message": "Continue the conversation",
                "conversation_id": "e2e_test_conv"
            }))
            
            # Verify context-aware response
            followup_ack = await websocket.recv()
            followup_response = await websocket.recv()
            followup_data = json.loads(followup_response)
            
            assert followup_data["type"] == "message_response"
            # Response should show conversation continuity
            assert len(followup_data["response"]) > 0
    
    @pytest.mark.asyncio
    async def test_multi_user_conversation_isolation(self):
        """Test conversation isolation between multiple users."""
        websocket_urls = [
            "ws://localhost:8000/api/v1/agent/ws?user_id=user_1",
            "ws://localhost:8000/api/v1/agent/ws?user_id=user_2"
        ]
        
        async def user_conversation(user_id: str, websocket_url: str):
            async with connect(websocket_url) as ws:
                # Send user-specific message
                await ws.send(json.dumps({
                    "type": "chat",
                    "message": f"This is {user_id}'s private conversation",
                    "conversation_id": f"{user_id}_private_conv"
                }))
                
                # Receive response
                await ws.recv()  # acknowledgment
                response = await ws.recv()  # actual response
                response_data = json.loads(response)
                
                return response_data["response"]
        
        # Run concurrent conversations
        user1_task = user_conversation("user_1", websocket_urls[0])
        user2_task = user_conversation("user_2", websocket_urls[1])
        
        user1_response, user2_response = await asyncio.gather(user1_task, user2_task)
        
        # Verify responses are different and isolated
        assert user1_response != user2_response
        assert "user_1" not in user2_response
        assert "user_2" not in user1_response
    
    @pytest.mark.asyncio
    async def test_conversation_persistence_across_sessions(self):
        """Test conversation persistence across browser sessions."""
        # First session - create conversation
        async with connect("ws://localhost:8000/api/v1/agent/ws?user_id=persistence_user") as ws1:
            await ws1.send(json.dumps({
                "type": "chat",
                "message": "Remember this conversation about security policies",
                "conversation_id": "persistence_test_conv"
            }))
            
            await ws1.recv()  # ack
            first_response = await ws1.recv()  # response
            first_data = json.loads(first_response)
        
        # Simulate session break
        await asyncio.sleep(1)
        
        # Second session - continue conversation
        async with connect("ws://localhost:8000/api/v1/agent/ws?user_id=persistence_user") as ws2:
            await ws2.send(json.dumps({
                "type": "chat",
                "message": "What were we discussing about policies?",
                "conversation_id": "persistence_test_conv"
            }))
            
            await ws2.recv()  # ack
            second_response = await ws2.recv()  # response
            second_data = json.loads(second_response)
        
        # Verify conversation context was preserved
        assert "security" in second_data["response"].lower() or "polic" in second_data["response"].lower()
```

## 6. Performance Testing Framework

### 6.1 Load Testing with Conversation Simulation

```python
# test_performance_load.py
import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from httpx import AsyncClient
import statistics

class TestConversationPerformanceLoad:
    """Performance tests for conversation features under load."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_conversation_load(self):
        """Test system performance with many concurrent conversations."""
        base_url = "http://localhost:8000"
        num_users = 50
        messages_per_user = 10
        
        async def simulate_user_conversation(user_id: int) -> dict:
            """Simulate a complete user conversation."""
            async with AsyncClient(base_url=base_url) as client:
                conversation_times = []
                
                for msg_idx in range(messages_per_user):
                    start_time = time.time()
                    
                    response = await client.post("/api/v1/agent/chat", json={
                        "query": f"User {user_id} message {msg_idx}: security analysis",
                        "user_id": f"load_test_user_{user_id}",
                        "message_type": "chat"
                    })
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    conversation_times.append(response_time)
                    
                    assert response.status_code == 200
                    assert response.json()["success"] is True
                
                return {
                    "user_id": user_id,
                    "total_time": sum(conversation_times),
                    "avg_response_time": statistics.mean(conversation_times),
                    "max_response_time": max(conversation_times),
                    "min_response_time": min(conversation_times)
                }
        
        # Run concurrent user simulations
        start_time = time.time()
        tasks = [simulate_user_conversation(i) for i in range(num_users)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Analyze results
        all_avg_times = [r["avg_response_time"] for r in results]
        all_max_times = [r["max_response_time"] for r in results]
        
        overall_avg = statistics.mean(all_avg_times)
        overall_max = max(all_max_times)
        
        # Performance assertions
        assert overall_avg < 2.0  # Average response time under 2 seconds
        assert overall_max < 5.0  # No response takes longer than 5 seconds
        assert total_time < 120.0  # Complete test under 2 minutes
        
        print(f"Load test completed: {num_users} users, {messages_per_user} messages each")
        print(f"Overall average response time: {overall_avg:.2f}s")
        print(f"Maximum response time: {overall_max:.2f}s")
        print(f"Total test duration: {total_time:.2f}s")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory usage with many active conversations."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many conversations
        num_conversations = 1000
        base_url = "http://localhost:8000"
        
        async with AsyncClient(base_url=base_url) as client:
            for i in range(num_conversations):
                await client.post("/api/v1/agent/chat", json={
                    "query": f"Memory test conversation {i}",
                    "user_id": f"memory_test_user_{i}",
                    "conversation_id": f"memory_test_conv_{i}"
                })
                
                # Check memory every 100 conversations
                if i % 100 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory
                    
                    # Memory increase should be reasonable
                    assert memory_increase < 500  # Less than 500MB increase
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: Initial: {initial_memory:.1f}MB, Final: {final_memory:.1f}MB")
        print(f"Total increase: {total_memory_increase:.1f}MB")
        
        # Total memory increase should be reasonable for 1000 conversations
        assert total_memory_increase < 1000  # Less than 1GB total increase
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_conversation_context_cache_performance(self):
        """Test conversation context cache performance under load."""
        from src.conversation.context import ConversationContextCache
        
        cache = ConversationContextCache(max_size=1000, ttl_seconds=3600)
        
        # Performance test for cache operations
        num_operations = 10000
        
        # Test cache writes
        start_time = time.time()
        for i in range(num_operations):
            context = ConversationContext(
                conversation_id=f"perf_test_{i}",
                session_id="perf_session",
                user_id="perf_user"
            )
            await cache.put(f"perf_test_{i}", context)
        write_time = time.time() - start_time
        
        # Test cache reads
        start_time = time.time()
        hit_count = 0
        for i in range(num_operations):
            result = await cache.get(f"perf_test_{i}")
            if result is not None:
                hit_count += 1
        read_time = time.time() - start_time
        
        # Performance assertions
        write_throughput = num_operations / write_time
        read_throughput = num_operations / read_time
        hit_ratio = hit_count / num_operations
        
        assert write_throughput > 1000  # At least 1000 writes/second
        assert read_throughput > 5000   # At least 5000 reads/second
        assert hit_ratio > 0.8          # At least 80% cache hit ratio
        
        print(f"Cache performance: Write: {write_throughput:.0f} ops/s, Read: {read_throughput:.0f} ops/s")
        print(f"Cache hit ratio: {hit_ratio:.2%}")
```

## 7. Test Execution and Reporting

### 7.1 Test Automation Pipeline

```yaml
# .github/workflows/conversation_tests.yml
name: Conversation Features Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements-test.txt
      
      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=src --cov-report=xml
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3

  component-tests:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      postgres:
        image: postgres:15-alpine
        env:
          POSTGRES_PASSWORD: testpassword
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements-test.txt
      
      - name: Run component tests
        run: |
          pytest tests/component/ -v
        env:
          REDIS_URL: redis://localhost:6379
          POSTGRES_URL: postgresql://postgres:testpassword@localhost:5432/test

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.11'
      
      - name: Start backend services
        run: |
          docker-compose -f docker-compose.test.yml up -d
          sleep 30  # Wait for services to start
      
      - name: Run integration tests
        run: |
          pytest tests/integration/ -v
      
      - name: Stop services
        run: |
          docker-compose -f docker-compose.test.yml down

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements-test.txt
          playwright install
      
      - name: Start full application
        run: |
          docker-compose up -d
          sleep 60  # Wait for full startup
      
      - name: Run E2E tests
        run: |
          pytest tests/e2e/ -v --headed=false
      
      - name: Stop application
        run: |
          docker-compose down

  performance-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements-test.txt
      
      - name: Start application for performance testing
        run: |
          docker-compose -f docker-compose.perf.yml up -d
          sleep 60
      
      - name: Run performance tests
        run: |
          pytest tests/performance/ -v --benchmark-json=benchmark.json
      
      - name: Upload performance results
        uses: actions/upload-artifact@v3
        with:
          name: performance-results
          path: benchmark.json
```

### 7.2 Test Reporting and Metrics

```python
# conftest.py - Global test configuration
import pytest
import asyncio
from datetime import datetime
import json

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def test_metrics_collector(request):
    """Collect test execution metrics."""
    start_time = datetime.now()
    
    yield
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Store test metrics
    metrics = {
        "test_name": request.node.nodeid,
        "duration": duration,
        "status": "passed" if not hasattr(request.node, "rep_call") or request.node.rep_call.passed else "failed",
        "timestamp": start_time.isoformat()
    }
    
    # Write to metrics file
    with open("test_metrics.jsonl", "a") as f:
        f.write(json.dumps(metrics) + "\n")

# pytest.ini configuration
[tool:pytest]
addopts = 
    -v 
    --strict-markers 
    --disable-warnings
    --cov=src
    --cov-report=html:htmlcov
    --cov-report=term-missing
    --cov-fail-under=90
    --asyncio-mode=auto

markers =
    unit: Unit tests
    component: Component tests  
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    slow: Slow running tests

testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

## 8. Test Data Management

### 8.1 Test Data Fixtures

```python
# tests/fixtures/conversation_data.py
import pytest
from datetime import datetime, timedelta
from typing import List

from src.conversation.models import Conversation, ConversationMessage, ConversationContext

@pytest.fixture
def sample_conversation() -> Conversation:
    """Sample conversation for testing."""
    return Conversation(
        id="test_conv_123",
        session_id="test_session_456",
        user_id="test_user_789",
        created_at=datetime.now(),
        topic="Security Analysis",
        domain_focus="security",
        status="active"
    )

@pytest.fixture
def sample_conversation_messages() -> List[ConversationMessage]:
    """Sample conversation messages for testing."""
    base_time = datetime.now()
    
    return [
        ConversationMessage(
            id="msg_1",
            conversation_id="test_conv_123",
            session_id="test_session_456",
            role="user",
            content="What's my GCP security score?",
            timestamp=base_time,
            metadata={"client": "test"}
        ),
        ConversationMessage(
            id="msg_2",
            conversation_id="test_conv_123", 
            session_id="test_session_456",
            role="assistant",
            content="Your GCP security score is 85/100. Here's the breakdown...",
            timestamp=base_time + timedelta(seconds=2),
            agent_used="security_agent",
            delegation_path=["coordinator", "security_agent"],
            metadata={"response_time": 1.5}
        ),
        ConversationMessage(
            id="msg_3",
            conversation_id="test_conv_123",
            session_id="test_session_456", 
            role="user",
            content="How can I improve my security score?",
            timestamp=base_time + timedelta(seconds=30),
            metadata={"client": "test"}
        ),
        ConversationMessage(
            id="msg_4",
            conversation_id="test_conv_123",
            session_id="test_session_456",
            role="assistant", 
            content="Based on your current security posture, here are recommendations...",
            timestamp=base_time + timedelta(seconds=33),
            agent_used="security_agent",
            delegation_path=["coordinator", "security_agent"],
            metadata={"response_time": 2.1, "context_used": True}
        )
    ]

@pytest.fixture
def sample_conversation_context() -> ConversationContext:
    """Sample conversation context for testing."""
    return ConversationContext(
        conversation_id="test_conv_123",
        session_id="test_session_456",
        user_id="test_user_789",
        current_topic="security_improvement",
        domain_focus="security",
        complexity_level="medium",
        user_preferences={
            "analysis_depth": "detailed",
            "response_format": "structured",
            "preferred_agent": "security_agent"
        },
        frequently_asked_patterns=[
            "security_score_inquiry",
            "improvement_recommendations"
        ],
        successful_delegation_patterns=[
            "security_agent_for_detailed_analysis",
            "direct_agent_for_quick_queries"
        ],
        average_response_time=1.8,
        preferred_agent_types=["security_agent", "hybrid_agent"]
    )
```

## 9. Continuous Testing and Quality Gates

### 9.1 Quality Gates Configuration

```yaml
# quality_gates.yml
quality_gates:
  conversation_features:
    unit_tests:
      coverage_threshold: 90%
      pass_rate_threshold: 100%
      max_duration: 5_minutes
    
    component_tests:
      pass_rate_threshold: 95%
      max_duration: 15_minutes
      memory_leak_threshold: 10MB
    
    integration_tests:
      pass_rate_threshold: 95%
      max_duration: 30_minutes
      response_time_threshold: 2s
    
    e2e_tests:
      pass_rate_threshold: 90%
      max_duration: 60_minutes
      user_flow_success_rate: 95%
    
    performance_tests:
      response_time_95th_percentile: 3s
      concurrent_users_support: 50
      memory_usage_threshold: 1GB
      error_rate_threshold: 1%

blocking_conditions:
  - unit_test_coverage_below_90%
  - integration_test_failure_rate_above_5%
  - performance_degradation_above_20%
  - memory_leak_detected
  - security_vulnerability_found
```

## Conclusion

This comprehensive testing strategy ensures that Phase 2 conversation features are thoroughly validated across all levels - from individual function behavior to complete user workflows. The strategy emphasizes:

1. **Fast Feedback**: Heavy emphasis on unit tests for quick validation
2. **Real-world Scenarios**: Component and integration tests with actual dependencies
3. **User Experience**: E2E tests validating complete conversation flows
4. **Performance**: Load testing and performance monitoring
5. **Quality**: Automated quality gates and continuous monitoring

**Implementation Priority:**
1. **Week 1**: Set up unit and component test framework
2. **Week 2**: Implement integration tests with test containers
3. **Week 3**: Create E2E test suite with browser automation
4. **Week 4**: Add performance testing and quality gates
5. **Week 5**: Integrate with CI/CD pipeline
6. **Week 6**: Documentation and team training

**Next Steps:**
1. Review and approve testing strategy with development team
2. Set up test infrastructure and tooling
3. Create initial test suite templates
4. Begin TDD implementation of conversation features
5. Establish performance baselines and monitoring