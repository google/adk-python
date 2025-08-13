# Phase 2: Integration Guidelines and Best Practices

## Overview

This document provides detailed integration guidelines, coding standards, and best practices for implementing Phase 2: Conversational ADK Orchestration features while maintaining backward compatibility and system reliability.

**Technical Lead Review Date:** August 12, 2025  
**Version:** 1.0  
**Target Audience:** Development Team, Code Reviewers

## 1. Integration Patterns

### 1.1 Backward Compatibility Requirements

**Critical Rule:** All Phase 2 features MUST maintain 100% backward compatibility with existing ADK agent patterns.

```python
# ✅ CORRECT: Extend existing patterns without breaking them
class EnhancedCoordinatorAgent(Agent):
    """Enhanced coordinator with conversation awareness."""
    
    def __init__(self, project_id: str, enable_conversation_features: bool = True):
        # Initialize base agent first
        super().__init__(...)
        
        # Add conversation features as optional enhancement
        if enable_conversation_features:
            self.conversation_manager = ConversationManager()
            self.memory_manager = ConversationMemoryManager()
        else:
            # Fallback to existing behavior
            self.conversation_manager = None
            self.memory_manager = None
    
    async def process_query(self, query: str, context: Optional[Dict] = None):
        """Process query with optional conversation enhancement."""
        # Always support original query format
        if self.conversation_manager and context:
            # Enhanced processing with conversation awareness
            return await self._process_with_conversation_context(query, context)
        else:
            # Original processing without conversation features
            return await self._process_standard_query(query)

# ❌ INCORRECT: Breaking existing agent interface
class ConversationOnlyAgent(Agent):
    def __init__(self, conversation_context: ConversationContext):  # Breaking change
        # Missing backward compatibility
        super().__init__(...)
```

### 1.2 Progressive Enhancement Strategy

```python
# Implementation strategy: Feature flags for gradual rollout
class FeatureFlags:
    """Feature flags for Phase 2 conversation features."""
    
    ENABLE_CONVERSATION_MEMORY = os.getenv("ENABLE_CONVERSATION_MEMORY", "false").lower() == "true"
    ENABLE_CROSS_CONVERSATION_LEARNING = os.getenv("ENABLE_CROSS_CONVERSATION_LEARNING", "false").lower() == "true"
    ENABLE_REAL_TIME_UPDATES = os.getenv("ENABLE_REAL_TIME_UPDATES", "false").lower() == "true"
    ENABLE_PERFORMANCE_OPTIMIZATION = os.getenv("ENABLE_PERFORMANCE_OPTIMIZATION", "true").lower() == "true"

# Use feature flags throughout implementation
if FeatureFlags.ENABLE_CONVERSATION_MEMORY:
    conversation_context = await memory_manager.get_context(conversation_id)
else:
    conversation_context = None
```

### 1.3 Agent Integration Contract

```python
from abc import ABC, abstractmethod
from typing import Protocol

class ConversationCapable(Protocol):
    """Protocol for agents that support conversation features."""
    
    async def process_with_conversation_context(
        self,
        query: str,
        context: ConversationContext,
        history: List[ConversationMessage]
    ) -> ConversationResponse:
        """Process query with full conversation awareness."""
        ...
    
    async def extract_conversation_insights(
        self,
        response: str,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Extract insights for conversation learning."""
        ...

# Extend existing agents to support conversation protocol
class ConversationAwareSecurityAgent(SecurityAgent, ConversationCapable):
    """Security agent with conversation capabilities."""
    
    async def process_with_conversation_context(self, query: str, context: ConversationContext, history: List[ConversationMessage]) -> ConversationResponse:
        # First, process with standard security agent logic
        standard_response = await super().process_query(query)
        
        # Then enhance with conversation awareness
        conversation_insights = await self._analyze_conversation_patterns(history, context)
        enhanced_response = await self._apply_conversation_insights(standard_response, conversation_insights)
        
        return ConversationResponse(
            content=enhanced_response,
            context_updates=conversation_insights,
            agent_used="security_agent_conversation_aware"
        )
```

## 2. Code Standards and Patterns

### 2.1 Async/Await Best Practices

```python
# ✅ CORRECT: Proper async/await patterns for conversation features
class ConversationManager:
    
    async def get_conversation_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Retrieve conversation context with timeout and error handling."""
        try:
            # Use asyncio.wait_for for timeout control
            context = await asyncio.wait_for(
                self._fetch_context_from_store(conversation_id),
                timeout=0.05  # 50ms timeout
            )
            return context
        except asyncio.TimeoutError:
            logger.warning(f"Context retrieval timeout for conversation {conversation_id}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving context for {conversation_id}: {e}")
            return None
    
    async def batch_update_contexts(self, updates: List[Tuple[str, ConversationContext]]) -> List[bool]:
        """Batch update multiple contexts efficiently."""
        # Use asyncio.gather for concurrent updates
        tasks = [
            self._update_single_context(conv_id, context)
            for conv_id, context in updates
        ]
        
        # Handle partial failures gracefully
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [not isinstance(result, Exception) for result in results]

# ❌ INCORRECT: Blocking operations in async context
async def bad_conversation_processing(self, query: str):
    # Don't do synchronous database calls
    context = sync_database_call(conversation_id)  # Blocks event loop
    
    # Don't use blocking file I/O
    with open("conversation_log.txt", "w") as f:  # Blocks event loop
        f.write(query)
```

### 2.2 Error Handling Patterns

```python
class ConversationError(Exception):
    """Base exception for conversation-related errors."""
    pass

class ConversationContextError(ConversationError):
    """Error related to conversation context operations."""
    pass

class ConversationMemoryError(ConversationError):
    """Error related to conversation memory operations."""
    pass

# ✅ CORRECT: Comprehensive error handling with graceful degradation
async def process_conversation_message(self, message: ConversationMessage) -> ConversationResponse:
    """Process message with comprehensive error handling."""
    try:
        # Try enhanced conversation processing
        context = await self._get_conversation_context(message.conversation_id)
        if context:
            return await self._process_with_context(message, context)
    except ConversationContextError as e:
        logger.warning(f"Conversation context error: {e}, falling back to stateless processing")
    except ConversationMemoryError as e:
        logger.warning(f"Memory error: {e}, using local context only")
    except Exception as e:
        logger.error(f"Unexpected conversation error: {e}, using standard processing")
    
    # Fallback to standard processing without conversation features
    return await self._process_standard_message(message)

# ✅ CORRECT: Circuit breaker pattern for external dependencies
class ConversationMemoryCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
            else:
                raise ConversationMemoryError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise
```

### 2.3 Performance Optimization Patterns

```python
# ✅ CORRECT: Efficient caching with TTL and LRU eviction
class ConversationContextCache:
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self._cache: Dict[str, Tuple[ConversationContext, float]] = {}
        self._access_order: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    async def get(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get context with TTL and LRU tracking."""
        current_time = time.time()
        
        if conversation_id in self._cache:
            context, timestamp = self._cache[conversation_id]
            
            # Check TTL
            if current_time - timestamp > self.ttl_seconds:
                await self._remove(conversation_id)
                return None
            
            # Update LRU order
            self._access_order.move_to_end(conversation_id)
            return context
        
        return None
    
    async def put(self, conversation_id: str, context: ConversationContext):
        """Store context with size-based eviction."""
        current_time = time.time()
        
        # Evict if at capacity
        if len(self._cache) >= self.max_size and conversation_id not in self._cache:
            oldest_id = next(iter(self._access_order))
            await self._remove(oldest_id)
        
        self._cache[conversation_id] = (context, current_time)
        self._access_order[conversation_id] = None
        self._access_order.move_to_end(conversation_id)

# ✅ CORRECT: Batch operations for efficiency
async def batch_process_conversation_updates(self, updates: List[ConversationUpdate]) -> BatchResult:
    """Process multiple conversation updates efficiently."""
    # Group updates by type for batch processing
    context_updates = [u for u in updates if u.type == "context"]
    memory_updates = [u for u in updates if u.type == "memory"]
    
    # Process each type in batch
    context_results = await self._batch_update_contexts(context_updates)
    memory_results = await self._batch_update_memory(memory_updates)
    
    return BatchResult(
        context_results=context_results,
        memory_results=memory_results,
        total_processed=len(updates)
    )
```

## 3. Database Integration Standards

### 3.1 Conversation Schema Design

```sql
-- Conversation tables with proper indexing and constraints
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    topic VARCHAR(255),
    domain_focus VARCHAR(50) DEFAULT 'general',
    complexity_level VARCHAR(20) DEFAULT 'medium',
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB DEFAULT '{}',
    
    -- Indexes for efficient queries
    INDEX idx_conversations_session_id (session_id),
    INDEX idx_conversations_user_id (user_id),
    INDEX idx_conversations_created_at (created_at),
    INDEX idx_conversations_domain_focus (domain_focus)
);

CREATE TABLE conversation_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    agent_used VARCHAR(100),
    delegation_path TEXT[],
    metadata JSONB DEFAULT '{}',
    
    -- Indexes for message retrieval
    INDEX idx_messages_conversation_id (conversation_id),
    INDEX idx_messages_timestamp (timestamp),
    INDEX idx_messages_agent_used (agent_used)
);

CREATE TABLE conversation_contexts (
    conversation_id UUID PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    context_data JSONB NOT NULL,
    user_preferences JSONB DEFAULT '{}',
    learned_patterns JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Index for efficient context lookups
    INDEX idx_context_updated_at (updated_at)
);
```

### 3.2 Database Access Patterns

```python
# ✅ CORRECT: Connection pooling and transaction management
class ConversationRepository:
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
    
    async def get_conversation_with_context(self, conversation_id: str) -> Optional[ConversationWithContext]:
        """Get conversation with context in a single transaction."""
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                # Use prepared statements for better performance
                conversation_row = await conn.fetchrow(
                    "SELECT * FROM conversations WHERE id = $1",
                    conversation_id
                )
                
                if not conversation_row:
                    return None
                
                # Get context and recent messages in parallel
                context_task = conn.fetchrow(
                    "SELECT * FROM conversation_contexts WHERE conversation_id = $1",
                    conversation_id
                )
                messages_task = conn.fetch(
                    """SELECT * FROM conversation_messages 
                       WHERE conversation_id = $1 
                       ORDER BY timestamp DESC LIMIT 50""",
                    conversation_id
                )
                
                context_row, message_rows = await asyncio.gather(context_task, messages_task)
                
                return ConversationWithContext(
                    conversation=Conversation.from_row(conversation_row),
                    context=ConversationContext.from_row(context_row) if context_row else None,
                    recent_messages=[ConversationMessage.from_row(row) for row in message_rows]
                )

# ✅ CORRECT: Batch operations with proper error handling
async def batch_insert_messages(self, messages: List[ConversationMessage]) -> List[bool]:
    """Insert multiple messages efficiently."""
    if not messages:
        return []
    
    async with self.db_pool.acquire() as conn:
        async with conn.transaction():
            try:
                # Use COPY for bulk inserts
                await conn.copy_records_to_table(
                    'conversation_messages',
                    records=[
                        (msg.id, msg.conversation_id, msg.role, msg.content, 
                         msg.timestamp, msg.agent_used, msg.delegation_path, msg.metadata)
                        for msg in messages
                    ],
                    columns=['id', 'conversation_id', 'role', 'content', 
                            'timestamp', 'agent_used', 'delegation_path', 'metadata']
                )
                return [True] * len(messages)
            except Exception as e:
                logger.error(f"Batch insert failed: {e}")
                # Fallback to individual inserts
                return await self._insert_messages_individually(messages)
```

## 4. WebSocket Integration Best Practices

### 4.1 Connection Management

```python
# ✅ CORRECT: Robust WebSocket connection management
class ConversationWebSocketManager:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, Set[str]] = {}
        self.connection_metadata: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str, conversation_id: str):
        """Establish WebSocket connection with proper tracking."""
        connection_id = f"{user_id}_{conversation_id}_{int(time.time())}"
        
        try:
            await websocket.accept()
            
            # Track connections
            self.connections[connection_id] = websocket
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)
            
            self.connection_metadata[connection_id] = {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "connected_at": datetime.now(),
                "last_activity": datetime.now()
            }
            
            # Send connection confirmation
            await self._send_to_connection(connection_id, {
                "type": "connection_established",
                "connection_id": connection_id,
                "conversation_id": conversation_id
            })
            
            logger.info(f"WebSocket connected: {connection_id}")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            await self._cleanup_connection(connection_id)
    
    async def disconnect(self, connection_id: str):
        """Clean disconnect with proper cleanup."""
        if connection_id in self.connections:
            metadata = self.connection_metadata.get(connection_id, {})
            user_id = metadata.get("user_id")
            
            # Remove from tracking
            del self.connections[connection_id]
            if connection_id in self.connection_metadata:
                del self.connection_metadata[connection_id]
            
            if user_id and user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def broadcast_conversation_update(self, conversation_id: str, update: Dict):
        """Broadcast update to all connections for a conversation."""
        relevant_connections = [
            conn_id for conn_id, metadata in self.connection_metadata.items()
            if metadata.get("conversation_id") == conversation_id
        ]
        
        if relevant_connections:
            await asyncio.gather(*[
                self._send_to_connection(conn_id, update)
                for conn_id in relevant_connections
            ], return_exceptions=True)
```

### 4.2 Real-Time Message Processing

```python
# ✅ CORRECT: Non-blocking real-time message processing
class RealTimeConversationProcessor:
    def __init__(self, websocket_manager: ConversationWebSocketManager):
        self.websocket_manager = websocket_manager
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.worker_tasks: List[asyncio.Task] = []
    
    async def start_workers(self, num_workers: int = 5):
        """Start background workers for message processing."""
        self.worker_tasks = [
            asyncio.create_task(self._worker(f"worker_{i}"))
            for i in range(num_workers)
        ]
    
    async def _worker(self, worker_name: str):
        """Background worker for processing conversation messages."""
        while True:
            try:
                message_data = await self.processing_queue.get()
                await self._process_conversation_message(message_data)
                self.processing_queue.task_done()
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
    
    async def queue_message_for_processing(self, message: ConversationMessage):
        """Queue message for non-blocking processing."""
        await self.processing_queue.put({
            "message": message,
            "timestamp": datetime.now(),
            "priority": self._calculate_priority(message)
        })
        
        # Send immediate acknowledgment via WebSocket
        await self.websocket_manager.broadcast_conversation_update(
            message.conversation_id,
            {
                "type": "message_received",
                "message_id": message.id,
                "status": "processing"
            }
        )
    
    async def _process_conversation_message(self, message_data: Dict):
        """Process message and broadcast updates."""
        message = message_data["message"]
        
        try:
            # Process with conversation context
            response = await self._get_agent_response(message)
            
            # Broadcast response via WebSocket
            await self.websocket_manager.broadcast_conversation_update(
                message.conversation_id,
                {
                    "type": "message_response",
                    "message_id": message.id,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Message processing failed: {e}")
            await self.websocket_manager.broadcast_conversation_update(
                message.conversation_id,
                {
                    "type": "message_error",
                    "message_id": message.id,
                    "error": str(e)
                }
            )
```

## 5. Testing Integration Standards

### 5.1 Test Structure for Conversation Features

```python
# ✅ CORRECT: Comprehensive test structure
class TestConversationFeatures:
    """Test suite for conversation-aware ADK features."""
    
    @pytest.fixture
    async def conversation_setup(self):
        """Setup conversation test environment."""
        # Create test database
        test_db = await create_test_database()
        
        # Create test agents with conversation capabilities
        agents = await setup_test_agents()
        
        # Create test conversation manager
        conversation_manager = ConversationManager(test_db)
        
        yield {
            "db": test_db,
            "agents": agents,
            "conversation_manager": conversation_manager
        }
        
        # Cleanup
        await cleanup_test_database(test_db)
    
    async def test_conversation_context_preservation(self, conversation_setup):
        """Test that conversation context is preserved across messages."""
        manager = conversation_setup["conversation_manager"]
        
        # Start a conversation about security
        conv_id = await manager.create_conversation("user123", "Security Analysis")
        
        # Send initial message
        response1 = await manager.process_message(
            ConversationMessage(
                conversation_id=conv_id,
                role="user",
                content="What's my security score?"
            )
        )
        
        # Send follow-up that requires context
        response2 = await manager.process_message(
            ConversationMessage(
                conversation_id=conv_id,
                role="user", 
                content="What can I do to improve it?"  # Requires context from previous message
            )
        )
        
        # Verify context was preserved
        assert "security score" in response2.content.lower()
        assert response2.context_updates is not None
        assert response2.delegation_path is not None
    
    async def test_agent_delegation_optimization(self, conversation_setup):
        """Test that agent delegation improves based on conversation history."""
        manager = conversation_setup["conversation_manager"]
        
        # Simulate multiple conversations with similar patterns
        for i in range(10):
            conv_id = await manager.create_conversation(f"user{i}", "IAM Analysis")
            
            # Always ask about IAM permissions
            await manager.process_message(
                ConversationMessage(
                    conversation_id=conv_id,
                    role="user",
                    content="Show me IAM permissions for my project"
                )
            )
        
        # Check that IAM queries are now routed to optimal agent
        delegation_stats = await manager.get_delegation_statistics("iam")
        assert delegation_stats["preferred_agent"] in ["direct_agent", "iam_specialist"]
        assert delegation_stats["success_rate"] > 0.8
    
    @pytest.mark.performance
    async def test_conversation_performance_under_load(self, conversation_setup):
        """Test conversation performance with multiple concurrent users."""
        manager = conversation_setup["conversation_manager"]
        
        # Create concurrent conversations
        async def simulate_user_conversation(user_id: str):
            conv_id = await manager.create_conversation(user_id, "Load Test")
            
            start_time = time.time()
            
            # Send multiple messages in conversation
            for i in range(5):
                await manager.process_message(
                    ConversationMessage(
                        conversation_id=conv_id,
                        role="user",
                        content=f"Security question {i}"
                    )
                )
            
            return time.time() - start_time
        
        # Run 50 concurrent conversations
        tasks = [
            simulate_user_conversation(f"load_test_user_{i}")
            for i in range(50)
        ]
        
        conversation_times = await asyncio.gather(*tasks)
        
        # Verify performance requirements
        avg_time = sum(conversation_times) / len(conversation_times)
        assert avg_time < 10.0  # 10 seconds for 5-message conversation
        assert max(conversation_times) < 20.0  # No conversation takes >20s
```

### 5.2 Integration Test Patterns

```python
# ✅ CORRECT: End-to-end integration testing
class TestConversationIntegration:
    """Integration tests for conversation features with real agents."""
    
    async def test_full_conversation_flow(self):
        """Test complete conversation flow from WebSocket to agent response."""
        # Setup WebSocket test client
        async with WebSocketTestClient() as ws_client:
            # Connect to conversation endpoint
            await ws_client.connect("/api/v1/agent/ws", params={"user_id": "test_user"})
            
            # Send conversation message
            await ws_client.send_json({
                "type": "chat",
                "message": "Analyze my GCP security posture",
                "conversation_id": "test_conv_123"
            })
            
            # Verify immediate acknowledgment
            ack_message = await ws_client.receive_json()
            assert ack_message["type"] == "message_received"
            
            # Wait for agent response
            response_message = await ws_client.receive_json()
            assert response_message["type"] == "message_response"
            assert "security" in response_message["response"].lower()
            
            # Send follow-up message
            await ws_client.send_json({
                "type": "chat",
                "message": "What are the biggest risks?",
                "conversation_id": "test_conv_123"
            })
            
            # Verify context-aware response
            followup_response = await ws_client.receive_json()
            assert followup_response["type"] == "message_response"
            # Response should reference previous security analysis
            assert any(keyword in followup_response["response"].lower() 
                      for keyword in ["risk", "security", "finding"])
```

## 6. Monitoring and Observability

### 6.1 Conversation Metrics

```python
# ✅ CORRECT: Comprehensive conversation monitoring
class ConversationMetrics:
    """Metrics collection for conversation features."""
    
    def __init__(self):
        self.conversation_counter = Counter()
        self.response_time_histogram = Histogram()
        self.agent_delegation_counter = Counter()
        self.error_counter = Counter()
        self.context_hit_ratio = Gauge()
    
    def record_conversation_start(self, user_id: str, conversation_type: str):
        """Record conversation start metrics."""
        self.conversation_counter.labels(
            user_id=user_id,
            type=conversation_type
        ).inc()
    
    def record_response_time(self, agent_type: str, response_time: float):
        """Record agent response time."""
        self.response_time_histogram.labels(
            agent_type=agent_type
        ).observe(response_time)
    
    def record_delegation_decision(self, from_agent: str, to_agent: str, reason: str):
        """Record agent delegation for optimization analysis."""
        self.agent_delegation_counter.labels(
            from_agent=from_agent,
            to_agent=to_agent,
            reason=reason
        ).inc()
    
    def update_context_hit_ratio(self, hits: int, total: int):
        """Update conversation context cache hit ratio."""
        self.context_hit_ratio.set(hits / total if total > 0 else 0)

# Prometheus metrics exposure
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@app.get("/metrics")
async def get_metrics():
    """Expose Prometheus metrics for conversation features."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

### 6.2 Health Checks

```python
# ✅ CORRECT: Conversation-aware health checks
class ConversationHealthChecker:
    """Health checks for conversation features."""
    
    async def check_conversation_memory_health(self) -> HealthStatus:
        """Check conversation memory system health."""
        try:
            # Test memory write/read
            test_context = ConversationContext(
                conversation_id="health_check",
                session_id="health_check",
                user_id="health_check"
            )
            
            start_time = time.time()
            await self.memory_manager.store_context(test_context)
            retrieved_context = await self.memory_manager.get_context("health_check")
            response_time = time.time() - start_time
            
            if retrieved_context and response_time < 0.1:  # 100ms threshold
                return HealthStatus.HEALTHY
            else:
                return HealthStatus.DEGRADED
                
        except Exception as e:
            logger.error(f"Conversation memory health check failed: {e}")
            return HealthStatus.UNHEALTHY
    
    async def check_agent_delegation_health(self) -> HealthStatus:
        """Check agent delegation system health."""
        try:
            # Test agent delegation path
            test_query = "health check query"
            coordinator = self.agent_factory.get_coordinator()
            
            start_time = time.time()
            response = await coordinator.process_query(test_query)
            response_time = time.time() - start_time
            
            if response and response_time < 2.0:  # 2s threshold
                return HealthStatus.HEALTHY
            else:
                return HealthStatus.DEGRADED
                
        except Exception as e:
            logger.error(f"Agent delegation health check failed: {e}")
            return HealthStatus.UNHEALTHY
```

## 7. Deployment Considerations

### 7.1 Configuration Management

```yaml
# conversation_config.yaml
conversation_features:
  enabled: true
  memory:
    provider: "redis"  # redis, postgresql, memory
    ttl_seconds: 3600
    max_conversations_per_user: 100
  
  performance:
    cache_size: 1000
    batch_size: 50
    worker_threads: 5
    response_timeout_seconds: 30
  
  agents:
    enable_context_sharing: true
    delegation_optimization: true
    performance_tracking: true
  
  websocket:
    max_connections: 1000
    heartbeat_interval: 30
    message_queue_size: 100

# Environment-specific overrides
environments:
  development:
    conversation_features:
      memory:
        provider: "memory"
      performance:
        cache_size: 100
  
  production:
    conversation_features:
      memory:
        provider: "redis"
        connection_pool_size: 10
      performance:
        cache_size: 5000
        worker_threads: 10
```

### 7.2 Migration Strategy

```python
# ✅ CORRECT: Safe migration approach
class ConversationFeatureMigration:
    """Handles migration to conversation-aware features."""
    
    async def migrate_existing_sessions(self):
        """Migrate existing sessions to conversation format."""
        logger.info("Starting conversation feature migration...")
        
        # Get all existing sessions
        existing_sessions = await self.get_existing_sessions()
        
        migration_stats = {
            "total_sessions": len(existing_sessions),
            "migrated": 0,
            "failed": 0
        }
        
        for session in existing_sessions:
            try:
                # Create conversation from session
                conversation = await self.create_conversation_from_session(session)
                
                # Migrate messages if any
                if session.get("messages"):
                    await self.migrate_session_messages(session["messages"], conversation.id)
                
                migration_stats["migrated"] += 1
                
            except Exception as e:
                logger.error(f"Failed to migrate session {session.id}: {e}")
                migration_stats["failed"] += 1
        
        logger.info(f"Migration completed: {migration_stats}")
        return migration_stats
    
    async def enable_conversation_features_gradually(self, percentage: float = 0.1):
        """Enable conversation features for a percentage of users."""
        # Feature flag approach for gradual rollout
        users_with_feature = await self.get_users_for_feature_rollout(percentage)
        
        for user_id in users_with_feature:
            await self.enable_conversation_features_for_user(user_id)
        
        logger.info(f"Enabled conversation features for {len(users_with_feature)} users")
```

## 8. Security Considerations

### 8.1 Conversation Data Security

```python
# ✅ CORRECT: Secure conversation data handling
class ConversationSecurityManager:
    """Manages security for conversation data."""
    
    def __init__(self, encryption_key: str):
        self.fernet = Fernet(encryption_key.encode())
    
    async def encrypt_conversation_data(self, data: str) -> str:
        """Encrypt sensitive conversation data."""
        encrypted = self.fernet.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
    
    async def decrypt_conversation_data(self, encrypted_data: str) -> str:
        """Decrypt conversation data."""
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted = self.fernet.decrypt(encrypted_bytes)
        return decrypted.decode()
    
    async def sanitize_conversation_message(self, message: ConversationMessage) -> ConversationMessage:
        """Sanitize message content before storage."""
        # Remove sensitive patterns
        sanitized_content = self._remove_sensitive_patterns(message.content)
        
        # Validate user permissions
        if not await self.validate_user_permissions(message.user_id, message.conversation_id):
            raise PermissionError("User not authorized for conversation")
        
        return ConversationMessage(
            **message.dict(),
            content=sanitized_content
        )
    
    def _remove_sensitive_patterns(self, content: str) -> str:
        """Remove sensitive information patterns."""
        # Remove potential API keys, passwords, etc.
        sensitive_patterns = [
            r'AIza[0-9A-Za-z\-_]{35}',  # Google API keys
            r'AKIA[0-9A-Z]{16}',        # AWS access keys  
            r'[a-zA-Z0-9]{40}',         # General 40-char tokens
        ]
        
        for pattern in sensitive_patterns:
            content = re.sub(pattern, '[REDACTED]', content)
        
        return content
```

## Conclusion

These integration guidelines provide a comprehensive framework for implementing Phase 2 conversation features while maintaining system reliability, performance, and security. Following these patterns ensures:

1. **Backward Compatibility**: All existing functionality continues to work
2. **Performance**: Optimized patterns for high-throughput conversation processing  
3. **Reliability**: Comprehensive error handling and graceful degradation
4. **Scalability**: Efficient resource usage and horizontal scaling support
5. **Security**: Proper data protection and access controls
6. **Maintainability**: Clear code patterns and comprehensive testing

**Next Steps:**
1. Review and approve these guidelines with the development team
2. Create code templates and scaffolding based on these patterns
3. Implement feature flags for gradual rollout
4. Set up monitoring and alerting for conversation features
5. Begin Phase 2A implementation following these standards