# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from google.genai import types
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.memory.chroma_memory_service import ChromaMemoryService
from google.adk.sessions.session import Session

@pytest.fixture
def memory_service():
    """Creates a ChromaMemoryService instance for testing."""
    service = ChromaMemoryService(top_k=5, is_persistent=False)
    yield service
    # Teardown: Reset the client after the test has used the service
    if hasattr(service, 'client') and service.client:
        service.client.reset()

@pytest.fixture
def test_session():
    """Creates a test session with some events."""
    session = Session(
        app_name="test_app",
        user_id="test_user",
        id="test_session_1"
    )
    
    # Add some test events
    events = [
        Event(
            id="event1",
            invocation_id="inv1",
            author="user",
            timestamp=1.0,
            content=types.Content(
                parts=[types.Part(text="Hello, this is a test message about Python programming")]
            )
        ),
        Event(
            id="event2",
            invocation_id="inv2",
            author="assistant",
            timestamp=2.0,
            content=types.Content(
                parts=[types.Part(text="I can help you with Python programming questions")]
            )
        ),
        Event(
            id="event3",
            invocation_id="inv3",
            author="user",
            timestamp=3.0,
            content=types.Content(
                parts=[types.Part(text="What are the best practices for memory management in Python?")]
            )
        )
    ]
    
    session.events = events
    return session

@pytest.mark.asyncio
async def test_add_session_to_memory(memory_service, test_session):
    """Test adding a session to memory."""
    await memory_service.add_session_to_memory(test_session)
    
    # Verify the session was added by searching for it
    response = await memory_service.search_memory(
        app_name="test_app",
        user_id="test_user",
        query="Python programming"
    )
    
    assert len(response.memories) == 1
    assert response.memories[0].session_id == "test_session_1"
    assert len(response.memories[0].events) == 3  # Should find 3 events with "Python programming"

@pytest.mark.asyncio
async def test_search_memory_empty(memory_service):
    """Test searching memory when no sessions are added."""
    response = await memory_service.search_memory(
        app_name="test_app",
        user_id="test_user",
        query="test query"
    )
    
    assert len(response.memories) == 0

@pytest.mark.asyncio
async def test_search_memory_keywords(memory_service, test_session):
    """Test searching memory using keywords."""
    await memory_service.add_session_to_memory(test_session)
    
    # Test exact keyword match
    response = await memory_service.search_memory(
        app_name="test_app",
        user_id="test_user",
        query="memory management"
    )
    
    assert len(response.memories) == 1
    assert len(response.memories[0].events) == 1
    assert "memory management" in response.memories[0].events[0].content.parts[0].text.lower()

@pytest.mark.asyncio
async def test_search_memory_semantic(memory_service, test_session):
    """Test semantic search functionality."""
    await memory_service.add_session_to_memory(test_session)
    
    # Test semantic search with related terms
    response = await memory_service.search_memory(
        app_name="test_app",
        user_id="test_user",
        query="coding in Python"
    )
    
    assert len(response.memories) == 1
    assert len(response.memories[0].events) > 0

@pytest.mark.asyncio
async def test_search_memory_multiple_sessions(memory_service):
    """Test searching across multiple sessions."""
    # Create two sessions
    session1 = Session(
        app_name="test_app",
        user_id="test_user",
        id="test_session_1"
    )
    session1.events = [
        Event(
            id="event1",
            invocation_id="inv1",
            author="user",
            timestamp=1.0,
            content=types.Content(
                parts=[types.Part(text="First session about Python")]
            )
        )
    ]
    
    session2 = Session(
        app_name="test_app",
        user_id="test_user",
        id="test_session_2"
    )
    session2.events = [
        Event(
            id="event2",
            invocation_id="inv2",
            author="user",
            timestamp=1.0,
            content=types.Content(
                parts=[types.Part(text="Second session about Python")]
            )
        )
    ]
    
    await memory_service.add_session_to_memory(session1)
    await memory_service.add_session_to_memory(session2)
    
    response = await memory_service.search_memory(
        app_name="test_app",
        user_id="test_user",
        query="Python"
    )
    
    assert len(response.memories) == 2
    session_ids = {memory.session_id for memory in response.memories}
    assert session_ids == {"test_session_1", "test_session_2"}

@pytest.mark.asyncio
async def test_search_memory_top_k(memory_service):
    """Test that top_k parameter limits the number of results."""
    # Create a session with many events
    session = Session(
        app_name="test_app",
        user_id="test_user",
        id="test_session_1"
    )
    
    # Create 10 events with the same keyword
    session.events = [
        Event(
            id=f"event{i}",
            invocation_id=f"inv{i}",
            author="user",
            timestamp=float(i),
            content=types.Content(
                parts=[types.Part(text=f"Test event {i} about Python")]
            )
        )
        for i in range(10)
    ]
    
    await memory_service.add_session_to_memory(session)
    
    # Search with top_k=5
    response = await memory_service.search_memory(
        app_name="test_app",
        user_id="test_user",
        query="Python"
    )
    
    # Should only return 5 events due to top_k parameter
    assert len(response.memories) == 1
    assert len(response.memories[0].events) <= 5

@pytest.mark.asyncio
async def test_search_memory_event_metadata(memory_service, test_session):
    """Test that event metadata is preserved in search results."""
    await memory_service.add_session_to_memory(test_session)
    
    response = await memory_service.search_memory(
        app_name="test_app",
        user_id="test_user",
        query="Python"
    )
    
    assert len(response.memories) == 1
    event = response.memories[0].events[0]
    
    # Verify metadata is preserved
    assert event.id in ["event1", "event2"]  # One of the events with "Python"
    assert event.invocation_id.startswith("inv")
    assert event.author in ["user", "assistant"]
    assert isinstance(event.timestamp, float)
    assert event.content is not None
    assert len(event.content.parts) == 1
    assert "python" in event.content.parts[0].text 
