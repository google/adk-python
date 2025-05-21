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
import time
from typing import Optional

from google.adk.memory.in_memory_bm25_retrieval_memory_service import InMemoryBM25RetrievalMemoryService
from google.adk.sessions.session import Session
from google.adk.events.event import Event
from google.genai import types as genai_types

USER_AUTHOR = "user"
MODEL_AUTHOR = "model"


def _create_event(text: str, author: str = USER_AUTHOR, timestamp: Optional[float] = None) -> Event:
    return Event(
        invocation_id="test_invocation",
        author=author,
        content=genai_types.Content(
            role="user" if author == USER_AUTHOR else "model",
            parts=[genai_types.Part(text=text)],
        ),
        timestamp=timestamp if timestamp is not None else time.time(),
    )


@pytest.mark.asyncio
async def test_add_session_to_memory_single_session():
    service = InMemoryBM25RetrievalMemoryService()
    app_name = "test_app"
    user_id = "test_user"
    session_id = "session1"

    event1 = _create_event("Agent 1 added an event")
    event2 = _create_event("Agent 47 added an event")
    session = Session(app_name=app_name, user_id=user_id,
                      id=session_id, state={}, events=[event1, event2])

    await service.add_session_to_memory(session)

    user_key = f"{app_name}/{user_id}"
    assert user_key in service._session_events
    assert session.id in service._session_events[user_key]
    assert len(service._session_events[user_key][session.id]) == 2
    assert service._session_events[user_key][session.id][0].model_dump(
    ) == event1.model_dump()
    assert service._session_events[user_key][session.id][1].model_dump(
    ) == event2.model_dump()


@pytest.mark.asyncio
async def test_add_session_to_memory_multiple_sessions_same_user():
    service = InMemoryBM25RetrievalMemoryService()
    app_name = "test_app"
    user_id = "test_user"
    session_id1 = "session1"
    session_id2 = "session2"

    event1 = _create_event("Travel agent 1 added an event")
    session1 = Session(app_name=app_name, user_id=user_id,
                       id=session_id1, state={}, events=[event1])
    await service.add_session_to_memory(session1)

    event2 = _create_event("Booking confirmed")
    session2 = Session(app_name=app_name, user_id=user_id,
                       id=session_id2, state={}, events=[event2])
    await service.add_session_to_memory(session2)

    user_key = f"{app_name}/{user_id}"
    assert user_key in service._session_events
    assert session1.id in service._session_events[user_key]
    assert session2.id in service._session_events[user_key]
    assert service._session_events[user_key][session1.id][0].model_dump(
    ) == event1.model_dump()
    assert service._session_events[user_key][session2.id][0].model_dump(
    ) == event2.model_dump()


@pytest.mark.asyncio
async def test_add_session_to_memory_different_users():
    service = InMemoryBM25RetrievalMemoryService()
    app_name = "test_app"
    user_id1 = "user1"
    user_id2 = "user2"
    session_id1 = "session1"
    session_id2 = "session2"

    event1 = _create_event("User1 event")
    session1 = Session(app_name=app_name, user_id=user_id1,
                       id=session_id1, state={}, events=[event1])
    await service.add_session_to_memory(session1)

    event2 = _create_event("User2 event")
    session2 = Session(app_name=app_name, user_id=user_id2,
                       id=session_id2, state={}, events=[event2])
    await service.add_session_to_memory(session2)

    user_key1 = f"{app_name}/{user_id1}"
    user_key2 = f"{app_name}/{user_id2}"

    assert user_key1 in service._session_events
    assert service._session_events[user_key1][session1.id][0].model_dump(
    ) == event1.model_dump()
    assert user_key2 in service._session_events
    assert service._session_events[user_key2][session2.id][0].model_dump(
    ) == event2.model_dump()


@pytest.mark.asyncio
async def test_add_session_to_memory_event_filtering():
    service = InMemoryBM25RetrievalMemoryService()
    app_name = "test_app"
    user_id = "test_user"
    session_id = "session1"

    event_valid = _create_event("Arrived at NYC")
    event_no_content = Event(invocation_id="inv1",
                             author=USER_AUTHOR, content=None)
    event_no_parts = Event(invocation_id="inv2", author=USER_AUTHOR,
                           content=genai_types.Content(role="user", parts=[]))
    event_empty_text_part = Event(
        invocation_id="inv3",
        author=USER_AUTHOR,
        content=genai_types.Content(
            role="user", parts=[genai_types.Part(text="")]),
    )

    session = Session(
        app_name=app_name,
        user_id=user_id,
        id=session_id,
        state={},
        events=[event_valid, event_no_content,
                event_no_parts, event_empty_text_part],
    )
    await service.add_session_to_memory(session)

    user_key = f"{app_name}/{user_id}"
    assert len(service._session_events[user_key][session.id]) == 1
    assert service._session_events[user_key][session.id][0].model_dump(
    ) == event_valid.model_dump()


@pytest.mark.asyncio
async def test_search_memory_no_sessions_for_user():
    service = InMemoryBM25RetrievalMemoryService()
    app_name = "test_app"
    user_id = "non_existent_user"
    query = "anything"

    response = await service.search_memory(app_name=app_name, user_id=user_id, query=query)
    assert not response.memories


@pytest.mark.asyncio
async def test_search_memory_bm25_no_match():
    service = InMemoryBM25RetrievalMemoryService()
    app_name = "test_app"
    user_id = "test_user"
    session_id = "session1"

    event1 = _create_event("Entered the matrix")
    session = Session(app_name=app_name, user_id=user_id,
                      id=session_id, state={}, events=[event1])
    await service.add_session_to_memory(session)

    response = await service.search_memory(app_name=app_name, user_id=user_id, query="goodbye")
    assert not response.memories


@pytest.mark.asyncio
async def test_search_memory_bm25_match_simple():
    service = InMemoryBM25RetrievalMemoryService()
    app_name = "test_app"
    user_id = "test_user"
    session_id = "session1"

    event1_text = "google IO event has started"
    event1 = _create_event(event1_text, timestamp=1.0)
    event2_text = "new features released"
    event2 = _create_event(event2_text, timestamp=2.0)

    session = Session(app_name=app_name, user_id=user_id,
                      id=session_id, state={}, events=[event1, event2])
    await service.add_session_to_memory(session)

    response = await service.search_memory(app_name=app_name, user_id=user_id, query="google IO")
    assert len(response.memories) == 1
    assert response.memories[0].content.parts[0].text == event1_text


@pytest.mark.asyncio
async def test_search_memory_bm25_ranking():
    service = InMemoryBM25RetrievalMemoryService()
    app_name = "test_app"
    user_id = "test_user"
    session_id = "session1"

    event1_text = "Veo3 has been launched"
    event1 = _create_event(event1_text, timestamp=1.0)
    event2_text = "Awesome video has been added"
    event2 = _create_event(event2_text, timestamp=2.0)
    event3_text = "Video has been added to the list"
    event3 = _create_event(event3_text, timestamp=3.0)

    session = Session(
        app_name=app_name,
        user_id=user_id,
        id=session_id,
        state={},
        events=[event1, event2, event3],
    )
    await service.add_session_to_memory(session)

    response = await service.search_memory(app_name=app_name, user_id=user_id, query="awesome video")
    assert len(response.memories) > 0
    assert response.memories[0].content.parts[0].text == event2_text


@pytest.mark.asyncio
async def test_search_memory_bm25_max_results():
    service = InMemoryBM25RetrievalMemoryService()
    app_name = "test_app"
    user_id = "test_user"
    session_id = "session1"

    events_text = [
        "Relevant document one",
        "Relevant document two",
        "Relevant document three",
        "Slightly less relevant",
        "Irrelevant document",
    ]
    events = [_create_event(t, timestamp=float(i))
              for i, t in enumerate(events_text)]
    session = Session(app_name=app_name, user_id=user_id,
                      id=session_id, state={}, events=events)
    await service.add_session_to_memory(session)

    response = await service.search_memory(app_name=app_name, user_id=user_id, query="Relevant document", max_results=2)
    assert len(response.memories) == 2

    response_default_max = await service.search_memory(app_name=app_name, user_id=user_id, query="Relevant document")
    assert len(response_default_max.memories) == 3


@pytest.mark.asyncio
async def test_bm25_cache_invalidation_and_rebuild():
    service = InMemoryBM25RetrievalMemoryService()
    app_name = "test_app"
    user_id = "test_user"
    session_id1 = "s1"
    session_id2 = "s2"
    user_key = f"{app_name}/{user_id}"

    event1_text = "Initial document content"
    event1 = _create_event(event1_text, timestamp=1.0)
    session1 = Session(app_name=app_name, user_id=user_id,
                       id=session_id1, state={}, events=[event1])
    await service.add_session_to_memory(session1)

    await service.search_memory(app_name=app_name, user_id=user_id, query="document")
    assert user_key in service._bm25_cache
    bm25_instance_before, events_before = service._bm25_cache[user_key]
    assert len(events_before) == 1

    event2_text = "Newly added document"
    event2 = _create_event(event2_text, timestamp=2.0)
    session2 = Session(app_name=app_name, user_id=user_id,
                       id=session_id2, state={}, events=[event2])
    await service.add_session_to_memory(session2)
    assert user_key not in service._bm25_cache

    response = await service.search_memory(app_name=app_name, user_id=user_id, query="document")
    assert user_key in service._bm25_cache
    bm25_instance_after, events_after = service._bm25_cache[user_key]
    assert len(events_after) == 2
    assert bm25_instance_after is not bm25_instance_before

    found_texts = {m.content.parts[0].text for m in response.memories}
    assert event1_text in found_texts
    assert event2_text in found_texts


@pytest.mark.asyncio
async def test_search_memory_empty_query():
    service = InMemoryBM25RetrievalMemoryService()
    app_name = "test_app"
    user_id = "test_user"
    session_id = "s1"
    event1 = _create_event("NY would be good in summer", timestamp=1.0)
    session1 = Session(app_name=app_name, user_id=user_id,
                       id=session_id, events=[event1])
    await service.add_session_to_memory(session1)

    response_kw = await service.search_memory(app_name=app_name, user_id=user_id, query="")
    assert not response_kw.memories

    service_bm25 = InMemoryBM25RetrievalMemoryService()
    await service_bm25.add_session_to_memory(session1)
    response_bm25 = await service_bm25.search_memory(app_name=app_name, user_id=user_id, query="")
    assert not response_bm25.memories
