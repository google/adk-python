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

"""Tests for ContextCacheRequestProcessor."""

import time
from unittest.mock import MagicMock

from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.context_cache_processor import ContextCacheRequestProcessor
from google.adk.models.cache_metadata import CacheMetadata
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.session import Session
from google.genai import types
import pytest


class TestContextCacheRequestProcessor:
  """Test suite for ContextCacheRequestProcessor."""

  def setup_method(self):
    """Set up test fixtures."""
    self.processor = ContextCacheRequestProcessor()
    self.cache_config = ContextCacheConfig(
        cache_intervals=10, ttl_seconds=1800, min_tokens=1024
    )

  def create_invocation_context(
      self,
      agent,
      context_cache_config=None,
      session_events=None,
      invocation_id="test_invocation",
  ):
    """Helper to create InvocationContext."""
    mock_session = Session(
        id="test_session",
        app_name="test_app",
        user_id="test_user",
        events=session_events or [],
    )

    mock_session_service = MagicMock(spec=BaseSessionService)

    return InvocationContext(
        agent=agent,
        session=mock_session,
        session_service=mock_session_service,
        context_cache_config=context_cache_config,
        invocation_id=invocation_id,
    )

  def create_cache_metadata(
      self, invocations_used=1, cache_name="test-cache", cached_contents_count=3
  ):
    """Helper to create CacheMetadata."""
    return CacheMetadata(
        cache_name=(
            f"projects/test/locations/us-central1/cachedContents/{cache_name}"
        ),
        expire_time=time.time() + 1800,
        fingerprint="test_fingerprint",
        invocations_used=invocations_used,
        cached_contents_count=cached_contents_count,
        created_at=time.time() - 600,
    )

  async def test_no_cache_config(self):
    """Test processor with no cache config."""
    agent = LlmAgent(name="test_agent")
    invocation_context = self.create_invocation_context(
        agent, context_cache_config=None
    )

    llm_request = LlmRequest(
        model="gemini-2.0-flash",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="Hello")],
            )
        ],
    )

    # Process should complete without adding cache config
    events = []
    async for event in self.processor.run_async(
        invocation_context, llm_request
    ):
      events.append(event)

    assert len(events) == 0  # No events yielded
    assert llm_request.cache_config is None

  async def test_with_cache_config_no_session_events(self):
    """Test processor with cache config but no session events."""
    agent = LlmAgent(name="test_agent")
    invocation_context = self.create_invocation_context(
        agent, context_cache_config=self.cache_config
    )

    llm_request = LlmRequest(
        model="gemini-2.0-flash",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="Hello")],
            )
        ],
    )

    # Process should add cache config but no metadata
    events = []
    async for event in self.processor.run_async(
        invocation_context, llm_request
    ):
      events.append(event)

    assert len(events) == 0  # No events yielded
    assert llm_request.cache_config == self.cache_config
    assert llm_request.cache_metadata is None

  async def test_with_cache_metadata_same_invocation(self):
    """Test processor finds cache metadata from same invocation."""
    agent = LlmAgent(name="test_agent")
    cache_metadata = self.create_cache_metadata(invocations_used=5)

    # Event with same invocation ID
    events = [
        Event(
            author="test_agent",
            cache_metadata=cache_metadata,
            invocation_id="test_invocation",
        )
    ]

    invocation_context = self.create_invocation_context(
        agent,
        context_cache_config=self.cache_config,
        session_events=events,
        invocation_id="test_invocation",
    )

    llm_request = LlmRequest(
        model="gemini-2.0-flash",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="Hello")],
            )
        ],
    )

    # Process should add cache config and metadata (same invocation, no increment)
    async for event in self.processor.run_async(
        invocation_context, llm_request
    ):
      pass

    assert llm_request.cache_config == self.cache_config
    assert llm_request.cache_metadata == cache_metadata
    assert llm_request.cache_metadata.invocations_used == 5  # No increment

  async def test_with_cache_metadata_different_invocation(self):
    """Test processor finds cache metadata from different invocation."""
    agent = LlmAgent(name="test_agent")
    cache_metadata = self.create_cache_metadata(invocations_used=5)

    # Event with different invocation ID
    events = [
        Event(
            author="test_agent",
            cache_metadata=cache_metadata,
            invocation_id="previous_invocation",
        )
    ]

    invocation_context = self.create_invocation_context(
        agent,
        context_cache_config=self.cache_config,
        session_events=events,
        invocation_id="current_invocation",
    )

    llm_request = LlmRequest(
        model="gemini-2.0-flash",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="Hello")],
            )
        ],
    )

    # Process should add cache config and increment invocations_used
    async for event in self.processor.run_async(
        invocation_context, llm_request
    ):
      pass

    assert llm_request.cache_config == self.cache_config
    assert llm_request.cache_metadata is not None
    assert llm_request.cache_metadata.invocations_used == 6  # Incremented

  async def test_cache_metadata_agent_filtering(self):
    """Test that cache metadata is filtered by agent name."""
    agent = LlmAgent(name="target_agent")
    target_cache = self.create_cache_metadata(
        invocations_used=3, cache_name="target"
    )
    other_cache = self.create_cache_metadata(
        invocations_used=7, cache_name="other"
    )

    events = [
        Event(
            author="other_agent",
            cache_metadata=other_cache,
            invocation_id="other_invocation",
        ),
        Event(
            author="target_agent",
            cache_metadata=target_cache,
            invocation_id="target_invocation",
        ),
    ]

    invocation_context = self.create_invocation_context(
        agent,
        context_cache_config=self.cache_config,
        session_events=events,
        invocation_id="current_invocation",
    )

    llm_request = LlmRequest(
        model="gemini-2.0-flash",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="Hello")],
            )
        ],
    )

    # Should only use target_agent's cache metadata
    async for event in self.processor.run_async(
        invocation_context, llm_request
    ):
      pass

    assert llm_request.cache_metadata is not None
    assert llm_request.cache_metadata.cache_name == target_cache.cache_name
    assert llm_request.cache_metadata.invocations_used == 4  # target_cache + 1

  async def test_latest_cache_metadata_selected(self):
    """Test that the latest cache metadata is selected."""
    agent = LlmAgent(name="test_agent")
    older_cache = self.create_cache_metadata(
        invocations_used=2, cache_name="older"
    )
    newer_cache = self.create_cache_metadata(
        invocations_used=5, cache_name="newer"
    )

    # Events in chronological order (older first)
    events = [
        Event(
            author="test_agent",
            cache_metadata=older_cache,
            invocation_id="older_invocation",
        ),
        Event(
            author="test_agent",
            cache_metadata=newer_cache,
            invocation_id="newer_invocation",
        ),
    ]

    invocation_context = self.create_invocation_context(
        agent,
        context_cache_config=self.cache_config,
        session_events=events,
        invocation_id="current_invocation",
    )

    llm_request = LlmRequest(
        model="gemini-2.0-flash",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="Hello")],
            )
        ],
    )

    # Should use the newer (latest) cache metadata
    async for event in self.processor.run_async(
        invocation_context, llm_request
    ):
      pass

    assert llm_request.cache_metadata is not None
    assert llm_request.cache_metadata.cache_name == newer_cache.cache_name
    assert llm_request.cache_metadata.invocations_used == 6  # newer_cache + 1

  async def test_no_cache_metadata_events(self):
    """Test when session has events but no cache metadata."""
    agent = LlmAgent(name="test_agent")

    events = [
        Event(author="test_agent", cache_metadata=None),
        Event(author="other_agent", cache_metadata=None),
    ]

    invocation_context = self.create_invocation_context(
        agent,
        context_cache_config=self.cache_config,
        session_events=events,
    )

    llm_request = LlmRequest(
        model="gemini-2.0-flash",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="Hello")],
            )
        ],
    )

    # Should add cache config but no metadata
    async for event in self.processor.run_async(
        invocation_context, llm_request
    ):
      pass

    assert llm_request.cache_config == self.cache_config
    assert llm_request.cache_metadata is None

  async def test_empty_session(self):
    """Test with empty session."""
    agent = LlmAgent(name="test_agent")

    invocation_context = self.create_invocation_context(
        agent,
        context_cache_config=self.cache_config,
        session_events=[],
    )

    llm_request = LlmRequest(
        model="gemini-2.0-flash",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="Hello")],
            )
        ],
    )

    # Should add cache config but no metadata
    async for event in self.processor.run_async(
        invocation_context, llm_request
    ):
      pass

    assert llm_request.cache_config == self.cache_config
    assert llm_request.cache_metadata is None

  async def test_processor_yields_no_events(self):
    """Test that processor yields no events."""
    agent = LlmAgent(name="test_agent")

    invocation_context = self.create_invocation_context(
        agent, context_cache_config=self.cache_config
    )

    llm_request = LlmRequest(
        model="gemini-2.0-flash",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="Hello")],
            )
        ],
    )

    events = []
    async for event in self.processor.run_async(
        invocation_context, llm_request
    ):
      events.append(event)

    # Processor should never yield events
    assert len(events) == 0

  async def test_mixed_events_scenario(self):
    """Test complex scenario with mixed events."""
    agent = LlmAgent(name="test_agent")
    cache_metadata = self.create_cache_metadata(invocations_used=10)

    events = [
        Event(author="other_agent", cache_metadata=None),
        Event(author="test_agent", cache_metadata=None),  # No cache metadata
        Event(
            author="different_agent", cache_metadata=cache_metadata
        ),  # Wrong agent
        Event(
            author="test_agent",
            cache_metadata=cache_metadata,
            invocation_id="prev",
        ),
    ]

    invocation_context = self.create_invocation_context(
        agent,
        context_cache_config=self.cache_config,
        session_events=events,
        invocation_id="current",
    )

    llm_request = LlmRequest(
        model="gemini-2.0-flash",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="Hello")],
            )
        ],
    )

    async for event in self.processor.run_async(
        invocation_context, llm_request
    ):
      pass

    # Should find the test_agent's cache metadata and increment it
    assert llm_request.cache_config == self.cache_config
    assert llm_request.cache_metadata is not None
    assert llm_request.cache_metadata.invocations_used == 11  # 10 + 1
