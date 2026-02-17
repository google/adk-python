# Copyright 2026 Google LLC
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

"""Tests for CachePerformanceAnalyzer."""

import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from google.adk.events.event import Event
from google.adk.models.cache_metadata import CacheMetadata
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.session import Session
from google.adk.utils.cache_performance_analyzer import CachePerformanceAnalyzer
from google.genai import types
import pytest


class TestCachePerformanceAnalyzer:
  """Test suite for CachePerformanceAnalyzer."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_session_service = MagicMock(spec=BaseSessionService)
    self.analyzer = CachePerformanceAnalyzer(self.mock_session_service)

  def create_cache_metadata(
      self, invocations_used=1, cache_name="test-cache", contents_count=5
  ):
    """Helper to create test CacheMetadata."""
    return CacheMetadata(
        cache_name=(
            f"projects/test/locations/us-central1/cachedContents/{cache_name}"
        ),
        expire_time=time.time() + 1800,
        fingerprint="test_fingerprint",
        invocations_used=invocations_used,
        contents_count=contents_count,
        created_at=time.time() - 600,
    )

  def create_mock_usage_metadata(
      self, prompt_tokens=1000, cached_tokens=500, candidates_tokens=100
  ):
    """Helper to create mock usage metadata."""
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt_tokens,
        cached_content_token_count=cached_tokens,
        candidates_token_count=candidates_tokens,
        total_token_count=prompt_tokens + candidates_tokens,
    )

  def create_mock_event(
      self, author="test_agent", cache_metadata=None, usage_metadata=None
  ):
    """Helper to create mock event."""
    event = Event(author=author, cache_metadata=cache_metadata)
    if usage_metadata:
      event.usage_metadata = usage_metadata
    return event

  def test_init(self):
    """Test analyzer initialization."""
    assert self.analyzer.session_service == self.mock_session_service

  async def test_get_agent_cache_history_empty_session(self):
    """Test getting cache history from empty session."""
    mock_session = Session(
        id="test_session",
        app_name="test_app",
        user_id="test_user",
        events=[],
    )
    self.mock_session_service.get_session = AsyncMock(return_value=mock_session)

    result = await self.analyzer._get_agent_cache_history(
        "test_session", "test_user", "test_app", "test_agent"
    )

    assert result == []

  async def test_get_agent_cache_history_no_cache_events(self):
    """Test getting cache history when no events have cache metadata."""
    events = [
        self.create_mock_event(author="test_agent"),
        self.create_mock_event(author="other_agent"),
        self.create_mock_event(author="test_agent"),
    ]

    mock_session = Session(
        id="test_session",
        app_name="test_app",
        user_id="test_user",
        events=events,
    )
    self.mock_session_service.get_session = AsyncMock(return_value=mock_session)

    result = await self.analyzer._get_agent_cache_history(
        "test_session", "test_user", "test_app", "test_agent"
    )

    assert result == []

  async def test_get_agent_cache_history_specific_agent(self):
    """Test getting cache history for specific agent."""
    cache1 = self.create_cache_metadata(invocations_used=1, cache_name="cache1")
    cache2 = self.create_cache_metadata(invocations_used=3, cache_name="cache2")
    cache3 = self.create_cache_metadata(invocations_used=5, cache_name="cache3")

    events = [
        self.create_mock_event(author="test_agent", cache_metadata=cache1),
        self.create_mock_event(author="other_agent", cache_metadata=cache2),
        self.create_mock_event(author="test_agent", cache_metadata=cache3),
        self.create_mock_event(author="test_agent"),  # No cache metadata
    ]

    mock_session = Session(
        id="test_session",
        app_name="test_app",
        user_id="test_user",
        events=events,
    )
    self.mock_session_service.get_session = AsyncMock(return_value=mock_session)

    result = await self.analyzer._get_agent_cache_history(
        "test_session", "test_user", "test_app", "test_agent"
    )

    # Should only return cache metadata for test_agent
    assert len(result) == 2
    assert result[0] == cache1
    assert result[1] == cache3

  async def test_get_agent_cache_history_all_agents(self):
    """Test getting cache history for all agents."""
    cache1 = self.create_cache_metadata(invocations_used=1, cache_name="cache1")
    cache2 = self.create_cache_metadata(invocations_used=3, cache_name="cache2")

    events = [
        self.create_mock_event(author="agent1", cache_metadata=cache1),
        self.create_mock_event(author="agent2", cache_metadata=cache2),
        self.create_mock_event(author="agent1"),  # No cache metadata
    ]

    mock_session = Session(
        id="test_session",
        app_name="test_app",
        user_id="test_user",
        events=events,
    )
    self.mock_session_service.get_session = AsyncMock(return_value=mock_session)

    # Pass None for agent_name to get all agents
    result = await self.analyzer._get_agent_cache_history(
        "test_session", "test_user", "test_app", None
    )

    # Should return cache metadata for all agents
    assert len(result) == 2
    assert result[0] == cache1
    assert result[1] == cache2

  async def test_analyze_agent_cache_performance_no_cache_data(self):
    """Test analysis with no cache data."""
    mock_session = Session(
        id="test_session",
        app_name="test_app",
        user_id="test_user",
        events=[],
    )
    self.mock_session_service.get_session = AsyncMock(return_value=mock_session)

    result = await self.analyzer.analyze_agent_cache_performance(
        "test_session", "test_user", "test_app", "test_agent"
    )

    assert result["status"] == "no_cache_data"

  async def test_analyze_agent_cache_performance_with_cache_data(self):
    """Test comprehensive analysis with cache data and token metrics."""
    cache1 = self.create_cache_metadata(invocations_used=2, cache_name="cache1")
    cache2 = self.create_cache_metadata(invocations_used=5, cache_name="cache2")
    cache3 = self.create_cache_metadata(invocations_used=8, cache_name="cache3")

    usage1 = self.create_mock_usage_metadata(
        prompt_tokens=1000, cached_tokens=800
    )
    usage2 = self.create_mock_usage_metadata(
        prompt_tokens=1500, cached_tokens=1200
    )
    usage3 = self.create_mock_usage_metadata(prompt_tokens=800, cached_tokens=0)

    events = [
        self.create_mock_event(
            author="test_agent", cache_metadata=cache1, usage_metadata=usage1
        ),
        self.create_mock_event(author="other_agent", cache_metadata=cache2),
        self.create_mock_event(
            author="test_agent", cache_metadata=cache2, usage_metadata=usage2
        ),
        self.create_mock_event(
            author="test_agent", cache_metadata=cache3, usage_metadata=usage3
        ),
    ]

    mock_session = Session(
        id="test_session",
        app_name="test_app",
        user_id="test_user",
        events=events,
    )
    self.mock_session_service.get_session = AsyncMock(return_value=mock_session)

    result = await self.analyzer.analyze_agent_cache_performance(
        "test_session", "test_user", "test_app", "test_agent"
    )

    # Basic cache metrics
    assert result["status"] == "active"
    assert result["requests_with_cache"] == 3
    assert result["cache_refreshes"] == 3  # 3 unique cache names
    assert result["total_invocations"] == 15  # 2 + 5 + 8

    expected_avg_invocations = (2 + 5 + 8) / 3  # 5.0
    assert result["avg_invocations_used"] == expected_avg_invocations

    # Token metrics
    assert result["total_prompt_tokens"] == 3300  # 1000 + 1500 + 800
    assert result["total_cached_tokens"] == 2000  # 800 + 1200 + 0
    assert result["total_requests"] == 3
    assert (
        result["requests_with_cache_hits"] == 2
    )  # Only first two have cached tokens

    # Calculated metrics
    expected_hit_ratio = (2000 / 3300) * 100  # ~60.6%
    expected_utilization = (2 / 3) * 100  # ~66.7%
    expected_avg_cached = 2000 / 3  # ~666.7

    assert abs(result["cache_hit_ratio_percent"] - expected_hit_ratio) < 0.01
    assert (
        abs(result["cache_utilization_ratio_percent"] - expected_utilization)
        < 0.01
    )
    assert (
        abs(result["avg_cached_tokens_per_request"] - expected_avg_cached)
        < 0.01
    )

  async def test_analyze_agent_cache_performance_single_cache(self):
    """Test analysis with single cache instance."""
    cache = self.create_cache_metadata(
        invocations_used=10, cache_name="single_cache"
    )
    usage = self.create_mock_usage_metadata(
        prompt_tokens=2000, cached_tokens=1500
    )

    events = [
        self.create_mock_event(
            author="test_agent", cache_metadata=cache, usage_metadata=usage
        ),
    ]

    mock_session = Session(
        id="test_session",
        app_name="test_app",
        user_id="test_user",
        events=events,
    )
    self.mock_session_service.get_session = AsyncMock(return_value=mock_session)

    result = await self.analyzer.analyze_agent_cache_performance(
        "test_session", "test_user", "test_app", "test_agent"
    )

    assert result["status"] == "active"
    assert result["requests_with_cache"] == 1
    assert result["avg_invocations_used"] == 10.0
    assert result["cache_refreshes"] == 1
    assert result["total_invocations"] == 10
    assert result["latest_cache"] == cache.cache_name

    # Token metrics for single request
    assert result["total_prompt_tokens"] == 2000
    assert result["total_cached_tokens"] == 1500
    assert result["cache_hit_ratio_percent"] == 75.0  # 1500/2000 * 100
    assert result["cache_utilization_ratio_percent"] == 100.0  # 1/1 * 100
    assert result["avg_cached_tokens_per_request"] == 1500.0

  async def test_analyze_agent_cache_performance_no_token_data(self):
    """Test analysis when events have no usage_metadata."""
    cache = self.create_cache_metadata(invocations_used=5)

    events = [
        self.create_mock_event(author="test_agent", cache_metadata=cache),
    ]

    mock_session = Session(
        id="test_session",
        app_name="test_app",
        user_id="test_user",
        events=events,
    )
    self.mock_session_service.get_session = AsyncMock(return_value=mock_session)

    result = await self.analyzer.analyze_agent_cache_performance(
        "test_session", "test_user", "test_app", "test_agent"
    )

    # Should still work but with zero token metrics
    assert result["status"] == "active"
    assert result["requests_with_cache"] == 1
    assert result["total_prompt_tokens"] == 0
    assert result["total_cached_tokens"] == 0
    assert result["cache_hit_ratio_percent"] == 0.0
    assert result["cache_utilization_ratio_percent"] == 0.0
    assert result["avg_cached_tokens_per_request"] == 0.0

  async def test_analyze_agent_cache_performance_zero_invocations(self):
    """Test analysis with zero invocations."""
    cache = self.create_cache_metadata(
        invocations_used=0, cache_name="zero_cache"
    )
    usage = self.create_mock_usage_metadata(
        prompt_tokens=1000, cached_tokens=500
    )

    events = [
        self.create_mock_event(
            author="test_agent", cache_metadata=cache, usage_metadata=usage
        ),
    ]

    mock_session = Session(
        id="test_session",
        app_name="test_app",
        user_id="test_user",
        events=events,
    )
    self.mock_session_service.get_session = AsyncMock(return_value=mock_session)

    result = await self.analyzer.analyze_agent_cache_performance(
        "test_session", "test_user", "test_app", "test_agent"
    )

    assert result["status"] == "active"
    assert result["avg_invocations_used"] == 0.0
    assert result["total_invocations"] == 0

    # Token metrics should still work
    assert result["total_prompt_tokens"] == 1000
    assert result["total_cached_tokens"] == 500

  async def test_session_service_integration(self):
    """Test integration with session service."""
    cache_metadata = self.create_cache_metadata(invocations_used=7)

    events = [
        self.create_mock_event(
            author="integration_agent", cache_metadata=cache_metadata
        ),
    ]

    mock_session = Session(
        id="integration_session",
        app_name="integration_app",
        user_id="integration_user",
        events=events,
    )

    # Configure the mock to return the session
    self.mock_session_service.get_session = AsyncMock(return_value=mock_session)

    result = await self.analyzer.analyze_agent_cache_performance(
        "integration_session",
        "integration_user",
        "integration_app",
        "integration_agent",
    )

    # Verify the session service was called with correct parameters (twice internally)
    assert self.mock_session_service.get_session.call_count == 2
    self.mock_session_service.get_session.assert_called_with(
        session_id="integration_session",
        app_name="integration_app",
        user_id="integration_user",
    )

    assert result["status"] == "active"
    assert result["requests_with_cache"] == 1

  async def test_mixed_agents_filtering(self):
    """Test that analysis correctly filters by agent name."""
    target_cache = self.create_cache_metadata(
        invocations_used=3, cache_name="target"
    )
    other_cache = self.create_cache_metadata(
        invocations_used=5, cache_name="other"
    )

    target_usage = self.create_mock_usage_metadata(
        prompt_tokens=1000, cached_tokens=800
    )
    other_usage = self.create_mock_usage_metadata(
        prompt_tokens=2000, cached_tokens=1600
    )

    events = [
        self.create_mock_event(
            author="target_agent",
            cache_metadata=target_cache,
            usage_metadata=target_usage,
        ),
        self.create_mock_event(
            author="other_agent",
            cache_metadata=other_cache,
            usage_metadata=other_usage,
        ),
        self.create_mock_event(author="target_agent"),  # No cache data
    ]

    mock_session = Session(
        id="test_session",
        app_name="test_app",
        user_id="test_user",
        events=events,
    )
    self.mock_session_service.get_session = AsyncMock(return_value=mock_session)

    result = await self.analyzer.analyze_agent_cache_performance(
        "test_session", "test_user", "test_app", "target_agent"
    )

    # Should only include target_agent's data
    assert result["requests_with_cache"] == 1
    assert result["total_invocations"] == 3
    assert result["total_prompt_tokens"] == 1000  # Only target_agent's tokens
    assert result["total_cached_tokens"] == 800  # Only target_agent's tokens
