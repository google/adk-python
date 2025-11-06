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

"""Cache performance analysis utilities for ADK context caching system.

This module provides tools to analyze cache performance metrics from event
history, including hit ratios, cost savings, and cache refresh patterns.
"""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from google.adk.models.cache_metadata import CacheMetadata
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.utils.feature_decorator import experimental


@experimental
class CachePerformanceAnalyzer:
  """Analyzes cache performance through event history."""

  def __init__(self, session_service: BaseSessionService):
    self.session_service = session_service

  async def _get_agent_cache_history(
      self,
      session_id: str,
      user_id: str,
      app_name: str,
      agent_name: Optional[str] = None,
  ) -> List[CacheMetadata]:
    """Get cache usage history for agent from events.

    Args:
        session_id: Session to analyze
        user_id: User ID for session lookup
        app_name: App name for session lookup
        agent_name: Agent to get history for. If None, gets all cache events.

    Returns:
        List of cache metadata in chronological order
    """
    session = await self.session_service.get_session(
        session_id=session_id,
        app_name=app_name,
        user_id=user_id,
    )
    cache_history = []

    for event in session.events:
      # Check if event has cache metadata and optionally filter by agent
      if event.cache_metadata is not None and (
          agent_name is None or event.author == agent_name
      ):
        cache_history.append(event.cache_metadata)

    return cache_history

  async def analyze_agent_cache_performance(
      self, session_id: str, user_id: str, app_name: str, agent_name: str
  ) -> Dict[str, Any]:
    """Analyze cache performance for agent.

    Args:
        session_id: Session to analyze
        user_id: User ID for session lookup
        app_name: App name for session lookup
        agent_name: Agent to analyze

    Returns:
        Performance analysis dictionary containing:
        - status: "active" if cache data found, "no_cache_data" if none
        - requests_with_cache: Number of requests that used caching
        - avg_invocations_used: Average number of invocations each cache was used
        - latest_cache: Resource name of most recent cache used
        - cache_refreshes: Number of unique cache instances created
        - total_invocations: Total number of invocations across all caches
        - total_prompt_tokens: Total prompt tokens across all requests
        - total_cached_tokens: Total cached content tokens across all requests
        - cache_hit_ratio_percent: Percentage of tokens served from cache
        - cache_utilization_ratio_percent: Percentage of requests with cache hits
        - avg_cached_tokens_per_request: Average cached tokens per request
        - total_requests: Total number of requests processed
        - requests_with_cache_hits: Number of requests that had cache hits
    """
    cache_history = await self._get_agent_cache_history(
        session_id, user_id, app_name, agent_name
    )

    if not cache_history:
      return {"status": "no_cache_data"}

    # Get all events for token analysis
    session = await self.session_service.get_session(
        session_id=session_id,
        app_name=app_name,
        user_id=user_id,
    )

    # Collect token metrics from events
    total_prompt_tokens = 0
    total_cached_tokens = 0
    requests_with_cache_hits = 0
    total_requests = 0

    for event in session.events:
      if event.author == agent_name and event.usage_metadata:
        total_requests += 1
        if event.usage_metadata.prompt_token_count:
          total_prompt_tokens += event.usage_metadata.prompt_token_count
        if event.usage_metadata.cached_content_token_count:
          total_cached_tokens += event.usage_metadata.cached_content_token_count
          requests_with_cache_hits += 1

    # Calculate cache metrics
    cache_hit_ratio_percent = (
        (total_cached_tokens / total_prompt_tokens) * 100
        if total_prompt_tokens > 0
        else 0.0
    )

    cache_utilization_ratio_percent = (
        (requests_with_cache_hits / total_requests) * 100
        if total_requests > 0
        else 0.0
    )

    avg_cached_tokens_per_request = (
        total_cached_tokens / total_requests if total_requests > 0 else 0.0
    )

    invocations_used = [c.invocations_used for c in cache_history]
    total_invocations = sum(invocations_used)

    return {
        "status": "active",
        "requests_with_cache": len(cache_history),
        "avg_invocations_used": (
            sum(invocations_used) / len(invocations_used)
            if invocations_used
            else 0
        ),
        "latest_cache": cache_history[-1].cache_name,
        "cache_refreshes": len(set(c.cache_name for c in cache_history)),
        "total_invocations": total_invocations,
        "total_prompt_tokens": total_prompt_tokens,
        "total_cached_tokens": total_cached_tokens,
        "cache_hit_ratio_percent": cache_hit_ratio_percent,
        "cache_utilization_ratio_percent": cache_utilization_ratio_percent,
        "avg_cached_tokens_per_request": avg_cached_tokens_per_request,
        "total_requests": total_requests,
        "requests_with_cache_hits": requests_with_cache_hits,
    }
