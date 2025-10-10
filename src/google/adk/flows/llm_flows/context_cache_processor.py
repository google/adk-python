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

"""Context cache processor for LLM requests."""

from __future__ import annotations

import logging
from typing import AsyncGenerator
from typing import Optional
from typing import TYPE_CHECKING

from ...events.event import Event
from ...models.cache_metadata import CacheMetadata
from ._base_llm_processor import BaseLlmRequestProcessor

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext
  from ...models.llm_request import LlmRequest

logger = logging.getLogger('google_adk.' + __name__)


class ContextCacheRequestProcessor(BaseLlmRequestProcessor):
  """Request processor that enables context caching for LLM requests.

  This processor sets up context caching configuration for agents that have
  context caching enabled and finds the latest cache metadata from session
  events. The actual cache management is handled by the model-specific cache
  managers (e.g., GeminiContextCacheManager).
  """

  async def run_async(
      self, invocation_context: 'InvocationContext', llm_request: 'LlmRequest'
  ) -> AsyncGenerator[Event, None]:
    """Process LLM request to enable context caching.

    Args:
        invocation_context: Invocation context containing agent and session info
        llm_request: Request to process for caching

    Yields:
        Event: No events are yielded by this processor
    """
    agent = invocation_context.agent

    # Return early if no cache config
    if not invocation_context.context_cache_config:
      return

    # Set cache config to request
    llm_request.cache_config = invocation_context.context_cache_config

    # Find latest cache metadata and previous token count from session events
    latest_cache_metadata, previous_token_count = (
        self._find_cache_info_from_events(
            invocation_context, agent.name, invocation_context.invocation_id
        )
    )

    if latest_cache_metadata:
      llm_request.cache_metadata = latest_cache_metadata
      logger.debug(
          'Found cache metadata for agent %s: %s',
          agent.name,
          latest_cache_metadata,
      )

    if previous_token_count is not None:
      llm_request.cacheable_contents_token_count = previous_token_count
      logger.debug(
          'Found previous prompt token count for agent %s: %d',
          agent.name,
          previous_token_count,
      )

    logger.debug('Context caching enabled for agent %s', agent.name)

    # This processor yields no events
    return
    yield  # AsyncGenerator requires a yield in function body

  def _find_cache_info_from_events(
      self,
      invocation_context: 'InvocationContext',
      agent_name: str,
      current_invocation_id: str,
  ) -> tuple[Optional[CacheMetadata], Optional[int]]:
    """Find cache metadata and previous token count from session events.

    Args:
        invocation_context: Context containing session with events
        agent_name: Name of agent to find cache info for
        current_invocation_id: Current invocation ID to compare for increment

    Returns:
        Tuple of (cache_metadata, previous_prompt_token_count)
        cache_metadata: Latest cache metadata with invocations_used incremented
            only if this is a different invocation and has active cache
        previous_prompt_token_count: Most recent prompt token count from
            LLM response
    """
    if not invocation_context.session or not invocation_context.session.events:
      return None, None

    cache_metadata = None
    previous_token_count = None

    # Search events from most recent to oldest using index traversal
    events = invocation_context.session.events
    for i in range(len(events) - 1, -1, -1):
      event = events[i]
      if event.author != agent_name:
        continue

      # Look for cache metadata (only in actual LLM response events)
      if cache_metadata is None and event.cache_metadata is not None:
        # Check if this is a different invocation and has active cache
        if (
            event.invocation_id
            and event.invocation_id != current_invocation_id
            and event.cache_metadata.cache_name is not None
        ):
          # Different invocation with active cache - increment invocations_used
          cache_metadata = event.cache_metadata.model_copy(
              update={
                  'invocations_used': event.cache_metadata.invocations_used + 1
              }
          )
        else:
          # Same invocation or no active cache - return copy as-is
          cache_metadata = event.cache_metadata.model_copy()

      # Look for previous prompt token count (from actual LLM response events)
      if (
          previous_token_count is None
          and event.usage_metadata
          and event.usage_metadata.prompt_token_count is not None
      ):
        previous_token_count = event.usage_metadata.prompt_token_count

      # Stop early if we found both pieces of information
      if cache_metadata is not None and previous_token_count is not None:
        break

    return cache_metadata, previous_token_count


# Create processor instance for use in flows
request_processor = ContextCacheRequestProcessor()
