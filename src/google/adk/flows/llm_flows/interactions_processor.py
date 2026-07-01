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
"""Interactions API processor for LLM requests."""

from __future__ import annotations

import logging
from typing import AsyncGenerator
from typing import Optional
from typing import TYPE_CHECKING

from ...events.event import Event
from ._base_llm_processor import BaseLlmRequestProcessor

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext
  from ...models.llm_request import LlmRequest
logger = logging.getLogger('google_adk.' + __name__)


def _is_event_in_branch(current_branch: Optional[str], event: Event) -> bool:
  """Return True if ``event`` belongs to ``current_branch`` (or the root)."""
  if not current_branch:
    # No branch means we're at the root; include all events without a branch.
    return not event.branch
  return event.branch == current_branch or not event.branch


def _find_previous_interaction_state(
    events: list[Event],
    *,
    agent_name: str,
    current_branch: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
  """Find the most recent (interaction_id, environment_id) for ``agent_name``.

  Scans ``events`` in reverse, skipping events outside ``current_branch``, and
  returns the ids from the first event authored by this agent that carries an
  interaction_id.
  """
  logger.debug(
      'Finding previous_interaction_id: agent=%s, branch=%s, num_events=%d',
      agent_name,
      current_branch,
      len(events),
  )
  for event in reversed(events):
    if not _is_event_in_branch(current_branch, event):
      logger.debug(
          'Skipping event not in branch: author=%s, branch=%s, current=%s',
          event.author,
          event.branch,
          current_branch,
      )
      continue
    logger.debug(
        'Checking event: author=%s, interaction_id=%s, branch=%s',
        event.author,
        event.interaction_id,
        event.branch,
    )
    if event.author == agent_name and event.interaction_id:
      logger.debug(
          'Found interaction_id from agent %s: %s',
          agent_name,
          event.interaction_id,
      )
      return event.interaction_id, event.environment_id
  return None, None


class InteractionsRequestProcessor(BaseLlmRequestProcessor):
  """Request processor for Interactions API stateful conversations.
  This processor extracts the previous_interaction_id from session events
  to enable stateful conversation chaining via the Interactions API.
  The actual content filtering (retaining only latest user messages) is
  done in the Gemini class when using the Interactions API.
  """

  async def run_async(
      self, invocation_context: 'InvocationContext', llm_request: 'LlmRequest'
  ) -> AsyncGenerator[Event, None]:
    """Process LLM request to extract previous_interaction_id.
    Args:
        invocation_context: Invocation context containing agent and session info
        llm_request: Request to process
    Yields:
        Event: No events are yielded by this processor
    """
    from ...models.google_llm import Gemini

    agent = invocation_context.agent
    # Only process if using Gemini with interactions API
    if not hasattr(agent, 'canonical_model'):
      return
    model = agent.canonical_model
    if not isinstance(model, Gemini):
      return
    if not model.use_interactions_api:
      return
    # Extract previous interaction ID from session events
    previous_interaction_id = self._find_previous_interaction_id(
        invocation_context
    )
    if previous_interaction_id:
      llm_request.previous_interaction_id = previous_interaction_id
      logger.debug(
          'Found previous_interaction_id for interactions API: %s',
          previous_interaction_id,
      )
    # Don't yield any events - this is just a preprocessing step
    return
    yield  # Required for AsyncGenerator

  def _find_previous_interaction_id(
      self, invocation_context: 'InvocationContext'
  ) -> Optional[str]:
    """Find the previous interaction ID from session events."""
    interaction_id, _ = _find_previous_interaction_state(
        invocation_context.session.events,
        agent_name=invocation_context.agent.name,
        current_branch=invocation_context.branch,
    )
    return interaction_id


# Module-level processor instance for use in flow configuration
request_processor = InteractionsRequestProcessor()
