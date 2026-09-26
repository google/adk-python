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

from ....events.event import Event
from .._base_llm_processor import BaseLlmRequestProcessor
from ..core._utils import as_llm_agent
from ..core._utils import require_agent_name

if TYPE_CHECKING:
  from ....agents.invocation_context import InvocationContext
  from ....models.llm_request import LlmRequest

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
  done by the content request processor after this processor runs.
  """

  name = 'interactions'

  _last_tier_warning_invocation_id: Optional[str] = None
  """Invocation that last logged the unusable-tier warning.

  The processor runs once per model call, so warning unconditionally would
  repeat on every turn of a run. Tracking the most recent invocation keeps it
  to once per run without accumulating ids for the life of the process.
  """

  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    """Process LLM request to extract previous_interaction_id.

    Args:
        invocation_context: Invocation context containing agent and session info
        llm_request: Request to process

    Yields:
        Event: No events are yielded by this processor
    """
    from ....models.google_llm import Gemini

    agent = as_llm_agent(invocation_context)
    model = getattr(agent, 'canonical_model', None)

    # Only process if using Gemini with interactions API
    if not isinstance(model, Gemini) or not model.use_interactions_api:
      self._warn_if_service_tier_unusable(invocation_context)
      return

    run_config = invocation_context.run_config
    if run_config and run_config.service_tier:
      llm_request.service_tier = run_config.service_tier
      logger.debug(
          'Using service_tier from run_config: %s', run_config.service_tier
      )

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

  def _warn_if_service_tier_unusable(
      self, invocation_context: InvocationContext
  ) -> None:
    """Warn that this agent's model calls will ignore the run's service tier.

    Only the interactions path carries a serving tier, so a tier set on the
    run does nothing for an agent that does not use it. That is legitimate in
    a multi-agent run where only some agents are on the interactions API,
    which is why this warns rather than raising. It fires once per run: the
    processor runs on every model call, so warning each time would repeat on
    every turn.

    Args:
        invocation_context: Invocation context carrying the run config and the
          agent whose model cannot apply the tier.
    """
    run_config = invocation_context.run_config
    if not run_config or not run_config.service_tier:
      return
    if (
        self._last_tier_warning_invocation_id
        == invocation_context.invocation_id
    ):
      return
    self._last_tier_warning_invocation_id = invocation_context.invocation_id
    logger.warning(
        'run_config.service_tier=%r has no effect for agent %s: its model does'
        ' not use the interactions API, which is the only path with a serving'
        ' tier. Set use_interactions_api=True on the model to apply the tier.',
        run_config.service_tier,
        require_agent_name(invocation_context),
    )

  def _find_previous_interaction_id(
      self, invocation_context: InvocationContext
  ) -> Optional[str]:
    """Find the previous interaction ID from session events."""
    interaction_id, _ = _find_previous_interaction_state(
        invocation_context.session.events,
        agent_name=require_agent_name(invocation_context),
        current_branch=invocation_context.branch,
    )
    return interaction_id


request_processor = InteractionsRequestProcessor()

__all__ = [
    'InteractionsRequestProcessor',
    '_find_previous_interaction_state',
    '_is_event_in_branch',
    'request_processor',
]
