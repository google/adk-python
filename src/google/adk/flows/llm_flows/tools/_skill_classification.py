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

"""Lets a skill toolset load a skill before the model's first step of a turn."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from google.genai import types
from typing_extensions import override

from ....agents.invocation_context import InvocationContext
from ....events.event import Event
from ....models.llm_request import LlmRequest
from .._base_llm_processor import BaseLlmRequestProcessor


def _unanswered_user_message(
    invocation_context: InvocationContext,
) -> types.Content | None:
  """Returns the user's message for this turn if nothing has answered it yet.

  Only the first model step of a turn sees the user's message as the last
  event of its branch. Later steps, a sub-agent the turn was transferred to,
  and a resumed turn all see an event of their own there instead. A user event
  that carries function responses is a tool result or a confirmation, not a
  message.
  """
  events = invocation_context._get_events(current_branch=True)
  if not events:
    return None
  last_event = events[-1]
  if (
      last_event.author != 'user'
      or last_event.invocation_id != invocation_context.invocation_id
      or last_event.content is None
      or not last_event.content.parts
      or last_event.get_function_responses()
  ):
    return None
  return last_event.content


class _SkillClassificationLlmRequestProcessor(BaseLlmRequestProcessor):
  """Gives each `SkillToolset` of the agent a chance to preload a skill.

  A toolset preloads by emitting a `load_skill` call and its response. Runs
  before `contents`, so the request being built already includes them, and
  before the tool processors, so the skill's additional tools are declared in
  that same request.
  """

  name = 'skill_classification'

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ....agents.llm_agent import LlmAgent
    from ....tools.skill_toolset import SkillToolset

    del llm_request  # The events reach it through the `contents` processor.
    agent = invocation_context.agent
    # A live stream is one invocation, with no turn boundary to classify at.
    if (
        not isinstance(agent, LlmAgent)
        or invocation_context.live_request_queue is not None
    ):
      return
    toolsets = [tool for tool in agent.tools if isinstance(tool, SkillToolset)]
    if not toolsets:
      return
    user_content = _unanswered_user_message(invocation_context)
    if user_content is None:
      return
    for toolset in toolsets:
      async for event in toolset._preload_classified_skill(
          invocation_context, user_content
      ):
        yield event


request_processor = _SkillClassificationLlmRequestProcessor()
