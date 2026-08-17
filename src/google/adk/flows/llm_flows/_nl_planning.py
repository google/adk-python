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

"""Handles NL planning related logic."""

from __future__ import annotations

from typing import AsyncGenerator
from typing import Optional
from typing import TYPE_CHECKING

from typing_extensions import override

from ...agents.callback_context import CallbackContext
from ...agents.invocation_context import InvocationContext
from ...agents.readonly_context import ReadonlyContext
from ...events.event import Event
from ...planners.plan_re_act_planner import PlanReActPlanner
from ._base_llm_processor import BaseLlmRequestProcessor
from ._base_llm_processor import BaseLlmResponseProcessor
from ._invocation_utils import as_llm_agent
from ._invocation_utils import require_agent_name
from .contents import _is_part_invisible

if TYPE_CHECKING:
  from ...models.llm_request import LlmRequest
  from ...models.llm_response import LlmResponse
  from ...planners.base_planner import BasePlanner


class _NlPlanningRequestProcessor(BaseLlmRequestProcessor):
  """Processor for NL planning."""

  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...planners.built_in_planner import BuiltInPlanner

    planner = _get_planner(invocation_context)

    if planner and not isinstance(planner, BuiltInPlanner):
      # This planner marks its own reasoning as a thought so a caller can hide
      # it, and needs that reasoning kept in context, so the marker is cleared
      # rather than the part dropped.
      if planning_instruction := planner.build_planning_instruction(
          ReadonlyContext(invocation_context), llm_request
      ):
        llm_request.append_instructions([planning_instruction])

      _remove_thought_from_request(llm_request)
    else:
      if isinstance(planner, BuiltInPlanner):
        planner.apply_thinking_config(llm_request)

      _drop_display_thoughts_from_request(llm_request)

    # Maintain async generator behavior
    if False:  # Ensures it behaves as a generator
      yield  # This is a no-op but maintains generator structure


request_processor = _NlPlanningRequestProcessor()


class _NlPlanningResponse(BaseLlmResponseProcessor):

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_response: LlmResponse
  ) -> AsyncGenerator[Event, None]:
    from ...planners.built_in_planner import BuiltInPlanner

    if (
        not llm_response
        or not llm_response.content
        or not llm_response.content.parts
    ):
      return

    planner = _get_planner(invocation_context)
    if (
        not planner
        or type(planner).process_planning_response
        is BuiltInPlanner.process_planning_response
    ):
      return

    # Postprocess the LLM response.
    callback_context = CallbackContext(invocation_context)
    processed_parts = planner.process_planning_response(
        callback_context, llm_response.content.parts
    )
    if processed_parts:
      llm_response.content.parts = processed_parts

    if callback_context.state.has_delta():
      state_update_event = Event(
          invocation_id=invocation_context.invocation_id,
          author=require_agent_name(invocation_context),
          branch=invocation_context.branch,
          actions=callback_context._event_actions,
      )
      yield state_update_event


response_processor = _NlPlanningResponse()


def _get_planner(
    invocation_context: InvocationContext,
) -> Optional[BasePlanner]:
  from ...planners.base_planner import BasePlanner

  agent = as_llm_agent(invocation_context)
  if not hasattr(agent, 'planner') or not agent.planner:
    return None

  if isinstance(agent.planner, BasePlanner):
    return agent.planner
  return PlanReActPlanner()


def _remove_thought_from_request(llm_request: LlmRequest) -> None:
  if not llm_request.contents:
    return

  for content in llm_request.contents:
    if not content.parts:
      continue
    for part in content.parts:
      part.thought = None


def _drop_display_thoughts_from_request(llm_request: LlmRequest) -> None:
  """Drops the thought parts the model returned only for display.

  A thought summary is something the caller shows a user, not context the model
  needs handed back. History is re-sent whole on every call, so a summary left
  in the request is billed again on each remaining call of the session.

  Which thoughts are safe to drop is already decided by ``_is_part_invisible``,
  so this reuses it. A part that carries a thought signature, a function call or
  response, or a server-side tool call stays, because the model expects those
  back. A content whose parts are all thoughts is left alone rather than
  emptied, since an empty content is not a valid turn.
  """
  for content in llm_request.contents or []:
    if not content.parts:
      continue
    kept = [
        part
        for part in content.parts
        if not (part.thought and _is_part_invisible(part))
    ]
    if kept and len(kept) != len(content.parts):
      content.parts = kept
