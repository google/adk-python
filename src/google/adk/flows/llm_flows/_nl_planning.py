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
    if not planner:
      return

    if isinstance(planner, BuiltInPlanner):
      planner.apply_thinking_config(llm_request)
    elif isinstance(planner, PlanReActPlanner):
      if planning_instruction := planner.build_planning_instruction(
          ReadonlyContext(invocation_context), llm_request
      ):
        llm_request.append_instructions([planning_instruction])

      _remove_thought_from_request(llm_request)

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
    if not planner or isinstance(planner, BuiltInPlanner):
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
          author=invocation_context.agent.name,
          branch=invocation_context.branch,
          actions=callback_context._event_actions,
      )
      yield state_update_event


response_processor = _NlPlanningResponse()


def _get_planner(
    invocation_context: InvocationContext,
) -> Optional[BasePlanner]:
  from ...agents.llm_agent import Agent
  from ...planners.base_planner import BasePlanner

  agent = invocation_context.agent
  if not isinstance(agent, Agent):
    return None
  if not agent.planner:
    return None

  if isinstance(agent.planner, BasePlanner):
    return agent.planner
  return PlanReActPlanner()


def _remove_thought_from_request(llm_request: LlmRequest):
  if not llm_request.contents:
    return

  for content in llm_request.contents:
    if not content.parts:
      continue
    for part in content.parts:
      part.thought = None
