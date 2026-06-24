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

"""Unit tests for NL planning logic."""

from typing import List
from typing import Optional
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.flows.llm_flows._nl_planning import request_processor
from google.adk.flows.llm_flows._nl_planning import response_processor
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.planners.plan_re_act_planner import PlanReActPlanner
from google.genai import types
import pytest

from ... import testing_utils


@pytest.mark.asyncio
async def test_built_in_planner_content_list_unchanged():
  """Test that BuiltInPlanner doesn't modify LlmRequest content list."""
  planner = BuiltInPlanner(thinking_config=types.ThinkingConfig())
  agent = Agent(name='test_agent', planner=planner)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  # Create user/model/user conversation with thought in model response
  llm_request = LlmRequest(
      contents=[
          types.UserContent(parts=[types.Part(text='Hello')]),
          types.ModelContent(
              parts=[
                  types.Part(text='thinking...', thought=True),
                  types.Part(text='Here is my response'),
              ]
          ),
          types.UserContent(parts=[types.Part(text='Follow up')]),
      ]
  )
  original_contents = llm_request.contents.copy()

  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  assert llm_request.contents == original_contents


@pytest.mark.asyncio
async def test_built_in_planner_apply_thinking_config_called():
  """Test that BuiltInPlanner.apply_thinking_config is called."""
  planner = BuiltInPlanner(thinking_config=types.ThinkingConfig())
  planner.apply_thinking_config = MagicMock()
  agent = Agent(name='test_agent', planner=planner)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  llm_request = LlmRequest()

  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  planner.apply_thinking_config.assert_called_once_with(llm_request)


@pytest.mark.asyncio
async def test_plan_react_planner_instruction_appended():
  """Test that PlanReActPlanner appends planning instruction."""
  planner = PlanReActPlanner()
  planner.build_planning_instruction = MagicMock(
      return_value='Test instruction'
  )
  agent = Agent(name='test_agent', planner=planner)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )

  llm_request = LlmRequest()
  llm_request.config.system_instruction = 'Original instruction'

  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  assert llm_request.config.system_instruction == ("""\
Original instruction

Test instruction""")


@pytest.mark.asyncio
async def test_remove_thought_from_request_with_thoughts():
  """Test that PlanReActPlanner removes thought flags from content parts."""
  planner = PlanReActPlanner()
  agent = Agent(name='test_agent', planner=planner)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  llm_request = LlmRequest(
      contents=[
          types.UserContent(parts=[types.Part(text='initial query')]),
          types.ModelContent(
              parts=[
                  types.Part(text='Text with thought', thought=True),
                  types.Part(text='Regular text'),
              ]
          ),
          types.UserContent(parts=[types.Part(text='follow up')]),
      ]
  )

  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  assert all(
      part.thought is None
      for content in llm_request.contents
      for part in content.parts or []
  )


class OverriddenBuiltInPlanner(BuiltInPlanner):
  """Subclass that overrides process_planning_response."""

  def __init__(self, *, thinking_config: types.ThinkingConfig):
    super().__init__(thinking_config=thinking_config)
    self.process_planning_response_called = False
    self.received_parts = None

  def process_planning_response(
      self,
      callback_context: CallbackContext,
      response_parts: List[types.Part],
  ) -> Optional[List[types.Part]]:
    self.process_planning_response_called = True
    self.received_parts = response_parts
    return response_parts


class NonOverriddenBuiltInPlanner(BuiltInPlanner):
  """Subclass that does NOT override process_planning_response."""

  pass


@pytest.mark.asyncio
async def test_overridden_subclass_process_planning_response_called():
  """Test that subclasses overriding process_planning_response have it called.

  Regression test for issue #4133.
  """
  planner = OverriddenBuiltInPlanner(thinking_config=types.ThinkingConfig())
  agent = Agent(name='test_agent', planner=planner)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )

  response_parts = [
      types.Part(text='thinking...', thought=True),
      types.Part(text='Here is my response'),
  ]
  llm_response = LlmResponse(
      content=types.Content(role='model', parts=response_parts)
  )

  async for _ in response_processor.run_async(invocation_context, llm_response):
    pass

  assert planner.process_planning_response_called
  assert planner.received_parts == response_parts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'planner_class',
    [BuiltInPlanner, NonOverriddenBuiltInPlanner],
    ids=['base_class', 'non_overridden_subclass'],
)
async def test_process_planning_response_not_called_without_override(
    planner_class,
):
  """Test that process_planning_response is not called for base or non-overridden subclasses."""
  planner = planner_class(thinking_config=types.ThinkingConfig())
  agent = Agent(name='test_agent', planner=planner)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )

  response_parts = [
      types.Part(text='thinking...', thought=True),
      types.Part(text='Here is my response'),
  ]
  llm_response = LlmResponse(
      content=types.Content(role='model', parts=response_parts)
  )

  with patch.object(
      BuiltInPlanner,
      'process_planning_response',
  ) as mock_method:
    async for _ in response_processor.run_async(
        invocation_context, llm_response
    ):
      pass
    mock_method.assert_not_called()
