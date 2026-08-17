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
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.flows.llm_flows._nl_planning import request_processor
from google.adk.flows.llm_flows._nl_planning import response_processor
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.planners.base_planner import BasePlanner
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.planners.plan_re_act_planner import PlanReActPlanner
from google.genai import types
import pytest

from ... import testing_utils


@pytest.mark.asyncio
async def test_built_in_planner_only_drops_the_display_thought():
  """Test that BuiltInPlanner leaves everything but the display thought alone."""
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

  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  assert llm_request.contents == [
      types.UserContent(parts=[types.Part(text='Hello')]),
      types.ModelContent(parts=[types.Part(text='Here is my response')]),
      types.UserContent(parts=[types.Part(text='Follow up')]),
  ]


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

  Regression test: the base implementation used to be called instead.
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


class CustomPlanner(BasePlanner):
  """A planner deriving straight from BasePlanner."""

  def build_planning_instruction(
      self,
      readonly_context: ReadonlyContext,
      llm_request: LlmRequest,
  ) -> Optional[str]:
    return 'Custom instruction'

  def process_planning_response(
      self,
      callback_context: CallbackContext,
      response_parts: List[types.Part],
  ) -> Optional[List[types.Part]]:
    return response_parts


@pytest.mark.asyncio
async def test_custom_planner_instruction_appended():
  """Test that a planner deriving from BasePlanner gets its instruction used.

  Regression test: the request processor used to dispatch only on the two
  built-in planner types, so a custom planner's instruction was dropped.
  """
  agent = Agent(name='test_agent', planner=CustomPlanner())
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  llm_request = LlmRequest()

  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  assert llm_request.config.system_instruction == 'Custom instruction'


@pytest.mark.asyncio
async def test_custom_planner_removes_thought_from_request():
  """Test that thought parts are stripped for a custom planner."""
  agent = Agent(name='test_agent', planner=CustomPlanner())
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
      ]
  )

  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  for content in llm_request.contents:
    for part in content.parts or []:
      assert part.thought is None


def _request_text(llm_request: LlmRequest) -> str:
  return '\n'.join(
      part.text or ''
      for content in llm_request.contents or []
      for part in content.parts or []
  )


@pytest.mark.asyncio
async def test_plan_re_act_reasoning_survives_a_later_request():
  """PlanReActPlanner's own reasoning has to stay in the conversation history.

  The planner marks its reasoning as a thought so a caller can hide it, then
  relies on this processor clearing the marker so the text stays in context.
  Dropping thought parts anywhere earlier in the flow deletes that reasoning
  instead, which is what this test is here to catch.
  """
  reasoning = '/*PLANNING*/ Look the account up, then read the orders.\n'
  model = testing_utils.MockModel.create(
      responses=[
          types.Part(text=reasoning + '/*FINAL_ANSWER*/ You have two orders.'),
          'Both shipped yesterday.',
      ]
  )
  agent = Agent(name='test_agent', model=model, planner=PlanReActPlanner())
  runner = testing_utils.InMemoryRunner(agent)

  runner.run('How many orders do I have?')
  runner.run('When did they ship?')

  history = _request_text(model.requests[-1])
  assert reasoning in history
  assert 'You have two orders.' in history


@pytest.mark.asyncio
async def test_display_thought_is_not_resent_in_a_later_request():
  """A thought summary is for display, so it is not handed back to the model."""
  summary = 'The user is asking about orders, so I should look the account up.'
  model = testing_utils.MockModel.create(
      responses=[
          [
              types.Part(text=summary, thought=True),
              types.Part(text='You have two orders.'),
          ],
          'Both shipped yesterday.',
      ]
  )
  agent = Agent(
      name='test_agent',
      model=model,
      planner=BuiltInPlanner(
          thinking_config=types.ThinkingConfig(include_thoughts=True)
      ),
  )
  runner = testing_utils.InMemoryRunner(agent)

  runner.run('How many orders do I have?')
  runner.run('When did they ship?')

  history = _request_text(model.requests[-1])
  assert summary not in history
  assert 'You have two orders.' in history


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'kept_part',
    [
        types.Part(
            text='A summary the model wants back verbatim.',
            thought=True,
            thought_signature=b'opaque-thought-signature',
        ),
        types.Part(
            function_call=types.FunctionCall(name='get_orders', args={}),
            thought=True,
        ),
        types.Part(
            function_response=types.FunctionResponse(
                name='get_orders', response={'orders': []}
            ),
            thought=True,
        ),
    ],
    ids=['thought_signature', 'function_call', 'function_response'],
)
async def test_thought_part_the_model_needs_back_is_kept(kept_part):
  """A thought that carries state the model needs back is not a display thought."""
  agent = Agent(
      name='test_agent',
      planner=BuiltInPlanner(thinking_config=types.ThinkingConfig()),
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  llm_request = LlmRequest(
      contents=[
          types.ModelContent(
              parts=[kept_part, types.Part(text='You have two orders.')]
          )
      ]
  )

  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  assert llm_request.contents[0].parts == [
      kept_part,
      types.Part(text='You have two orders.'),
  ]


@pytest.mark.asyncio
async def test_content_of_only_thoughts_is_left_alone():
  """An empty content is not a valid turn, so the drop backs off."""
  agent = Agent(
      name='test_agent',
      planner=BuiltInPlanner(thinking_config=types.ThinkingConfig()),
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  only_thought = types.Part(text='Thinking about it.', thought=True)
  llm_request = LlmRequest(contents=[types.ModelContent(parts=[only_thought])])

  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  assert llm_request.contents[0].parts == [only_thought]


@pytest.mark.asyncio
async def test_display_thought_is_dropped_without_a_planner():
  """An agent with no planner gets the same treatment as a built-in planner."""
  agent = Agent(name='test_agent')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  llm_request = LlmRequest(
      contents=[
          types.ModelContent(
              parts=[
                  types.Part(text='Thinking about it.', thought=True),
                  types.Part(text='You have two orders.'),
              ]
          )
      ]
  )

  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  assert llm_request.contents[0].parts == [
      types.Part(text='You have two orders.')
  ]
