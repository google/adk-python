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

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from tests.unittests import testing_utils
from tests.unittests.agents.llm.event_utils import text_parts

_USER_ID = 'test_user'
_SESSION_ID = 'test_session'


async def _setup_runner(mock_model, tools=None, **agent_kwargs):
  """Setup runner with LlmAgent directly."""
  llm_agent = LlmAgent(
      name='test_agent',
      model=mock_model,
      tools=tools or [],
      **agent_kwargs,
  )
  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name='test', user_id=_USER_ID, session_id=_SESSION_ID
  )
  runner = Runner(
      app_name='test',
      agent=llm_agent,
      session_service=session_service,
  )
  return runner


async def _run_turn(runner, user_message):
  """Run a single turn."""
  return [
      e
      async for e in runner.run_async(
          user_id=_USER_ID,
          session_id=_SESSION_ID,
          new_message=types.Content(
              role='user', parts=[types.Part(text=user_message)]
          ),
      )
  ]


async def _resume_turn(
    runner, prev_events, tool_name, tool_response_value='done'
):
  """Resume after an interrupt."""
  fc_ids = []
  for e in prev_events:
    if e.content and e.content.parts:
      for p in e.content.parts:
        if (
            p.function_call
            and p.function_call.name == tool_name
            and p.function_call.id
        ):
          fc_ids.append(p.function_call.id)
    if getattr(e.output, 'function_calls', None):
      for fc in e.output.function_calls:
        if fc.name == tool_name and fc.id:
          fc_ids.append(fc.id)

  if not fc_ids:
    for e in prev_events:
      if e.long_running_tool_ids:
        fc_ids = list(e.long_running_tool_ids)
        break

  invocation_id = prev_events[0].invocation_id

  fr_parts = [
      types.Part(
          function_response=types.FunctionResponse(
              name=tool_name,
              id=fc_id,
              response={'result': tool_response_value},
          )
      )
      for fc_id in fc_ids
  ]
  resume_msg = types.Content(role='user', parts=fr_parts)

  return [
      e
      async for e in runner.run_async(
          user_id=_USER_ID,
          session_id=_SESSION_ID,
          invocation_id=invocation_id,
          new_message=resume_msg,
      )
  ]


def create_lro_tool(name: str = 'long_running_op') -> LongRunningFunctionTool:
  """Creates a minimal LRO tool for testing."""

  def _impl() -> None:
    return None

  _impl.__name__ = name
  return LongRunningFunctionTool(_impl)


# ---------------------------------------------------------------------------
# Tests: Single Agent
# ---------------------------------------------------------------------------


class TestSingleAgentInterruptions:
  """Tests for single agent triggering interruptions."""

  async def test_single_agent_yields_on_long_running_tool(self):
    """Single agent yields on Long Running Tool.

    Arrange: Set up a single agent with a long running tool.
    Act: Run the agent with a prompt that triggers the tool.
    Assert: Verify that the execution yields a long running tool interrupt.
    """

    fc = types.Part.from_function_call(name='long_running_op', args={})
    mock_model = testing_utils.MockModel.create(responses=[fc, 'Final answer'])

    lro_tool = create_lro_tool()
    runner = await _setup_runner(mock_model, tools=[lro_tool])

    # Act: Run first turn
    events = await _run_turn(runner, 'Go')

    # Assert: Should have triggered function call
    assert any(
        any(
            p.function_call and p.function_call.name == 'long_running_op'
            for p in e.content.parts or []
        )
        for e in events
    )
    assert len(mock_model.requests) == 1

    # Act: Resume
    resume_events = await _resume_turn(runner, events, 'long_running_op')

    # Assert: Should have completed
    assert any('Final answer' in t for t in text_parts(resume_events))
    assert len(mock_model.requests) == 2


class TestNestedAgentInterruptions:
  """Tests for multi-agent setups with interruptions."""

  async def test_child_agent_interrupt_and_resume(self):
    """Child agent yields on LRO and resumes successfully.

    Arrange: Parent agent with Child agent. Parent transfers to Child.
      Child calls LRO tool.
    Act: Run, expect LRO interrupt. Then resume.
    Assert: Should complete successfully.
    """

    def transfer_to_child(tool_context: ToolContext) -> str:
      tool_context.actions.transfer_to_agent = 'child_agent'
      return 'transferring'

    # Child agent
    fc_child = types.Part.from_function_call(name='child_lro', args={})
    child_mock_model = testing_utils.MockModel.create(
        responses=[fc_child, 'Child final answer']
    )

    lro_tool = create_lro_tool('child_lro')

    child_agent = LlmAgent(
        name='child_agent',
        model=child_mock_model,
        tools=[lro_tool],
    )

    # Parent agent
    fc_parent = types.Part.from_function_call(name='transfer_to_child', args={})
    parent_mock_model = testing_utils.MockModel.create(
        responses=[fc_parent, fc_parent, 'Parent final answer']
    )

    parent_agent = LlmAgent(
        name='parent_agent',
        model=parent_mock_model,
        tools=[transfer_to_child],
        sub_agents=[child_agent],
    )

    # Setup runner
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name='test', user_id=_USER_ID, session_id=_SESSION_ID
    )
    runner = Runner(
        app_name='test', agent=parent_agent, session_service=session_service
    )

    # When Parent runs the first turn
    events = await _run_turn(runner, 'Go')

    # Then it should trigger child LRO interrupt
    assert any(e.long_running_tool_ids for e in events)

    # When Parent resumes the turn
    resume_events = await _resume_turn(runner, events, 'child_lro')

    # Then it should complete successfully
    assert any('Child final answer' in t for t in text_parts(resume_events))

  @pytest.mark.xfail(reason='Task agent as subagent not supported yet.')
  async def test_task_child_agent_interrupt_and_resume(self):
    """Task child agent yields on LRO and resumes successfully.

    Arrange: Parent agent with Task Child agent. Parent transfers to Child.
      Child calls LRO tool.
    Act: Run, expect LRO interrupt. Then resume.
    Assert: Should complete successfully.
    """

    def transfer_to_child(tool_context: ToolContext) -> str:
      tool_context.actions.transfer_to_agent = 'child_agent'
      return 'transferring'

    # Child agent (Task mode)
    fc_child = types.Part.from_function_call(name='child_lro', args={})
    fc_finish = types.Part.from_function_call(
        name='finish_task', args={'result': 'Task done'}
    )
    child_mock_model = testing_utils.MockModel.create(
        responses=[fc_child, fc_finish, 'Child final answer']
    )

    lro_tool = create_lro_tool('child_lro')

    child_agent = LlmAgent(
        name='child_agent',
        model=child_mock_model,
        tools=[lro_tool],
        mode='task',
    )

    # Parent agent
    fc_parent = types.Part.from_function_call(name='transfer_to_child', args={})
    parent_mock_model = testing_utils.MockModel.create(
        responses=[fc_parent, 'Parent final answer']
    )

    parent_agent = LlmAgent(
        name='parent_agent',
        model=parent_mock_model,
        tools=[transfer_to_child],
        sub_agents=[child_agent],
    )

    # Setup runner
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name='test', user_id=_USER_ID, session_id=_SESSION_ID
    )
    runner = Runner(
        app_name='test', agent=parent_agent, session_service=session_service
    )

    # When Parent runs the first turn
    events = await _run_turn(runner, 'Go')

    # Then it should trigger child LRO interrupt
    assert any(e.long_running_tool_ids for e in events)

    # When Parent resumes the turn
    resume_events = await _resume_turn(runner, events, 'child_lro')

    # Then it should complete successfully
    assert any('Parent final answer' in t for t in text_parts(resume_events))

  @pytest.mark.xfail(reason='Single-turn agent as subagent not supported yet.')
  async def test_single_turn_child_agent_interrupt_and_resume(self):
    """Single-turn child agent yields on LRO and resumes successfully.

    Arrange: Parent agent with Single-turn Child agent.
      Parent calls Child via tool.
      Child calls LRO tool.
    Act: Run, expect LRO interrupt. Then resume.
    Assert: Should complete successfully.
    """

    # Child agent (Single-turn)
    fc_child = types.Part.from_function_call(name='child_lro', args={})
    child_mock_model = testing_utils.MockModel.create(
        responses=[fc_child, 'Child final answer']
    )

    lro_tool = create_lro_tool('child_lro')

    child_agent = LlmAgent(
        name='child_agent',
        model=child_mock_model,
        tools=[lro_tool],
        mode='single_turn',
    )

    # Parent agent
    fc_call_child = types.Part.from_function_call(
        name='child_agent', args={'request': 'Go to child'}
    )
    parent_mock_model = testing_utils.MockModel.create(
        responses=[fc_call_child, 'Parent final answer']
    )

    parent_agent = LlmAgent(
        name='parent_agent',
        model=parent_mock_model,
        sub_agents=[child_agent],
    )

    # Setup runner
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name='test', user_id=_USER_ID, session_id=_SESSION_ID
    )
    runner = Runner(
        app_name='test', agent=parent_agent, session_service=session_service
    )

    # When Parent runs the first turn
    events = await _run_turn(runner, 'Go')

    # Then it should trigger child LRO interrupt
    assert any(e.long_running_tool_ids for e in events)

    # When Parent resumes the turn
    resume_events = await _resume_turn(runner, events, 'child_lro')

    # Then it should complete successfully
    assert any('Parent final answer' in t for t in text_parts(resume_events))
