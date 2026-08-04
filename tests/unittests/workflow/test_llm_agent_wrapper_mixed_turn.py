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

"""Unit tests for chat-wrapper mixed-turn FR draining helpers.

Verifies that the wrapper can detect eager (non-deferred) tool calls that
must be drained before breaking out of ``run_async`` on task delegation.
"""

from __future__ import annotations

from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.tools.agent_tool import _TaskAgentTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.workflow import _llm_agent_wrapper as wrapper
from google.genai import types
import pytest


def _echo(value: str) -> dict[str, str]:
  """Return the provided value."""
  return {'value': value}


def _model_event(*parts: types.Part) -> Event:
  return Event(
      author='coordinator',
      content=types.Content(role='model', parts=list(parts)),
  )


def _fc(name: str, call_id: str) -> types.Part:
  return types.Part(
      function_call=types.FunctionCall(name=name, args={}, id=call_id)
  )


def _fr(name: str, call_id: str) -> types.Part:
  return types.Part(
      function_response=types.FunctionResponse(
          name=name, response={'ok': True}, id=call_id
      )
  )


def test_event_has_eager_tool_calls_true_for_regular_plus_task():
  """A mixed turn with a FunctionTool and task tool reports eager calls."""
  task_agent = LlmAgent(name='specialist', mode='task', model='unused')
  tools_dict = {
      'echo': FunctionTool(_echo),
      'specialist': _TaskAgentTool(task_agent),
  }
  event = _model_event(_fc('echo', '1'), _fc('specialist', '2'))

  assert wrapper._event_has_eager_tool_calls(event, tools_dict) is True


def test_event_has_eager_tool_calls_false_for_task_only():
  """Task-only turns should not drain (no FR is produced by the flow)."""
  task_agent = LlmAgent(name='specialist', mode='task', model='unused')
  tools_dict = {'specialist': _TaskAgentTool(task_agent)}
  event = _model_event(_fc('specialist', '1'))

  assert wrapper._event_has_eager_tool_calls(event, tools_dict) is False


@pytest.mark.asyncio
async def test_drain_pending_tool_response_events_yields_fr_then_stops():
  """Drain yields the FR event and stops before a following model event."""

  async def _gen():
    yield Event(
        author='coordinator',
        content=types.Content(role='user', parts=[_fr('echo', '1')]),
    )
    yield _model_event(types.Part.from_text(text='should not be drained'))

  drained = [
      event
      async for event in wrapper._drain_pending_tool_response_events(_gen())
  ]

  assert len(drained) == 1
  assert drained[0].get_function_responses()[0].name == 'echo'


@pytest.mark.asyncio
async def test_drain_pending_tool_response_events_stops_on_model_role():
  """Drain stops immediately when the next event is already a model turn."""

  async def _gen():
    yield _model_event(types.Part.from_text(text='next round'))
    yield Event(
        author='coordinator',
        content=types.Content(role='user', parts=[_fr('echo', '1')]),
    )

  drained = [
      event
      async for event in wrapper._drain_pending_tool_response_events(_gen())
  ]

  assert drained == []
