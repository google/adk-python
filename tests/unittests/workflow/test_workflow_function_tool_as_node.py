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

"""Tests for FunctionTool nodes in a Workflow."""

from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.adk.workflow import START
from google.adk.workflow._workflow import Workflow
from google.genai import types
import pytest

from .workflow_testing_utils import simplify_events_with_node


def _produce_input() -> None:
  """Absorbs user content so downstream ToolNodes receive None."""
  return None


def _func_a() -> dict[str, str]:
  """Returns a value from function A."""
  return {'val': 'Hello'}


def _func_b(val: str) -> str:
  """Returns a value incorporating input from A."""
  return f'{val}_world'


@pytest.mark.asyncio
async def test_run_async_with_function_tools():
  """FunctionTool output is piped as input to the next FunctionTool."""
  tool_a = FunctionTool(_func_a)
  tool_b = FunctionTool(_func_b)
  wf = Workflow(
      name='wf_with_tools',
      edges=[
          (START, _produce_input, tool_a, tool_b),
      ],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(parts=[types.Part(text='start')], role='user')
  events: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)

  assert simplify_events_with_node(events) == [
      (
          'wf_with_tools@1/_func_a@1',
          {'output': {'val': 'Hello'}},
      ),
      (
          'wf_with_tools@1/_func_b@1',
          {'output': 'Hello_world'},
      ),
  ]
