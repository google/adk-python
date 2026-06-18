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

"""Tests for BaseAgent instances used as nodes in a Workflow."""

from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext as BaseInvocationContext
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import START
from google.adk.workflow._workflow import Workflow
from google.genai import types
import pytest

from .workflow_testing_utils import InputCapturingNode
from .workflow_testing_utils import simplify_events_with_node


class SimpleAgent(BaseAgent):
  """A simple agent for testing."""

  message: str = ''

  async def _run_async_impl(
      self, ctx: BaseInvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Yields a single event with a message."""
    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        content=types.Content(parts=[types.Part(text=self.message)]),
    )


@pytest.mark.asyncio
async def test_run_async_with_agent_nodes(request: pytest.FixtureRequest):
  """BaseAgent nodes emit content events through the workflow."""
  agent_a = SimpleAgent(name='AgentA', message='Hello')
  agent_b = SimpleAgent(name='AgentB', message='World')
  wf = Workflow(
      name='wf_with_agents',
      edges=[
          (START, agent_a),
          (agent_a, agent_b),
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
      ('wf_with_agents@1/AgentA@1', 'Hello'),
      ('wf_with_agents@1/AgentB@1', 'World'),
  ]


@pytest.mark.asyncio
async def test_run_async_with_agent_node_piping_data(
    request: pytest.FixtureRequest,
):
  """AgentNode content is not piped as output to the next node."""
  agent_a = SimpleAgent(name='AgentA', message='Hello')
  node_b = InputCapturingNode(name='NodeB')
  wf = Workflow(
      name='wf_with_agent_piping',
      edges=[
          (START, agent_a),
          (agent_a, node_b),
      ],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(parts=[types.Part(text='start')], role='user')
  async for _ in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    pass

  # AgentNode does not record content as node output, so the next node
  # receives None as input.
  assert node_b.received_inputs == [None]
