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
from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext as BaseInvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.workflow import START
from google.adk.workflow._workflow import Workflow
from google.genai import types
import pytest

from tests.unittests import testing_utils

from .workflow_testing_utils import create_runner
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


class MockAgent(BaseAgent):
  """A mock agent that yields predefined events."""

  async def _run_async_impl(
      self, ctx: BaseInvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(author=self.name)
    yield Event(author='sub_agent')


@pytest.mark.asyncio
async def test_base_agent_as_node_run():
  """Tests that BaseAgent runs as a node and preserves event authors."""
  agent = MockAgent(name='mock_agent')

  session = Session(app_name='test', user_id='user', id='session')
  session_service = InMemorySessionService()
  ic = BaseInvocationContext(
      invocation_id='inv',
      session=session,
      session_service=session_service,
  )
  ctx = Context(ic, node_path='wf')

  events = []
  async for event in agent.run(ctx=ctx, node_input=None):
    events.append(event)

  assert len(events) == 2
  assert events[0].author == 'mock_agent'
  assert events[0].node_info.path == 'wf'
  assert events[1].author == 'sub_agent'
  assert not events[1].node_info.path
  assert ctx.event_author == 'sub_agent'


class _PresetPathAgent(BaseAgent):

  async def _run_async_impl(
      self, ctx: BaseInvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(author=self.name, node_path='custom/existing@1')


@pytest.mark.asyncio
async def test_base_agent_as_node_keeps_preset_node_path():
  """An agent's own event that already has a node path keeps it."""
  agent = _PresetPathAgent(name='preset_agent')
  session = Session(app_name='test', user_id='user', id='session')
  ic = BaseInvocationContext(
      invocation_id='inv',
      session=session,
      session_service=InMemorySessionService(),
  )
  ctx = Context(ic, node_path='wf')

  events = [e async for e in agent.run(ctx=ctx, node_input=None)]

  assert events[0].node_info.path == 'custom/existing@1'


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
  runner = create_runner(app_name='test', node=wf, session_service=ss)
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
  runner = create_runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(parts=[types.Part(text='start')], role='user')
  async for _ in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    pass

  # AgentNode does not record content as node output, so the next node
  # receives None as input.
  assert node_b.received_inputs == [None]


async def _run_workflow(wf: Workflow) -> list[Event]:
  ss = InMemorySessionService()
  runner = create_runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')
  msg = types.Content(parts=[types.Part(text='start')], role='user')
  return [
      event
      async for event in runner.run_async(
          user_id='u', session_id=session.id, new_message=msg
      )
  ]


@pytest.mark.asyncio
async def test_node_after_parallel_agent_node_sees_every_branch():
  """A node after a ParallelAgent node runs on the workflow's own branch.

  The ParallelAgent relays each sub-agent's events on that sub-agent's
  branch. Its successor must not inherit whichever of those branches spoke
  last: an agent there reads history filtered by branch, so it would see only
  that one sub-agent's answer, and its own replies would land on that
  sub-agent's branch.
  """
  researcher_a = LlmAgent(
      name='researcher_a',
      model=testing_utils.MockModel.create(responses=['finding from a']),
  )
  researcher_b = LlmAgent(
      name='researcher_b',
      model=testing_utils.MockModel.create(responses=['finding from b']),
  )
  research = ParallelAgent(
      name='research', sub_agents=[researcher_a, researcher_b]
  )
  summarizer_model = testing_utils.MockModel.create(responses=['summary'])
  report = SequentialAgent(
      name='report',
      sub_agents=[LlmAgent(name='summarizer', model=summarizer_model)],
  )

  events = await _run_workflow(
      Workflow(name='wf', edges=[(START, research, report)])
  )

  summarizer_request = ' '.join(
      part.text
      for content in summarizer_model.requests[0].contents
      for part in content.parts or []
      if part.text
  )
  assert 'finding from a' in summarizer_request
  assert 'finding from b' in summarizer_request
  assert {event.author: event.branch for event in events if event.content} == {
      'researcher_a': 'research.researcher_a',
      'researcher_b': 'research.researcher_b',
      'summarizer': None,
  }


class _Dispatcher(BaseAgent):
  """Runs its sub-agent on a sub-branch and hands back the final answer."""

  async def _run_async_impl(
      self, ctx: BaseInvocationContext
  ) -> AsyncGenerator[Event, None]:
    target = self.sub_agents[0]
    parent_branch = ctx.branch
    ctx.branch = f'{self.name}.{target.name}'
    async for event in target.run_async(ctx):
      if event.is_final_response():
        event.branch = parent_branch
      yield event


@pytest.mark.asyncio
async def test_answer_relayed_onto_parent_branch_stays_there():
  """A relayed answer re-stamped onto the parent branch is not pulled back.

  The dispatcher keeps its sub-agent's tool traffic on a sub-branch and hands
  the final answer back on its own branch, so later agents see the answer
  but not the tool calls. The tool events must not move the dispatcher node
  onto the sub-branch, or the branch-less answer gets stamped with it.
  """

  def lookup_arrival() -> dict[str, str]:
    """Looks up the arrival time."""
    return {'arrival': 'Tuesday 14:05'}

  bookings = LlmAgent(
      name='bookings',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(name='lookup_arrival', args={}),
              'You land on Tuesday at 14:05.',
          ]
      ),
      tools=[lookup_arrival],
  )
  dispatcher = _Dispatcher(name='dispatcher', sub_agents=[bookings])

  events = await _run_workflow(Workflow(name='wf', edges=[(START, dispatcher)]))

  bookings_events = [e for e in events if e.author == 'bookings' and e.content]
  assert [e.branch for e in bookings_events] == [
      'dispatcher.bookings',
      'dispatcher.bookings',
      None,
  ]
  assert bookings_events[-1].content.parts[0].text == (
      'You land on Tuesday at 14:05.'
  )
