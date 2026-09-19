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

"""Tests for StreamingRouterNode (mid-stream preemptive graph advancement)."""

import asyncio
from typing import Any
from typing import AsyncGenerator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.workflow import Edge
from google.adk.workflow import JoinNode
from google.adk.workflow import START
from google.adk.workflow import StreamDecision
from google.adk.workflow import StreamingRouterNode
from google.adk.workflow import StreamView
from google.adk.workflow._node_runner import NodeRunner
from google.adk.workflow._workflow import Workflow
from google.genai import types
from pydantic import Field
import pytest

from .workflow_testing_utils import run_workflow
from .workflow_testing_utils import simplify_events_with_node
from .workflow_testing_utils import TestingNode as _RoutingNode


def _partial(text: str) -> Event:
  return Event(
      author='scripted',
      content=types.Content(role='model', parts=[types.Part(text=text)]),
      partial=True,
  )


def _final(text: str) -> Event:
  return Event(
      author='scripted',
      content=types.Content(role='model', parts=[types.Part(text=text)]),
      partial=False,
  )


class _ScriptedAgent(LlmAgent):
  """An LlmAgent whose stream is a fixed script of events.

  ``consumed`` counts how many scripted events were actually pulled — a
  preempting router closes the stream early, so a preempted run consumes
  fewer events than the script contains.
  """

  script: list[Event] = Field(default_factory=list)
  consumed: list[Event] = Field(default_factory=list)

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    for event in self.script:
      self.consumed.append(event)
      yield event


async def _run_router(
    router: StreamingRouterNode, node_input: str = 'hi'
) -> tuple[Context, list[Event]]:
  """Drives a router node via NodeRunner in isolation.

  ``_enqueue_event`` is mocked to collect events without blocking; the real
  method blocks non-partial events on the Runner main loop, which is absent
  in this unit-level harness. Returns the child context and the events the
  node emitted (state/artifact deltas are flushed onto these events and
  cleared from ``ctx.actions``).
  """
  session = Session(id='s', app_name='a', user_id='u')
  ic = InvocationContext(
      invocation_id='inv',
      agent=MagicMock(spec=BaseAgent),
      session=session,
      session_service=InMemorySessionService(),
  )
  collected: list[Event] = []

  async def _enqueue(event: Event) -> None:
    collected.append(event)

  object.__setattr__(ic, '_enqueue_event', AsyncMock(side_effect=_enqueue))
  parent_ctx = Context(invocation_context=ic, node_path='', run_id='1')
  child = await NodeRunner(node=router, parent_ctx=parent_ctx, run_id='1').run(
      node_input=node_input
  )
  return child, collected


def _billing_monitor(view: StreamView):
  low = view.text.lower()
  if 'billing' in low:
    return StreamDecision(route='billing')
  if 'technical' in low:
    return StreamDecision(route='technical')
  return None


@pytest.mark.asyncio
async def test_preempts_and_routes_midstream():
  agent = _ScriptedAgent(
      name='classifier',
      script=[
          _partial('Bil'),
          _partial('ling'),
          _partial(' department, definitely'),
          _final('Billing department, definitely'),
      ],
  )
  router = StreamingRouterNode(
      name='intent_router', agent=agent, monitor=_billing_monitor
  )

  child, _ = await _run_router(router)

  assert child.route == 'billing'
  # The decision fired after the second chunk completed "Billing"; the
  # remaining two events must never have been pulled from the stream.
  assert len(agent.consumed) == 2


@pytest.mark.asyncio
async def test_commits_output_and_state_delta_midstream():
  agent = _ScriptedAgent(
      name='classifier',
      output_key='intent',
      script=[_partial('ans'), _partial('wer=42'), _final('answer=42')],
  )

  def monitor(view: StreamView):
    if '=' in view.text:
      return StreamDecision(output=view.text)
    return None

  router = StreamingRouterNode(
      name='intent_router', agent=agent, monitor=monitor
  )

  child, events = await _run_router(router)

  assert child.output == 'answer=42'
  # The output_key delta is flushed onto an emitted event.
  state_deltas = [
      e.actions.state_delta for e in events if e.actions.state_delta
  ]
  assert {'intent': 'answer=42'} in state_deltas
  assert len(agent.consumed) == 2


@pytest.mark.asyncio
async def test_no_decision_falls_back_to_final_output():
  agent = _ScriptedAgent(
      name='classifier',
      script=[_partial('Hel'), _partial('lo'), _final('Hello world')],
  )
  # Monitor never decides.
  router = StreamingRouterNode(
      name='intent_router', agent=agent, monitor=lambda view: None
  )

  child, _ = await _run_router(router)

  assert child.route is None
  assert child.output == 'Hello world'
  # No preemption: the whole script is consumed.
  assert len(agent.consumed) == 3


@pytest.mark.asyncio
async def test_async_monitor_supported():
  agent = _ScriptedAgent(
      name='classifier',
      script=[_partial('techni'), _partial('cal'), _final('technical')],
  )

  async def monitor(view: StreamView):
    await asyncio.sleep(0)
    if 'technical' in view.text.lower():
      return StreamDecision(route='technical')
    return None

  router = StreamingRouterNode(
      name='intent_router', agent=agent, monitor=monitor
  )

  child, _ = await _run_router(router)

  assert child.route == 'technical'
  assert len(agent.consumed) == 2


@pytest.mark.asyncio
async def test_stop_false_continues_streaming_without_double_output():
  agent = _ScriptedAgent(
      name='classifier',
      script=[_partial('go'), _partial(' now'), _final('go now')],
  )

  def monitor(view: StreamView):
    if 'go' in view.text:
      return StreamDecision(route='fast', stop=False)
    return None

  router = StreamingRouterNode(
      name='intent_router', agent=agent, monitor=monitor
  )

  # Must not raise "Output already set": the final event's output is
  # suppressed because the decision already owns routing.
  child, _ = await _run_router(router)

  assert child.route == 'fast'
  # stop=False lets generation run to completion.
  assert len(agent.consumed) == 3


@pytest.mark.asyncio
async def test_forward_partials_false_suppresses_partial_messages():
  agent = _ScriptedAgent(
      name='classifier',
      script=[_partial('a'), _partial('b'), _final('ab')],
  )
  seen: list[bool] = []

  def monitor(view: StreamView):
    seen.append(view.event.partial)
    return None

  router = StreamingRouterNode(
      name='intent_router',
      agent=agent,
      monitor=monitor,
      forward_partials=False,
  )

  child, events = await _run_router(router)

  # Monitor still saw the partials even though they were not forwarded.
  assert seen == [True, True]
  assert child.output == 'ab'
  # No partial (streaming-message) events were emitted downstream.
  assert not any(e.partial for e in events)


@pytest.mark.asyncio
async def test_streaming_router_in_workflow_advances_graph():
  agent = _ScriptedAgent(
      name='classifier',
      script=[
          _partial('Bil'),
          _partial('ling'),
          _partial(' and more text the model never gets to finish'),
          _final('Billing and more text the model never gets to finish'),
      ],
  )
  router = StreamingRouterNode(
      name='intent_router', agent=agent, monitor=_billing_monitor
  )
  billing = _RoutingNode(name='billing_node', output='handled-billing')
  technical = _RoutingNode(name='technical_node', output='handled-technical')

  wf = Workflow(
      name='support_wf',
      edges=[
          Edge(from_node=START, to_node=router),
          Edge(from_node=router, to_node=billing, route='billing'),
          Edge(from_node=router, to_node=technical, route='technical'),
      ],
  )

  events, _, _ = await run_workflow(wf, message='my invoice is wrong')
  simplified = simplify_events_with_node(events)

  authors = [author for author, _ in simplified]
  assert any('billing_node' in a for a in authors)
  assert not any('technical_node' in a for a in authors)
  # Preemption held even through the full workflow run.
  assert len(agent.consumed) == 2


def test_stream_decision_requires_route_or_output():
  with pytest.raises(ValueError):
    StreamDecision()


def _relevance_monitor(view: StreamView):
  """Stops reading a source the instant it declares itself irrelevant."""
  if view.text.lstrip().upper().startswith('IRRELEVANT'):
    return StreamDecision(output={'relevant': False})
  return None


@pytest.mark.asyncio
async def test_fan_out_preempts_only_irrelevant_branch():
  """Parallel readers: the irrelevant branch cancels, the others run on."""
  irrelevant = _ScriptedAgent(
      name='src_a',
      script=[
          _partial('IRRELE'),
          _partial('VANT'),
          _partial(' the model keeps talking but nobody is listening'),
          _final('IRRELEVANT ...'),
      ],
  )
  relevant_1 = _ScriptedAgent(
      name='src_b',
      script=[_partial('fact'), _partial(' one'), _final('fact one')],
  )
  relevant_2 = _ScriptedAgent(
      name='src_c',
      script=[_partial('fact'), _partial(' two'), _final('fact two')],
  )

  reader_a = StreamingRouterNode(
      name='reader_a', agent=irrelevant, monitor=_relevance_monitor
  )
  reader_b = StreamingRouterNode(
      name='reader_b', agent=relevant_1, monitor=_relevance_monitor
  )
  reader_c = StreamingRouterNode(
      name='reader_c', agent=relevant_2, monitor=_relevance_monitor
  )

  join = JoinNode(name='join_sources')
  captured: dict[str, Any] = {}

  async def synthesize(node_input: dict[str, Any]):
    captured['fan_in'] = node_input
    yield Event(message='synthesized')

  wf = Workflow(
      name='fan_out_wf',
      edges=[('START', (reader_a, reader_b, reader_c), join, synthesize)],
  )

  await run_workflow(wf, message='the query')

  # Only the irrelevant branch was preempted; it consumed 2 of its 4 events.
  assert len(irrelevant.consumed) == 2
  # The relevant branches ran their full streams uninterrupted.
  assert len(relevant_1.consumed) == 3
  assert len(relevant_2.consumed) == 3

  # The join fanned all three branches back in, keyed by node name.
  fan_in = captured['fan_in']
  assert fan_in['reader_a'] == {'relevant': False}
  assert fan_in['reader_b'] == 'fact one'
  assert fan_in['reader_c'] == 'fact two'
