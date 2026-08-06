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

"""Tests for SpeculativeRouterNode and the JSON repairer."""

import asyncio
import json
from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.workflow import repair_json
from google.adk.workflow import SpeculativeRouterNode
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._workflow import Workflow
from google.genai import types
from pydantic import ConfigDict
from pydantic import Field
import pytest
from typing_extensions import override

from .workflow_testing_utils import get_outputs
from .workflow_testing_utils import run_workflow

# Target lifecycle, recorded as (state, path). Module-global to survive the
# scheduler's model_copy of the target node.
_EVENTS: list[tuple[str, Any]] = []


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
  """An LlmAgent whose stream is a fixed script of events."""

  script: list[Event] = Field(default_factory=list)

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    for event in self.script:
      yield event


def _path(payload: Any) -> Any:
  if isinstance(payload, dict):
    return payload.get('arguments', {}).get('path')
  return None


class _CaptureTarget(BaseNode):
  """Records the payload it ran with; sleeps so speculation can overlap."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  delay: float = 0.0

  @override
  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    p = _path(node_input)
    _EVENTS.append(('start', p))
    try:
      await asyncio.sleep(self.delay)
    except asyncio.CancelledError:
      _EVENTS.append(('cancelled', p))
      raise
    _EVENTS.append(('done', p))
    yield {'read': p}


# --- repair_json --------------------------------------------------------------


@pytest.mark.parametrize(
    'fragment,expected',
    [
        ('{"a": "b', {'a': 'b'}),
        ('{"a": [1, 2', {'a': [1, 2]}),
        ('{"name":"x","arguments":{"path":"src/ma', {
            'name': 'x',
            'arguments': {'path': 'src/ma'},
        }),
        ('{"ok": tru', {'ok': True}),
        ('{"a":1,', {'a': 1}),
        ('{"a":1, "b":', {'a': 1, 'b': None}),
    ],
)
def test_repair_json_completes_truncated_fragments(fragment, expected):
  assert json.loads(repair_json(fragment)) == expected


def test_repair_json_leaves_complete_json_parseable():
  assert json.loads(repair_json('{"a": 1, "b": [2, 3]}')) == {
      'a': 1,
      'b': [2, 3],
  }


# --- SpeculativeRouterNode ----------------------------------------------------


def _spec_node(agent: _ScriptedAgent, target: _CaptureTarget):
  return SpeculativeRouterNode(name='spec', agent=agent, target=target)


@pytest.mark.asyncio
async def test_speculation_hit_keeps_result_and_runs_target_once():
  _EVENTS.clear()
  # Partial already carries the full path; final only closes the braces.
  agent = _ScriptedAgent(
      name='m',
      script=[
          _partial('TOOL_CALL: {"name":"read_file","arguments":{"path":"src/main.c"'),
          _final('TOOL_CALL: {"name":"read_file","arguments":{"path":"src/main.c"}}'),
      ],
  )
  node = _spec_node(agent, _CaptureTarget(name='reader', delay=0.05))
  wf = Workflow(name='w_hit', edges=[('START', node)])

  events, _, _ = await run_workflow(wf)

  assert {'read': 'src/main.c'} in get_outputs(events)
  starts = [p for state, p in _EVENTS if state == 'start']
  # Speculation was correct -> the target ran exactly once (no re-run).
  assert starts == ['src/main.c']
  assert ('done', 'src/main.c') in _EVENTS
  assert ('cancelled', 'src/main.c') not in _EVENTS


@pytest.mark.asyncio
async def test_speculation_miss_cancels_and_reruns_with_final_payload():
  _EVENTS.clear()
  # Partial repairs to the wrong (truncated) path; final has the real one.
  agent = _ScriptedAgent(
      name='m',
      script=[
          _partial('TOOL_CALL: {"name":"read_file","arguments":{"path":"src/ma'),
          _final('TOOL_CALL: {"name":"read_file","arguments":{"path":"src/main.c"}}'),
      ],
  )
  # Long delay so the speculative run is still in-flight when the final arrives.
  node = _spec_node(agent, _CaptureTarget(name='reader', delay=5.0))
  wf = Workflow(name='w_miss', edges=[('START', node)])

  events, _, _ = await run_workflow(wf)

  # The verified (final) payload wins.
  assert {'read': 'src/main.c'} in get_outputs(events)
  # The wrong speculative guess was started then cancelled...
  assert ('start', 'src/ma') in _EVENTS
  assert ('cancelled', 'src/ma') in _EVENTS
  assert ('done', 'src/ma') not in _EVENTS
  # ...and the correct payload was run to completion.
  assert ('done', 'src/main.c') in _EVENTS


@pytest.mark.asyncio
async def test_no_partial_call_runs_target_once_on_final():
  _EVENTS.clear()
  agent = _ScriptedAgent(
      name='m',
      script=[
          _partial('thinking about it...'),
          _final('TOOL_CALL: {"name":"read_file","arguments":{"path":"x"}}'),
      ],
  )
  node = _spec_node(agent, _CaptureTarget(name='reader', delay=0.01))
  wf = Workflow(name='w_none', edges=[('START', node)])

  events, _, _ = await run_workflow(wf)

  assert {'read': 'x'} in get_outputs(events)
  starts = [p for state, p in _EVENTS if state == 'start']
  assert starts == ['x']  # never speculated; ran once on the final call


@pytest.mark.asyncio
async def test_combine_returns_agent_text_and_target_result():
  _EVENTS.clear()
  # The agent emits a directive plus a trailing rationale that the caller wants.
  rationale = ' because this file holds the entrypoint.'
  agent = _ScriptedAgent(
      name='m',
      script=[
          _partial('TOOL_CALL: {"name":"read_file","arguments":{"path":"src/main.c"'),
          _final(
              'TOOL_CALL: {"name":"read_file","arguments":{"path":"src/main.c"}}'
              + rationale
          ),
      ],
  )
  node = SpeculativeRouterNode(
      name='spec',
      agent=agent,
      target=_CaptureTarget(name='reader', delay=0.02),
      combine=lambda plan, result: {'plan': plan, 'result': result},
  )
  wf = Workflow(name='w_combine', edges=[('START', node)])

  events, _, _ = await run_workflow(wf)

  outputs = get_outputs(events)
  combined = next(o for o in outputs if isinstance(o, dict) and 'plan' in o)
  # The target's verified result is carried through...
  assert combined['result'] == {'read': 'src/main.c'}
  # ...alongside the agent's FULL text (directive + required rationale), proving
  # the streamed tail is a returned deliverable, not a discarded artifact.
  assert rationale.strip() in combined['plan']


@pytest.mark.asyncio
async def test_should_speculate_gate_suppresses_early_dispatch():
  _EVENTS.clear()
  agent = _ScriptedAgent(
      name='m',
      script=[
          _partial('TOOL_CALL: {"name":"read_file","arguments":{"path":"s'),
          _final('TOOL_CALL: {"name":"read_file","arguments":{"path":"src/main.c"}}'),
      ],
  )
  # Only speculate once the path looks long enough to be worth a guess.
  node = SpeculativeRouterNode(
      name='spec',
      agent=agent,
      target=_CaptureTarget(name='reader', delay=0.01),
      should_speculate=lambda p: len(_path(p) or '') >= 6,
  )
  wf = Workflow(name='w_gate', edges=[('START', node)])

  events, _, _ = await run_workflow(wf)

  assert {'read': 'src/main.c'} in get_outputs(events)
  # The short 's' guess was gated out, so the target only ran on the final.
  starts = [p for state, p in _EVENTS if state == 'start']
  assert starts == ['src/main.c']
