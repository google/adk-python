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

"""Tests for nested workflow output event deduplication.

Verifies that nested workflows produce only leaf terminal events and
resolve output via terminal path resolution from the graph structure.
"""

from typing import Any

from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.workflow._workflow import Workflow
import pytest

from . import testing_utils


def _is_checkpoint(event: Event) -> bool:
  """Returns True if event is an agent state checkpoint or end_of_agent."""
  if event.actions.agent_state is not None:
    return True
  if event.actions.end_of_agent:
    return True
  return False


def _output_events(events: list[Event]) -> list[Event]:
  """Returns only events with output data (not checkpoint/state)."""
  return [e for e in events if not _is_checkpoint(e) and e.output is not None]


async def test_two_level_nesting_deduplicates(
    request,
):
  """Two-level nesting emits only the leaf's output event, not finalize events.

  Setup: outer → inner → leaf.
  Assert: exactly 1 output event from 'outer/inner/leaf', no
    duplicate finalize events from inner or outer.
  """

  async def leaf(node_input: Any):
    return 'leaf_data'

  inner = Workflow(name='inner', edges=[('START', leaf)])
  outer = Workflow(name='outer', edges=[('START', inner)])

  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hi'))
  out_events = _output_events(events)

  assert len(out_events) == 1
  assert out_events[0].output == 'leaf_data'
  assert out_events[0].node_info.path == 'outer@1/inner@1/leaf@1'


async def test_nested_with_output_schema_validates_at_read_time(
    request,
):
  """Nested workflow with output_schema validates without emitting extra events.

  Setup: outer → inner(output_schema=str) → leaf.
  Assert: 1 output event with validated data, no extra finalize event.
  """

  async def leaf(node_input: Any):
    return 'raw_data'

  inner = Workflow(
      name='inner',
      edges=[('START', leaf)],
      output_schema=str,
  )
  outer = Workflow(name='outer', edges=[('START', inner)])

  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hi'))
  out_events = _output_events(events)

  assert len(out_events) == 1
  assert out_events[0].output == 'raw_data'


async def test_multiple_terminals_in_nested_workflow_raises(
    request,
):
  """Fan-out with no join raises ValueError for multiple terminal outputs.

  Setup: outer → inner → (branch_a, branch_b).
  Assert: ValueError because inner has two terminal nodes producing output.
  """

  async def branch_a(node_input: Any):
    return 'a_out'

  async def branch_b(node_input: Any):
    return 'b_out'

  inner = Workflow(
      name='inner',
      edges=[('START', (branch_a, branch_b))],
  )
  outer = Workflow(name='outer', edges=[('START', inner)])

  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)

  with pytest.raises(ValueError, match='multiple terminal nodes'):
    await runner.run_async(testing_utils.get_user_content('hi'))


async def test_non_terminal_output_not_exposed_as_workflow_output(
    request,
):
  """Downstream node receives terminal output, not intermediate node output.

  Setup: outer → inner(step_a → step_b) → consume.
  Assert: consume receives step_b's 'final' (terminal), not step_a's
    'intermediate'.
  """

  async def step_a(node_input: Any):
    return 'intermediate'

  async def step_b(node_input: str):
    return 'final'

  inner = Workflow(name='inner', edges=[('START', step_a, step_b)])

  async def consume(node_input: str):
    return f'got: {node_input}'

  outer = Workflow(name='outer', edges=[('START', inner, consume)])

  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hi'))
  out_events = _output_events(events)

  consume_events = [
      e
      for e in out_events
      if e.node_info.name and e.node_info.name.split('@')[0] == 'consume'
  ]
  assert len(consume_events) == 1
  assert consume_events[0].output == 'got: final'
