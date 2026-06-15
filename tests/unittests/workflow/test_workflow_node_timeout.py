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

"""Tests for node timeout behavior."""

from __future__ import annotations

import asyncio

from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import START
from google.adk.workflow._errors import NodeTimeoutError
from google.adk.workflow._node import node
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._workflow import Workflow
from google.genai import types
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_workflow(wf, message='start'):
  """Run a Workflow through Runner, return collected events."""
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)

  session = await ss.create_session(app_name='test', user_id='u')
  msg = types.Content(parts=[types.Part(text=message)], role='user')
  events = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)
  return events, ss, session


def _output_by_node(events):
  """Extract (node_name_from_path, output) for child node events."""
  results = []
  for e in events:
    if e.output is not None and e.node_info.path and '/' in e.node_info.path:
      node_name = e.node_info.path.rsplit('/', 1)[-1]
      if '@' in node_name:
        node_name = node_name.rsplit('@', 1)[0]
      results.append((node_name, e.output))
  return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_completes_within_timeout():
  """A node that finishes before the timeout should succeed normally."""

  @node(timeout=5.0)
  async def my_slow_node():
    await asyncio.sleep(0.01)
    return 'done'

  wf = Workflow(
      name='test_workflow',
      edges=[
          (START, my_slow_node),
      ],
  )
  events, _, _ = await _run_workflow(wf)
  by_node = _output_by_node(events)

  assert ('my_slow_node', 'done') in by_node


@pytest.mark.asyncio
async def test_node_exceeds_timeout():
  """A node that exceeds its timeout should fail."""

  from google.adk.workflow import FunctionNode

  async def raw_slow_func():
    await asyncio.sleep(1.0)
    return 'done'

  my_too_slow_node = FunctionNode(
      name='my_too_slow_node', func=raw_slow_func, timeout=0.05
  )

  wf = Workflow(
      name='test_workflow',
      edges=[
          (START, my_too_slow_node),
      ],
  )
  with pytest.raises(NodeTimeoutError) as exc_info:
    await _run_workflow(wf)

  assert 'my_too_slow_node' in str(exc_info.value)


@pytest.mark.asyncio
async def test_node_no_timeout():
  """A node with timeout=None should run without any time limit."""

  @node(timeout=None)
  async def my_no_timeout_node():
    await asyncio.sleep(0.01)
    return 'done'

  wf = Workflow(
      name='test_workflow',
      edges=[
          (START, my_no_timeout_node),
      ],
  )
  events, _, _ = await _run_workflow(wf)
  by_node = _output_by_node(events)

  assert ('my_no_timeout_node', 'done') in by_node


@pytest.mark.asyncio
async def test_node_timeout_with_retry():
  """A timed-out node should be retried if retry_config is set."""
  run_count = 0

  @node(
      timeout=0.05,
      retry_config=RetryConfig(max_attempts=3, initial_delay=0.0, jitter=0.0),
  )
  async def node_a():
    nonlocal run_count
    run_count += 1
    if run_count == 1:
      await asyncio.sleep(1.0)
    return 'success'

  wf = Workflow(
      name='test_workflow',
      edges=[
          (START, node_a),
      ],
  )
  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  # Verify that the final result was successfully obtained.
  assert ('node_a', 'success') in by_node

  # Verify that the node was actually executed more than once (i.e., retried).
  assert run_count == 2, f'Expected run_count == 2, got {run_count}'


@pytest.mark.asyncio
async def test_nested_workflow_timeout():
  """A nested workflow that exceeds its timeout in the outer workflow should fail.

  Setup: outer_wf -> inner_wf -> slow_node. inner_wf has timeout=0.05.
  Act: Run the outer workflow.
  Assert: Execution raises NodeTimeoutError referencing inner_wf.
  """
  import sys

  if sys.version_info < (3, 11):
    pytest.skip('asyncio.timeout requires Python 3.11+')

  # Given an outer workflow containing a slow inner workflow with a timeout
  @node()
  async def slow_node():
    await asyncio.sleep(1.0)
    return 'done'

  inner_wf = Workflow(
      name='inner_wf',
      edges=[(START, slow_node)],
      timeout=0.05,
  )

  outer_wf = Workflow(
      name='outer_wf',
      edges=[(START, inner_wf)],
  )

  # When the outer workflow is executed
  # Then it should raise NodeTimeoutError referencing the inner workflow
  with pytest.raises(NodeTimeoutError) as exc_info:
    await _run_workflow(outer_wf)

  assert 'inner_wf' in str(exc_info.value)
