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

"""Unit tests for _batch_tool_executor."""

from __future__ import annotations

import asyncio

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows import _batch_tool_executor
from google.adk.tools.base_tool import BaseTool
from google.genai import types
import pytest


def test_deep_merge_dicts_basic() -> None:
  d1 = {'a': 1, 'nested': {'x': 10}}
  d2 = {'b': 2, 'nested': {'y': 20}}
  merged = _batch_tool_executor.deep_merge_dicts(d1, d2)
  assert merged == {'a': 1, 'b': 2, 'nested': {'x': 10, 'y': 20}}


def test_deep_merge_dicts_overwrite() -> None:
  d1 = {'a': 1, 'nested': {'x': 10}}
  d2 = {'a': 99, 'nested': {'x': 20}}
  merged = _batch_tool_executor.deep_merge_dicts(d1, d2)
  assert merged == {'a': 99, 'nested': {'x': 20}}


def test_merge_parallel_function_response_events_empty() -> None:
  with pytest.raises(ValueError, match='No function response events provided.'):
    _batch_tool_executor.merge_parallel_function_response_events([])


def test_merge_parallel_function_response_events_single() -> None:
  event = Event(
      invocation_id='inv-1',
      author='agent',
      content=types.Content(
          role='user',
          parts=[types.Part.from_text(text='result')],
      ),
  )
  merged = _batch_tool_executor.merge_parallel_function_response_events([event])
  assert merged is event


def test_merge_parallel_function_response_events_multiple() -> None:
  event1 = Event(
      invocation_id='inv-1',
      author='agent',
      branch='main',
      content=types.Content(
          role='user',
          parts=[types.Part.from_text(text='part1')],
      ),
      actions=EventActions(state_delta={'key1': 'val1'}),
  )
  event2 = Event(
      invocation_id='inv-1',
      author='agent',
      branch='main',
      content=types.Content(
          role='user',
          parts=[types.Part.from_text(text='part2')],
      ),
      actions=EventActions(state_delta={'key2': 'val2'}),
  )
  merged = _batch_tool_executor.merge_parallel_function_response_events(
      [event1, event2]
  )
  assert merged.invocation_id == 'inv-1'
  assert merged.author == 'agent'
  assert merged.content is not None
  assert merged.content.parts is not None
  assert len(merged.content.parts) == 2
  assert merged.content.parts[0].text == 'part1'
  assert merged.content.parts[1].text == 'part2'
  assert merged.actions.state_delta == {'key1': 'val1', 'key2': 'val2'}


def test_is_non_blocking_tool() -> None:
  assert not _batch_tool_executor._is_non_blocking_tool(None)

  tool_without_scheduling = BaseTool(name='t1', description='desc')
  assert not _batch_tool_executor._is_non_blocking_tool(tool_without_scheduling)

  tool_with_scheduling = BaseTool(
      name='t2',
      description='desc',
      response_scheduling=types.FunctionResponseScheduling.WHEN_IDLE,
  )
  assert _batch_tool_executor._is_non_blocking_tool(tool_with_scheduling)


@pytest.mark.asyncio
async def test_gather_or_cancel_success() -> None:
  async def worker(n: int) -> int:
    return n * 2

  tasks = [asyncio.create_task(worker(i)) for i in range(3)]
  results = await _batch_tool_executor._gather_or_cancel(tasks)
  assert results == [0, 2, 4]


@pytest.mark.asyncio
async def test_gather_or_cancel_cancels_siblings_on_failure() -> None:
  cancelled = False

  async def failing_worker() -> None:
    await asyncio.sleep(0.01)
    raise RuntimeError('boom')

  async def slow_worker() -> None:
    nonlocal cancelled
    try:
      await asyncio.sleep(10)
    except asyncio.CancelledError:
      cancelled = True
      raise

  tasks = [
      asyncio.create_task(failing_worker()),
      asyncio.create_task(slow_worker()),
  ]
  with pytest.raises(RuntimeError, match='boom'):
    await _batch_tool_executor._gather_or_cancel(tasks)

  assert cancelled
