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

"""Unit tests for ActiveStreamingTool model."""

import asyncio

from google.adk.agents.active_streaming_tool import ActiveStreamingTool as CompatActiveStreamingTool
from google.adk.live import LiveRequestQueue
import google.adk.live as live
from google.adk.live._active_streaming_tool import ActiveStreamingTool
from pydantic import ValidationError
import pytest


def test_active_streaming_tool_backward_compat_identity():
  """Verifies that the backward compatibility export is the exact same class."""
  assert CompatActiveStreamingTool is ActiveStreamingTool


def test_active_streaming_tool_not_in_live_facade():
  """Verifies that internal runtime model is not exported in live public facade."""
  assert "ActiveStreamingTool" not in live.__all__
  assert not hasattr(live, "ActiveStreamingTool")


def test_active_streaming_tool_defaults():
  """Verifies default values are None."""
  tool = ActiveStreamingTool()
  assert tool.task is None
  assert tool.stream is None


@pytest.mark.asyncio
async def test_active_streaming_tool_with_task_and_stream():
  """Verifies assignment of task and LiveRequestQueue stream."""

  async def _dummy():
    pass

  task = asyncio.create_task(_dummy())
  queue = LiveRequestQueue()
  tool = ActiveStreamingTool(task=task, stream=queue)

  assert tool.task is task
  assert tool.stream is queue
  assert tool._active_tasks() == {task}
  assert tool._active_streams() == [queue]
  await task


def test_active_streaming_tool_extra_fields_forbidden():
  """Verifies that extra attributes are rejected by pydantic configuration."""
  with pytest.raises(ValidationError):
    ActiveStreamingTool(unexpected_arg="not_allowed")


@pytest.mark.asyncio
async def test_active_streaming_tool_tracks_concurrent_calls():
  """Tracks independent streams and releases each completed call."""
  release = asyncio.Event()

  async def _wait():
    await release.wait()

  first_task = asyncio.create_task(_wait())
  second_task = asyncio.create_task(_wait())
  first_stream = LiveRequestQueue()
  second_stream = LiveRequestQueue()
  tool = ActiveStreamingTool()
  tool._track_task(first_task, first_stream)
  tool._track_task(second_task, second_stream)

  assert tool._active_tasks() == {first_task, second_task}
  assert tool._active_streams() == [first_stream, second_stream]

  second_task.cancel()
  await asyncio.gather(second_task, return_exceptions=True)
  await asyncio.sleep(0)
  assert tool._active_tasks() == {first_task}
  assert tool.task is first_task
  assert tool.stream is first_stream

  release.set()
  await first_task
  await asyncio.sleep(0)
  assert tool._active_tasks() == set()
  assert tool._active_streams() == []
  assert tool.task is None
  assert tool.stream is None


@pytest.mark.asyncio
async def test_discard_snapshot_preserves_later_call():
  """Discarding a stop snapshot does not remove a later registration."""
  first_task = asyncio.create_task(asyncio.sleep(60))
  second_task = asyncio.create_task(asyncio.sleep(60))
  tool = ActiveStreamingTool()
  tool._track_task(first_task)
  snapshot = tool._active_tasks()
  tool._track_task(second_task)

  try:
    tool._discard_tasks(snapshot)
    assert tool._active_tasks() == {second_task}
    assert tool.task is second_task
  finally:
    first_task.cancel()
    second_task.cancel()
    await asyncio.gather(first_task, second_task, return_exceptions=True)
