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

"""Unit tests for the ActiveBackgroundTool model."""

import asyncio

from google.adk import live
from google.adk.live import LiveRequestQueue
from google.adk.live._active_background_tool import ActiveBackgroundTool
from pydantic import ValidationError
import pytest


async def _noop() -> None:
  pass


def test_active_background_tool_not_in_live_facade():
  """The internal model is not exported from the public live package."""
  assert 'ActiveBackgroundTool' not in live.__all__
  assert not hasattr(live, 'ActiveBackgroundTool')


async def test_active_background_tool_defaults():
  """An entry built from the required fields has no input stream and is not a generator."""
  task = asyncio.create_task(_noop())

  entry = ActiveBackgroundTool(
      tool_name='monitor', function_call_id='call-1', task=task
  )

  assert entry.tool_name == 'monitor'
  assert entry.function_call_id == 'call-1'
  assert entry.task is task
  assert entry.input_stream is None
  assert not entry.is_generator
  await task


async def test_active_background_tool_keeps_input_stream_and_generator_flag():
  """The input stream and generator flag are stored as given."""
  task = asyncio.create_task(_noop())
  input_stream = LiveRequestQueue()

  entry = ActiveBackgroundTool(
      tool_name='monitor',
      function_call_id='call-1',
      task=task,
      input_stream=input_stream,
      is_generator=True,
  )

  assert entry.input_stream is input_stream
  assert entry.is_generator
  await task


@pytest.mark.parametrize('missing', ['tool_name', 'function_call_id', 'task'])
async def test_active_background_tool_requires_identity_and_task(missing):
  """An entry cannot be built without its tool name, call id, or task."""
  task = asyncio.create_task(_noop())
  kwargs = {'tool_name': 'monitor', 'function_call_id': 'call-1', 'task': task}
  del kwargs[missing]

  with pytest.raises(ValidationError):
    ActiveBackgroundTool(**kwargs)
  await task


async def test_active_background_tool_rejects_non_task():
  """A plain coroutine is rejected where a task is expected."""
  coro = _noop()

  with pytest.raises(ValidationError):
    ActiveBackgroundTool(
        tool_name='monitor', function_call_id='call-1', task=coro
    )
  coro.close()


async def test_active_background_tool_extra_fields_forbidden():
  """Unknown fields are rejected."""
  task = asyncio.create_task(_noop())

  with pytest.raises(ValidationError):
    ActiveBackgroundTool(
        tool_name='monitor',
        function_call_id='call-1',
        task=task,
        unexpected_arg='not_allowed',
    )
  await task
