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
  await task


def test_active_streaming_tool_extra_fields_forbidden():
  """Verifies that extra attributes are rejected by pydantic configuration."""
  with pytest.raises(ValidationError):
    ActiveStreamingTool(unexpected_arg="not_allowed")
