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

"""Unit tests for flows.llm_flows.tools._live_caller."""

from __future__ import annotations

import inspect
from unittest import mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.flows.llm_flows.tools import _live_caller
from google.adk.live.live_request_queue import LiveRequestQueue
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest


def test_is_live_request_queue_annotation_matches_class_and_string() -> None:
  """Both LiveRequestQueue class and string annotations are recognized."""

  def fn_with_queue(input_stream: LiveRequestQueue, other: int) -> None:
    del input_stream, other

  params = inspect.signature(fn_with_queue).parameters
  assert (
      _live_caller._is_live_request_queue_annotation(params['input_stream'])
      is True
  )
  assert (
      _live_caller._is_live_request_queue_annotation(params['other']) is False
  )


@pytest.mark.asyncio
async def test_emit_streaming_tool_event_enqueues_user_message() -> None:
  """Events yielded by a streaming tool are enqueued as user-role messages on its tool@call_id branch."""
  tool = BaseTool(name='stream_tool', description='desc')
  tool_context = mock.create_autospec(ToolContext, instance=True)
  tool_context.function_call_id = 'fc-99'
  invocation_context = mock.create_autospec(InvocationContext, instance=True)
  invocation_context.invocation_id = 'inv-1'
  invocation_context.agent = mock.Mock()
  invocation_context.agent.name = 'stream_agent'
  invocation_context._enqueue_event = mock.AsyncMock()

  raw_event = Event(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text='progress update')]
      )
  )
  await _live_caller._emit_streaming_tool_event(
      raw_event,
      tool=tool,
      tool_context=tool_context,
      invocation_context=invocation_context,
  )

  invocation_context._enqueue_event.assert_awaited_once()
  enqueued = invocation_context._enqueue_event.call_args.args[0]
  assert isinstance(enqueued, Event)
  assert enqueued.branch == 'stream_tool@fc-99'
  assert enqueued.content is not None
  assert enqueued.content.role == 'user'
