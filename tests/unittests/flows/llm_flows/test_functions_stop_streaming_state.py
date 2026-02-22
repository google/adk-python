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

import asyncio
from types import SimpleNamespace

from google.adk.agents.active_streaming_tool import ActiveStreamingTool
from google.adk.flows.llm_flows import functions
from google.genai import types
import pytest


async def _infinite_stream() -> None:
  while True:
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_stop_streaming_persists_cancelled_state_atomically():
  task = asyncio.create_task(_infinite_stream())
  invocation_context = SimpleNamespace(
      active_streaming_tools={
          'monitor_stock_price': ActiveStreamingTool(task=task)
      }
  )
  tool_context = SimpleNamespace(state={})
  streaming_lock = asyncio.Lock()

  function_response = await functions._process_function_live_helper(
      tool=SimpleNamespace(name='stop_streaming'),
      tool_context=tool_context,
      function_call=types.FunctionCall(
          name='stop_streaming',
          args={'function_name': 'monitor_stock_price'},
      ),
      function_args={'function_name': 'monitor_stock_price'},
      invocation_context=invocation_context,
      streaming_lock=streaming_lock,
  )

  assert function_response == {
      'status': 'Successfully stopped streaming function monitor_stock_price'
  }
  assert (
      tool_context.state[functions.LONG_RUNNING_CANCELLATION_STATE_KEY][
          'monitor_stock_price'
      ]
      == 'cancelled'
  )


@pytest.mark.asyncio
async def test_stop_streaming_persists_not_found_state():
  invocation_context = SimpleNamespace(active_streaming_tools={})
  tool_context = SimpleNamespace(state={})
  streaming_lock = asyncio.Lock()

  function_response = await functions._process_function_live_helper(
      tool=SimpleNamespace(name='stop_streaming'),
      tool_context=tool_context,
      function_call=types.FunctionCall(
          name='stop_streaming',
          args={'function_name': 'missing_stream'},
      ),
      function_args={'function_name': 'missing_stream'},
      invocation_context=invocation_context,
      streaming_lock=streaming_lock,
  )

  assert function_response == {
      'status': 'No active streaming function named missing_stream found'
  }
  assert (
      tool_context.state[functions.LONG_RUNNING_CANCELLATION_STATE_KEY][
          'missing_stream'
      ]
      == 'not_found'
  )
