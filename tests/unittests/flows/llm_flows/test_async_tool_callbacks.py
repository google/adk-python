# Copyright 2025 Google LLC
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

from enum import Enum
from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from unittest import mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.functions import handle_function_calls_async
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from ... import testing_utils


class CallbackType(Enum):
  SYNC = 1
  ASYNC = 2


class AsyncBeforeToolCallback:

  def __init__(self, mock_response: Dict[str, Any]):
    self.mock_response = mock_response

  async def __call__(
      self,
      tool: FunctionTool,
      args: Dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[Dict[str, Any]]:
    return self.mock_response


class AsyncAfterToolCallback:

  def __init__(self, mock_response: Dict[str, Any]):
    self.mock_response = mock_response

  async def __call__(
      self,
      tool: FunctionTool,
      args: Dict[str, Any],
      tool_context: ToolContext,
      tool_response: Dict[str, Any],
  ) -> Optional[Dict[str, Any]]:
    return self.mock_response


async def invoke_tool_with_callbacks(
    before_cb=None, after_cb=None
) -> Optional[Event]:
  def simple_fn(**kwargs) -> Dict[str, Any]:
    return {"initial": "response"}

  tool = FunctionTool(simple_fn)
  model = testing_utils.MockModel.create(responses=[])
  agent = Agent(
      name="agent",
      model=model,
      tools=[tool],
      before_tool_callback=before_cb,
      after_tool_callback=after_cb,
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=""
  )
  # Build function call event
  function_call = types.FunctionCall(name=tool.name, args={})
  content = types.Content(parts=[types.Part(function_call=function_call)])
  event = Event(
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
      content=content,
  )
  tools_dict = {tool.name: tool}
  return await handle_function_calls_async(
      invocation_context,
      event,
      tools_dict,
  )


@pytest.mark.asyncio
async def test_async_before_tool_callback():
  mock_resp = {"test": "before_tool_callback"}
  before_cb = AsyncBeforeToolCallback(mock_resp)
  result_event = await invoke_tool_with_callbacks(before_cb=before_cb)
  assert result_event is not None
  part = result_event.content.parts[0]
  assert part.function_response.response == mock_resp


@pytest.mark.asyncio
async def test_async_after_tool_callback():
  mock_resp = {"test": "after_tool_callback"}
  after_cb = AsyncAfterToolCallback(mock_resp)
  result_event = await invoke_tool_with_callbacks(after_cb=after_cb)
  assert result_event is not None
  part = result_event.content.parts[0]
  assert part.function_response.response == mock_resp


def mock_async_before_cb_side_effect(
    tool: FunctionTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    ret_value: Optional[Dict[str, Any]] = None,
):
  if ret_value:
    return ret_value
  return None


def mock_sync_before_cb_side_effect(
    tool: FunctionTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    ret_value: Optional[Dict[str, Any]] = None,
):
  if ret_value:
    return ret_value
  return None


async def mock_async_after_cb_side_effect(
    tool: FunctionTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Dict[str, Any],
    ret_value: Optional[Dict[str, Any]] = None,
):
  if ret_value:
    return ret_value
  return None


def mock_sync_after_cb_side_effect(
    tool: FunctionTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Dict[str, Any],
    ret_value: Optional[Dict[str, Any]] = None,
):
  if ret_value:
    return ret_value
  return None


CALLBACK_PARAMS = [
    pytest.param(
        [
            (None, CallbackType.SYNC),
            ({"test": "callback_2_response"}, CallbackType.ASYNC),
            ({"test": "callback_3_response"}, CallbackType.SYNC),
            (None, CallbackType.ASYNC),
        ],
        {"test": "callback_2_response"},
        [1, 1, 0, 0],
        id="middle_async_callback_returns",
    ),
    pytest.param(
        [
            (None, CallbackType.SYNC),
            (None, CallbackType.ASYNC),
            (None, CallbackType.SYNC),
            (None, CallbackType.ASYNC),
        ],
        {"initial": "response"},
        [1, 1, 1, 1],
        id="all_callbacks_return_none",
    ),
    pytest.param(
        [
            ({"test": "callback_1_response"}, CallbackType.SYNC),
            ({"test": "callback_2_response"}, CallbackType.ASYNC),
        ],
        {"test": "callback_1_response"},
        [1, 0],
        id="first_sync_callback_returns",
    ),
]


@pytest.mark.parametrize(
    "callbacks, expected_response, expected_calls",
    CALLBACK_PARAMS,
)
@pytest.mark.asyncio
async def test_before_tool_callbacks_chain(
    callbacks: List[tuple[Optional[Dict[str, Any]], int]],
    expected_response: Dict[str, Any],
    expected_calls: List[int],
):
  mock_before_cbs = []
  for response, callback_type in callbacks:
    if callback_type == CallbackType.ASYNC:
      mock_cb = mock.AsyncMock(
          side_effect=partial(
              mock_async_before_cb_side_effect, ret_value=response
          )
      )
    else:
      mock_cb = mock.Mock(
          side_effect=partial(
              mock_sync_before_cb_side_effect, ret_value=response
          )
      )
    mock_before_cbs.append(mock_cb)
  result_event = await invoke_tool_with_callbacks(before_cb=mock_before_cbs)
  assert result_event is not None
  part = result_event.content.parts[0]
  assert part.function_response.response == expected_response

  # Assert that the callbacks were called the expected number of times
  for i, mock_cb in enumerate(mock_before_cbs):
    expected_calls_count = expected_calls[i]
    if expected_calls_count == 1:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_awaited_once()
      else:
        mock_cb.assert_called_once()
    elif expected_calls_count == 0:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_not_awaited()
      else:
        mock_cb.assert_not_called()
    else:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_awaited(expected_calls_count)
      else:
        mock_cb.assert_called(expected_calls_count)


@pytest.mark.parametrize(
    "callbacks, expected_response, expected_calls",
    CALLBACK_PARAMS,
)
@pytest.mark.asyncio
async def test_after_tool_callbacks_chain(
    callbacks: List[tuple[Optional[Dict[str, Any]], int]],
    expected_response: Dict[str, Any],
    expected_calls: List[int],
):
  mock_after_cbs = []
  for response, callback_type in callbacks:
    if callback_type == CallbackType.ASYNC:
      mock_cb = mock.AsyncMock(
          side_effect=partial(
              mock_async_after_cb_side_effect, ret_value=response
          )
      )
    else:
      mock_cb = mock.Mock(
          side_effect=partial(
              mock_sync_after_cb_side_effect, ret_value=response
          )
      )
    mock_after_cbs.append(mock_cb)
  result_event = await invoke_tool_with_callbacks(after_cb=mock_after_cbs)
  assert result_event is not None
  part = result_event.content.parts[0]
  assert part.function_response.response == expected_response

  # Assert that the callbacks were called the expected number of times
  for i, mock_cb in enumerate(mock_after_cbs):
    expected_calls_count = expected_calls[i]
    if expected_calls_count == 1:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_awaited_once()
      else:
        mock_cb.assert_called_once()
    elif expected_calls_count == 0:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_not_awaited()
      else:
        mock_cb.assert_not_called()
    else:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_awaited(expected_calls_count)
      else:
        mock_cb.assert_called(expected_calls_count)
