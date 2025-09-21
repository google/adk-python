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

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.functions import handle_function_calls_live
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


async def invoke_tool_with_callbacks_live(
    before_cb=None, after_cb=None
) -> Optional[Event]:
  """Test helper to invoke a tool with callbacks using handle_function_calls_live."""

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
  return await handle_function_calls_live(
      invocation_context,
      event,
      tools_dict,
  )


def mock_sync_before_cb_side_effect(
    tool, args, tool_context, ret_value=None
) -> Optional[Dict[str, Any]]:
  return ret_value


async def mock_async_before_cb_side_effect(
    tool, args, tool_context, ret_value=None
) -> Optional[Dict[str, Any]]:
  return ret_value


def mock_sync_after_cb_side_effect(
    tool, args, tool_context, tool_response, ret_value=None
) -> Optional[Dict[str, Any]]:
  return ret_value


async def mock_async_after_cb_side_effect(
    tool, args, tool_context, tool_response, ret_value=None
) -> Optional[Dict[str, Any]]:
  return ret_value


@pytest.mark.asyncio
async def test_live_async_before_tool_callback():
  """Test that async before tool callbacks work in live mode."""
  mock_resp = {"test": "before_tool_callback"}
  before_cb = AsyncBeforeToolCallback(mock_resp)
  result_event = await invoke_tool_with_callbacks_live(before_cb=before_cb)
  assert result_event is not None
  part = result_event.content.parts[0]
  assert part.function_response.response == mock_resp


@pytest.mark.asyncio
async def test_live_async_after_tool_callback():
  """Test that async after tool callbacks work in live mode."""
  mock_resp = {"test": "after_tool_callback"}
  after_cb = AsyncAfterToolCallback(mock_resp)
  result_event = await invoke_tool_with_callbacks_live(after_cb=after_cb)
  assert result_event is not None
  part = result_event.content.parts[0]
  assert part.function_response.response == mock_resp


@pytest.mark.asyncio
async def test_live_sync_before_tool_callback():
  """Test that sync before tool callbacks work in live mode."""

  def sync_before_cb(tool, args, tool_context):
    return {"test": "sync_before_callback"}

  result_event = await invoke_tool_with_callbacks_live(before_cb=sync_before_cb)
  assert result_event is not None
  part = result_event.content.parts[0]
  assert part.function_response.response == {"test": "sync_before_callback"}


@pytest.mark.asyncio
async def test_live_sync_after_tool_callback():
  """Test that sync after tool callbacks work in live mode."""

  def sync_after_cb(tool, args, tool_context, tool_response):
    return {"test": "sync_after_callback"}

  result_event = await invoke_tool_with_callbacks_live(after_cb=sync_after_cb)
  assert result_event is not None
  part = result_event.content.parts[0]
  assert part.function_response.response == {"test": "sync_after_callback"}


# Test parameters for callback chains
CALLBACK_PARAMS = [
    # Test single sync callback returning None (should allow tool execution)
    ([(None, CallbackType.SYNC)], {"initial": "response"}, [1]),
    # Test single async callback returning None (should allow tool execution)
    ([(None, CallbackType.ASYNC)], {"initial": "response"}, [1]),
    # Test single sync callback returning response (should skip tool execution)
    ([({}, CallbackType.SYNC)], {}, [1]),
    # Test single async callback returning response (should skip tool execution)
    ([({}, CallbackType.ASYNC)], {}, [1]),
    # Test callback chain where an empty dict from the first callback doesn't
    # stop the chain, allowing the second callback to execute.
    (
        [({}, CallbackType.SYNC), ({"second": "callback"}, CallbackType.ASYNC)],
        {"second": "callback"},
        [1, 1],
    ),
    # Test callback chain where first returns None, second returns response
    (
        [(None, CallbackType.SYNC), ({}, CallbackType.ASYNC)],
        {},
        [1, 1],
    ),
    # Test mixed sync/async chain where all return None
    (
        [(None, CallbackType.SYNC), (None, CallbackType.ASYNC)],
        {"initial": "response"},
        [1, 1],
    ),
]


@pytest.mark.parametrize(
    "callbacks, expected_response, expected_calls",
    CALLBACK_PARAMS,
)
@pytest.mark.asyncio
async def test_live_before_tool_callbacks_chain(
    callbacks: List[tuple[Optional[Dict[str, Any]], int]],
    expected_response: Dict[str, Any],
    expected_calls: List[int],
):
  """Test that before tool callback chains work correctly in live mode."""
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

  result_event = await invoke_tool_with_callbacks_live(
      before_cb=mock_before_cbs
  )
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
async def test_live_after_tool_callbacks_chain(
    callbacks: List[tuple[Optional[Dict[str, Any]], int]],
    expected_response: Dict[str, Any],
    expected_calls: List[int],
):
  """Test that after tool callback chains work correctly in live mode."""
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

  result_event = await invoke_tool_with_callbacks_live(after_cb=mock_after_cbs)
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


@pytest.mark.asyncio
async def test_live_mixed_callbacks():
  """Test that both before and after callbacks work together in live mode."""

  def before_cb(tool, args, tool_context):
    # Modify args and let tool run
    args["modified_by_before"] = True
    return None

  def after_cb(tool, args, tool_context, tool_response):
    # Modify response
    tool_response["modified_by_after"] = True
    return tool_response

  result_event = await invoke_tool_with_callbacks_live(
      before_cb=before_cb, after_cb=after_cb
  )
  assert result_event is not None
  part = result_event.content.parts[0]
  response = part.function_response.response
  assert response["modified_by_after"] is True
  assert "initial" in response  # Original response should still be there


@pytest.mark.asyncio
async def test_live_callback_compatibility_with_async():
  """Test that live callbacks have the same behavior as async callbacks."""
  # This test ensures that the behavior between handle_function_calls_async
  # and handle_function_calls_live is consistent for callbacks

  def before_cb(tool, args, tool_context):
    return {"bypassed": "by_before_callback"}

  # Test with async version
  from google.adk.flows.llm_flows.functions import handle_function_calls_async

  def simple_fn(**kwargs) -> Dict[str, Any]:
    return {"initial": "response"}

  tool = FunctionTool(simple_fn)
  model = testing_utils.MockModel.create(responses=[])
  agent = Agent(
      name="agent",
      model=model,
      tools=[tool],
      before_tool_callback=before_cb,
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=""
  )
  function_call = types.FunctionCall(name=tool.name, args={})
  content = types.Content(parts=[types.Part(function_call=function_call)])
  event = Event(
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
      content=content,
  )
  tools_dict = {tool.name: tool}

  # Get result from async version
  async_result = await handle_function_calls_async(
      invocation_context, event, tools_dict
  )

  # Get result from live version
  live_result = await handle_function_calls_live(
      invocation_context, event, tools_dict
  )

  # Both should have the same response
  assert async_result is not None
  assert live_result is not None
  async_response = async_result.content.parts[0].function_response.response
  live_response = live_result.content.parts[0].function_response.response
  assert async_response == live_response == {"bypassed": "by_before_callback"}
