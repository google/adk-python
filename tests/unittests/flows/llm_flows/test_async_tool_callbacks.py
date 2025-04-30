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

from typing import Any, Dict, Optional

import pytest

from google.adk.agents import Agent
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.flows.llm_flows.functions import handle_function_calls_async
from google.adk.events.event import Event
from google.genai import types

from ... import utils


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
  model = utils.MockModel.create(responses=[])
  agent = Agent(
      name="agent",
      model=model,
      tools=[tool],
      before_tool_callback=before_cb,
      after_tool_callback=after_cb,
  )
  invocation_context = utils.create_invocation_context(
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
