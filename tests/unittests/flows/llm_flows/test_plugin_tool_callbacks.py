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

from typing import Any
from typing import Dict
from typing import Optional

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.functions import handle_function_calls_async
from google.adk.flows.llm_flows.functions import handle_function_calls_live
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.genai.errors import ClientError
import pytest

from ... import testing_utils

mock_error = ClientError(
    code=429,
    response_json={
        "error": {
            "code": 429,
            "message": "Quota exceeded.",
            "status": "RESOURCE_EXHAUSTED",
        }
    },
)


class MockPlugin(BasePlugin):
  before_tool_response = {"MockPlugin": "before_tool_response from MockPlugin"}
  after_tool_response = {"MockPlugin": "after_tool_response from MockPlugin"}
  on_tool_error_response = {
      "MockPlugin": "on_tool_error_response from MockPlugin"
  }

  def __init__(self, name="mock_plugin"):
    self.name = name
    self.enable_before_tool_callback = False
    self.enable_after_tool_callback = False
    self.enable_on_tool_error_callback = False

  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[dict]:
    if not self.enable_before_tool_callback:
      return None
    return self.before_tool_response

  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict,
  ) -> Optional[dict]:
    if not self.enable_after_tool_callback:
      return None
    return self.after_tool_response

  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> Optional[dict]:
    if not self.enable_on_tool_error_callback:
      return None
    return self.on_tool_error_response


@pytest.fixture
def mock_tool():
  def simple_fn(**kwargs) -> Dict[str, Any]:
    return {"initial": "response"}

  return FunctionTool(simple_fn)


@pytest.fixture
def mock_error_tool():
  def raise_error_fn(**kwargs) -> Dict[str, Any]:
    raise mock_error

  return FunctionTool(raise_error_fn)


@pytest.fixture
def mock_plugin():
  return MockPlugin()


async def invoke_tool_with_plugin(mock_tool, mock_plugin) -> Optional[Event]:
  """Invokes a tool with a plugin."""
  model = testing_utils.MockModel.create(responses=[])
  agent = Agent(
      name="agent",
      model=model,
      tools=[mock_tool],
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content="", plugins=[mock_plugin]
  )
  # Build function call event
  function_call = types.FunctionCall(name=mock_tool.name, args={})
  content = types.Content(parts=[types.Part(function_call=function_call)])
  event = Event(
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
      content=content,
  )
  tools_dict = {mock_tool.name: mock_tool}
  return await handle_function_calls_async(
      invocation_context,
      event,
      tools_dict,
  )


@pytest.mark.asyncio
async def test_async_before_tool_callback(mock_tool, mock_plugin):
  mock_plugin.enable_before_tool_callback = True

  result_event = await invoke_tool_with_plugin(mock_tool, mock_plugin)

  assert result_event is not None
  part = result_event.content.parts[0]
  assert part.function_response.response == mock_plugin.before_tool_response


@pytest.mark.asyncio
async def test_async_after_tool_callback(mock_tool, mock_plugin):
  mock_plugin.enable_after_tool_callback = True

  result_event = await invoke_tool_with_plugin(mock_tool, mock_plugin)

  assert result_event is not None
  part = result_event.content.parts[0]
  assert part.function_response.response == mock_plugin.after_tool_response


@pytest.mark.asyncio
async def test_async_on_tool_error_use_plugin_response(
    mock_error_tool, mock_plugin
):
  mock_plugin.enable_on_tool_error_callback = True

  result_event = await invoke_tool_with_plugin(mock_error_tool, mock_plugin)

  assert result_event is not None
  part = result_event.content.parts[0]
  assert part.function_response.response == mock_plugin.on_tool_error_response


@pytest.mark.asyncio
async def test_async_on_tool_error_fallback_to_runner(
    mock_error_tool, mock_plugin
):
  mock_plugin.enable_on_tool_error_callback = False

  try:
    await invoke_tool_with_plugin(mock_error_tool, mock_plugin)
  except Exception as e:
    assert e == mock_error


async def invoke_tool_with_plugin_live(
    mock_tool, mock_plugin
) -> Optional[Event]:
  """Invokes a tool with a plugin using the live path."""
  model = testing_utils.MockModel.create(responses=[])
  agent = Agent(
      name="agent",
      model=model,
      tools=[mock_tool],
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content="", plugins=[mock_plugin]
  )
  # Build function call event
  function_call = types.FunctionCall(name=mock_tool.name, args={})
  content = types.Content(parts=[types.Part(function_call=function_call)])
  event = Event(
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
      content=content,
  )
  tools_dict = {mock_tool.name: mock_tool}
  return await handle_function_calls_live(
      invocation_context,
      event,
      tools_dict,
  )


@pytest.mark.asyncio
async def test_live_before_tool_callback(mock_tool, mock_plugin):
  mock_plugin.enable_before_tool_callback = True

  result_event = await invoke_tool_with_plugin_live(mock_tool, mock_plugin)

  assert result_event is not None
  part = result_event.content.parts[0]
  assert part.function_response.response == mock_plugin.before_tool_response


@pytest.mark.asyncio
async def test_live_after_tool_callback(mock_tool, mock_plugin):
  mock_plugin.enable_after_tool_callback = True

  result_event = await invoke_tool_with_plugin_live(mock_tool, mock_plugin)

  assert result_event is not None
  part = result_event.content.parts[0]
  assert part.function_response.response == mock_plugin.after_tool_response


@pytest.mark.asyncio
async def test_live_on_tool_error_use_plugin_response(
    mock_error_tool, mock_plugin
):
  mock_plugin.enable_on_tool_error_callback = True

  result_event = await invoke_tool_with_plugin_live(
      mock_error_tool, mock_plugin
  )

  assert result_event is not None
  part = result_event.content.parts[0]
  assert part.function_response.response == mock_plugin.on_tool_error_response


@pytest.mark.asyncio
async def test_live_on_tool_error_fallback_to_runner(
    mock_error_tool, mock_plugin
):
  mock_plugin.enable_on_tool_error_callback = False

  try:
    await invoke_tool_with_plugin_live(mock_error_tool, mock_plugin)
  except Exception as e:
    assert e == mock_error


@pytest.mark.asyncio
async def test_live_plugin_before_tool_callback_takes_priority(
    mock_tool, mock_plugin
):
  """Plugin before_tool_callback should run before agent canonical callbacks."""
  mock_plugin.enable_before_tool_callback = True

  def agent_before_cb(tool, args, tool_context):
    return {"agent": "should_not_be_called"}

  model = testing_utils.MockModel.create(responses=[])
  agent = Agent(
      name="agent",
      model=model,
      tools=[mock_tool],
      before_tool_callback=agent_before_cb,
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content="", plugins=[mock_plugin]
  )
  function_call = types.FunctionCall(name=mock_tool.name, args={})
  content = types.Content(parts=[types.Part(function_call=function_call)])
  event = Event(
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
      content=content,
  )
  tools_dict = {mock_tool.name: mock_tool}
  result_event = await handle_function_calls_live(
      invocation_context, event, tools_dict
  )

  assert result_event is not None
  part = result_event.content.parts[0]
  # Plugin response should win, not the agent callback
  assert part.function_response.response == mock_plugin.before_tool_response


@pytest.mark.asyncio
async def test_live_plugin_after_tool_callback_takes_priority(
    mock_tool, mock_plugin
):
  """Plugin after_tool_callback should run before agent canonical callbacks."""
  mock_plugin.enable_after_tool_callback = True

  def agent_after_cb(tool, args, tool_context, tool_response):
    return {"agent": "should_not_be_called"}

  model = testing_utils.MockModel.create(responses=[])
  agent = Agent(
      name="agent",
      model=model,
      tools=[mock_tool],
      after_tool_callback=agent_after_cb,
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content="", plugins=[mock_plugin]
  )
  function_call = types.FunctionCall(name=mock_tool.name, args={})
  content = types.Content(parts=[types.Part(function_call=function_call)])
  event = Event(
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
      content=content,
  )
  tools_dict = {mock_tool.name: mock_tool}
  result_event = await handle_function_calls_live(
      invocation_context, event, tools_dict
  )

  assert result_event is not None
  part = result_event.content.parts[0]
  # Plugin response should win, not the agent callback
  assert part.function_response.response == mock_plugin.after_tool_response


if __name__ == "__main__":
  pytest.main([__file__])
