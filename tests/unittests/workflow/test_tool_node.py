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

"""Tests for ToolNode input parsing and execution."""

from typing import Any

from google.adk.events.event import Event
from google.adk.tools.base_tool import BaseTool
from google.adk.workflow import START
from google.adk.workflow._tool_node import _ToolNode as ToolNode
from google.adk.workflow._workflow import Workflow
from google.genai import types
import pytest

from . import workflow_testing_utils
from .. import testing_utils


class MockTool(BaseTool):
  """A mock tool that returns the args it was called with."""

  def __init__(self, name="mock_tool", description="Mock tool"):
    super().__init__(name=name, description=description)

  async def run_async(self, *, args: dict[str, Any], tool_context) -> Any:
    return args


async def _run_tool_node_wf(node_input: Any) -> list[Any]:
  """Runs a workflow with a ToolNode that receives node_input."""
  tool_node = ToolNode(tool=MockTool())

  def start_node():
    return Event(output=node_input)

  wf = Workflow(
      name="tool_node_test_wf",
      edges=[
          (START, start_node),
          (start_node, tool_node),
      ],
  )
  app_instance = testing_utils.App(name="test_app", root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app_instance)
  events = await runner.run_async("start")
  return workflow_testing_utils.simplify_events_with_node(events)


@pytest.mark.asyncio
async def test_tool_node_accepts_dict():
  """Tests that ToolNode accepts a dict as input and passes it to the tool."""
  input_dict = {"param_a": 1, "param_b": "value"}
  simplified = await _run_tool_node_wf(input_dict)
  assert (
      "tool_node_test_wf@1/mock_tool@1",
      {"output": input_dict},
  ) in simplified


@pytest.mark.asyncio
async def test_tool_node_accepts_none():
  """Tests that ToolNode accepts None, converting it to an empty dict."""
  simplified = await _run_tool_node_wf(None)
  assert ("tool_node_test_wf@1/mock_tool@1", {"output": {}}) in simplified


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_input", ["", "   ", "\n\t"])
async def test_tool_node_accepts_empty_string(empty_input):
  """Tests that ToolNode treats an empty/whitespace string as no arguments."""
  simplified = await _run_tool_node_wf(empty_input)
  assert ("tool_node_test_wf@1/mock_tool@1", {"output": {}}) in simplified


@pytest.mark.asyncio
async def test_tool_node_accepts_json_string():
  """Tests that ToolNode accepts a valid JSON string representing a dict."""
  json_str = '{"param_a": 1, "param_b": "value"}'
  simplified = await _run_tool_node_wf(json_str)
  assert (
      "tool_node_test_wf@1/mock_tool@1",
      {"output": {"param_a": 1, "param_b": "value"}},
  ) in simplified


@pytest.mark.asyncio
async def test_tool_node_accepts_content_with_json_string():
  """Tests that ToolNode accepts a types.Content containing a JSON string."""
  json_str = '{"param_a": 1, "param_b": "value"}'
  content = types.Content(
      parts=[types.Part.from_text(text=json_str)], role="user"
  )
  simplified = await _run_tool_node_wf(content)
  assert (
      "tool_node_test_wf@1/mock_tool@1",
      {"output": {"param_a": 1, "param_b": "value"}},
  ) in simplified


@pytest.mark.asyncio
async def test_tool_node_rejects_non_dict_json_string():
  """Tests that ToolNode raises TypeError if JSON string represents a non-dict (e.g. list)."""
  json_str = "[1, 2, 3]"
  with pytest.raises(
      TypeError, match="The input to ToolNode must be a dictionary"
  ):
    await _run_tool_node_wf(json_str)


@pytest.mark.asyncio
async def test_tool_node_rejects_invalid_json_string():
  """Tests that ToolNode raises TypeError if string input is not valid JSON."""
  invalid_str = "not a json"
  with pytest.raises(
      TypeError, match="The input to ToolNode must be a dictionary"
  ):
    await _run_tool_node_wf(invalid_str)


@pytest.mark.asyncio
async def test_tool_node_rejects_non_dict_content():
  """Tests that ToolNode raises TypeError if Content contains non-dict text."""
  content = types.Content(
      parts=[types.Part.from_text(text="not a json")], role="user"
  )
  with pytest.raises(
      TypeError, match="The input to ToolNode must be a dictionary"
  ):
    await _run_tool_node_wf(content)
