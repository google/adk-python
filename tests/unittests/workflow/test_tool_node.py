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

import itertools
import re
from typing import Any

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.platform import uuid as platform_uuid
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.workflow import START
from google.adk.workflow._tool_node import _ToolNode as ToolNode
from google.adk.workflow._workflow import Workflow
from google.genai import types
from pydantic import BaseModel
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


@pytest.mark.asyncio
async def test_tool_node_function_call_id_uses_platform_id_provider():
  """Tests that the tool's function_call_id is minted via the platform seam.

  Frameworks that replay agent workflows (e.g. durable execution engines)
  install a deterministic id provider; the generated function_call_id must be
  stable across replays.
  """
  captured_ids: list[str] = []

  class CapturingTool(BaseTool):
    """A tool that records the function_call_id it was invoked with."""

    def __init__(self):
      super().__init__(name="capturing_tool", description="Captures ids")

    async def run_async(self, *, args: dict[str, Any], tool_context) -> Any:
      captured_ids.append(tool_context.function_call_id)
      return {}

  tool_node = ToolNode(tool=CapturingTool())

  def start_node():
    return Event(output={"param_a": 1})

  wf = Workflow(
      name="tool_node_id_wf",
      edges=[
          (START, start_node),
          (start_node, tool_node),
      ],
  )
  counter = itertools.count()
  platform_uuid.set_id_provider(lambda: f"fixed-{next(counter)}")
  try:
    app_instance = testing_utils.App(name="test_app", root_agent=wf)
    runner = testing_utils.InMemoryRunner(app=app_instance)
    await runner.run_async("start")
  finally:
    platform_uuid.reset_id_provider()

  assert len(captured_ids) == 1
  assert re.fullmatch(r"fixed-\d+", captured_ids[0])


class SampleInput(BaseModel):
  param_a: int
  param_b: str


class MockToolWithDeclaration(BaseTool):
  """A mock tool that exposes parameters in its FunctionDeclaration."""

  def __init__(
      self,
      name: str = "mock_tool_with_decl",
      param_names: tuple[str, ...] = ("param_a", "param_b"),
      required_param_names: tuple[str, ...] | None = None,
  ):
    super().__init__(name=name, description="Mock tool with declaration")
    self._param_names = param_names
    self._required_param_names = (
        param_names if required_param_names is None else required_param_names
    )

  def _get_declaration(self) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                p: types.Schema(type=types.Type.STRING)
                for p in self._param_names
            },
            required=list(self._required_param_names),
        ),
    )

  async def run_async(self, *, args: dict[str, Any], tool_context) -> Any:
    return args


@pytest.mark.asyncio
async def test_tool_node_accepts_pydantic_model():
  """Tests that ToolNode accepts a Pydantic BaseModel as input."""
  model_input = SampleInput(param_a=42, param_b="test")
  simplified = await _run_tool_node_wf(model_input)
  assert (
      "tool_node_test_wf@1/mock_tool@1",
      {"output": {"param_a": 42, "param_b": "test"}},
  ) in simplified


@pytest.mark.asyncio
async def test_tool_node_falls_back_to_ctx_state():
  """Tests that ToolNode falls back to ctx.state for missing declared parameters."""
  tool_node = ToolNode(
      tool=MockToolWithDeclaration(param_names=("city", "units"))
  )

  def start_node(ctx: Context):
    ctx.state["city"] = "Seattle"
    ctx.state["units"] = "metric"
    return Event(output=None)

  wf = Workflow(
      name="tool_node_state_fallback_wf",
      edges=[
          (START, start_node),
          (start_node, tool_node),
      ],
  )
  app_instance = testing_utils.App(name="test_app", root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app_instance)
  events = await runner.run_async("start")
  simplified = workflow_testing_utils.simplify_events_with_node(events)
  assert (
      "tool_node_state_fallback_wf@1/mock_tool_with_decl@1",
      {"output": {"city": "Seattle", "units": "metric"}},
  ) in simplified


@pytest.mark.asyncio
async def test_tool_node_prefers_node_input_over_ctx_state():
  """Tests that explicit node_input overrides ctx.state for declared parameters."""
  tool_node = ToolNode(
      tool=MockToolWithDeclaration(param_names=("city", "units"))
  )

  def start_node(ctx: Context):
    ctx.state["city"] = "Seattle"
    ctx.state["units"] = "metric"
    return Event(output={"city": "Tokyo"})

  wf = Workflow(
      name="tool_node_precedence_wf",
      edges=[
          (START, start_node),
          (start_node, tool_node),
      ],
  )
  app_instance = testing_utils.App(name="test_app", root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app_instance)
  events = await runner.run_async("start")
  simplified = workflow_testing_utils.simplify_events_with_node(events)
  assert (
      "tool_node_precedence_wf@1/mock_tool_with_decl@1",
      {"output": {"city": "Tokyo", "units": "metric"}},
  ) in simplified


@pytest.mark.asyncio
async def test_tool_node_falls_back_to_ctx_state_with_function_tool():
  """Tests that ToolNode falls back to ctx.state for required params with a FunctionTool."""

  def get_weather(city: str, units: str = "celsius") -> dict[str, str]:
    return {"city": city, "units": units}

  tool_node = ToolNode(tool=FunctionTool(func=get_weather))

  def start_node(ctx: Context):
    ctx.state["city"] = "Paris"
    ctx.state["units"] = "fahrenheit"
    return Event(output=None)

  wf = Workflow(
      name="tool_node_fn_tool_wf",
      edges=[
          (START, start_node),
          (start_node, tool_node),
      ],
  )
  app_instance = testing_utils.App(name="test_app", root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app_instance)
  events = await runner.run_async("start")
  simplified = workflow_testing_utils.simplify_events_with_node(events)
  assert (
      "tool_node_fn_tool_wf@1/get_weather@1",
      {"output": {"city": "Paris", "units": "celsius"}},
  ) in simplified


@pytest.mark.asyncio
async def test_tool_node_does_not_override_optional_parameters_with_ctx_state():
  """Tests that optional parameters not declared as required do not fall back to ctx.state."""
  tool_node = ToolNode(
      tool=MockToolWithDeclaration(
          param_names=("city", "units"),
          required_param_names=("city",),
      )
  )

  def start_node(ctx: Context):
    ctx.state["city"] = "Paris"
    ctx.state["units"] = "fahrenheit"
    return Event(output=None)

  wf = Workflow(
      name="tool_node_optional_state_wf",
      edges=[
          (START, start_node),
          (start_node, tool_node),
      ],
  )
  app_instance = testing_utils.App(name="test_app", root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app_instance)
  events = await runner.run_async("start")
  simplified = workflow_testing_utils.simplify_events_with_node(events)
  assert (
      "tool_node_optional_state_wf@1/mock_tool_with_decl@1",
      {"output": {"city": "Paris"}},
  ) in simplified
