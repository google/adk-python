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
from google.adk.plugins.base_plugin import BasePlugin
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


class _ArtifactAndStateTool(BaseTool):
  """A tool that saves an artifact and writes state through its context."""

  def __init__(self):
    super().__init__(name="artifact_tool", description="Saves an artifact")

  async def run_async(self, *, args: dict[str, Any], tool_context) -> Any:
    await tool_context.save_artifact(
        "report.txt", types.Part.from_text(text="hello")
    )
    tool_context.state["report_status"] = "saved"
    return {"saved": True}


@pytest.mark.asyncio
async def test_tool_node_propagates_artifact_and_state_delta():
  """Tests that artifact and state deltas recorded by the tool are emitted."""
  seen_downstream: list[Any] = []

  def start_node():
    return Event(output={})

  async def after(ctx: Context, node_input: Any):
    artifact = await ctx.load_artifact("report.txt")
    seen_downstream.append(
        (node_input, ctx.state.get("report_status"), artifact.text)
    )
    return node_input

  tool_node = ToolNode(tool=_ArtifactAndStateTool())
  wf = Workflow(
      name="tool_node_artifact_wf",
      edges=[
          (START, start_node),
          (start_node, tool_node),
          (tool_node, after),
      ],
  )
  app_instance = testing_utils.App(name="test_app", root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app_instance)
  events = await runner.run_async("start")

  tool_events = [
      e
      for e in events
      if e.node_info.path == "tool_node_artifact_wf@1/artifact_tool@1"
  ]
  artifact_deltas = [e.actions.artifact_delta for e in tool_events]
  state_deltas = [e.actions.state_delta for e in tool_events]
  assert any("report.txt" in d for d in artifact_deltas), artifact_deltas
  assert any(
      d.get("report_status") == "saved" for d in state_deltas
  ), state_deltas
  assert seen_downstream == [({"saved": True}, "saved", "hello")]


class _RecordingToolPlugin(BasePlugin):
  """A plugin that records tool callbacks and can answer them."""

  def __init__(
      self,
      *,
      before_response: Any = None,
      after_response: Any = None,
      error_response: Any = None,
  ):
    super().__init__(name="recording_tool_plugin")
    self.calls: list[tuple[str, str, Any]] = []
    self._before_response = before_response
    self._after_response = after_response
    self._error_response = error_response

  async def before_tool_callback(self, *, tool, tool_args, tool_context):
    self.calls.append(("before", tool.name, dict(tool_args)))
    return self._before_response

  async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
    self.calls.append(("after", tool.name, result))
    return self._after_response

  async def on_tool_error_callback(
      self, *, tool, tool_args, tool_context, error
  ):
    self.calls.append(("error", tool.name, str(error)))
    return self._error_response


async def _run_tool_node_with_plugin(
    tool: BaseTool, plugin: BasePlugin
) -> list[Any]:
  """Runs start -> tool node -> downstream and returns downstream inputs."""
  seen_downstream: list[Any] = []

  def start_node():
    return Event(output={"city": "Paris"})

  def after(node_input: Any):
    seen_downstream.append(node_input)
    return node_input

  tool_node = ToolNode(tool=tool)
  wf = Workflow(
      name="tool_node_plugin_wf",
      edges=[
          (START, start_node),
          (start_node, tool_node),
          (tool_node, after),
      ],
  )
  app_instance = testing_utils.App(
      name="test_app", root_agent=wf, plugins=[plugin]
  )
  runner = testing_utils.InMemoryRunner(app=app_instance)
  await runner.run_async("start")
  return seen_downstream


@pytest.mark.asyncio
async def test_tool_node_runs_plugin_before_and_after_tool_callbacks():
  """Tests that plugins observe a tool node call like an agent tool call."""
  plugin = _RecordingToolPlugin()

  seen_downstream = await _run_tool_node_with_plugin(
      MockTool(name="lookup"), plugin
  )

  assert plugin.calls == [
      ("before", "lookup", {"city": "Paris"}),
      ("after", "lookup", {"city": "Paris"}),
  ]
  assert seen_downstream == [{"city": "Paris"}]


@pytest.mark.asyncio
async def test_tool_node_plugin_before_callback_short_circuits_tool():
  """Tests that a before-tool callback answer replaces the tool call."""
  tool_calls: list[Any] = []

  def lookup(city: str) -> dict[str, str]:
    tool_calls.append(city)
    return {"city": city}

  plugin = _RecordingToolPlugin(before_response={"cached": True})

  seen_downstream = await _run_tool_node_with_plugin(
      FunctionTool(func=lookup), plugin
  )

  assert not tool_calls
  assert plugin.calls[-1] == ("after", "lookup", {"cached": True})
  assert seen_downstream == [{"cached": True}]


@pytest.mark.asyncio
async def test_tool_node_plugin_after_callback_replaces_result():
  """Tests that an after-tool callback answer replaces the node output."""
  plugin = _RecordingToolPlugin(after_response={"redacted": True})

  seen_downstream = await _run_tool_node_with_plugin(
      MockTool(name="lookup"), plugin
  )

  assert seen_downstream == [{"redacted": True}]


@pytest.mark.asyncio
async def test_tool_node_plugin_on_tool_error_callback_handles_failure():
  """Tests that an on-tool-error callback answer becomes the node output."""

  def lookup(city: str) -> dict[str, str]:
    raise ValueError(f"no data for {city}")

  plugin = _RecordingToolPlugin(error_response={"error": "handled"})

  seen_downstream = await _run_tool_node_with_plugin(
      FunctionTool(func=lookup), plugin
  )

  assert ("error", "lookup", "no data for Paris") in plugin.calls
  assert seen_downstream == [{"error": "handled"}]


@pytest.mark.asyncio
async def test_tool_node_failure_propagates_when_no_plugin_handles_it():
  """Tests that a tool failure no plugin answers still fails the workflow."""

  def lookup(city: str) -> dict[str, str]:
    raise ValueError(f"no data for {city}")

  plugin = _RecordingToolPlugin()

  with pytest.raises(ValueError, match="no data for Paris"):
    await _run_tool_node_with_plugin(FunctionTool(func=lookup), plugin)
  assert ("error", "lookup", "no data for Paris") in plugin.calls
