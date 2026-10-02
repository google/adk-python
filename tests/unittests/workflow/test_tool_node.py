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
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.platform import uuid as platform_uuid
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.bash_tool import ExecuteBashTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.workflow import node
from google.adk.workflow import START
from google.adk.workflow._tool_node import _ToolNode as ToolNode
from google.adk.workflow._workflow import Workflow
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
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


@pytest.mark.asyncio
async def test_tool_node_failure_drops_confirmation_requested_before_it():
  """A tool that fails after requesting confirmation leaves no request pending.

  The on-tool-error answer is the call's result, so the after-tool callback
  sees no pending confirmation request and the workflow does not pause.
  """

  class _PendingRequestPlugin(_RecordingToolPlugin):

    def __init__(self):
      super().__init__(error_response={"error": "handled"})
      self.pending_after: list[dict[str, Any]] = []

    async def after_tool_callback(
        self, *, tool, tool_args, tool_context, result
    ):
      self.pending_after.append(
          dict(tool_context.actions.requested_tool_confirmations)
      )
      return await super().after_tool_callback(
          tool=tool,
          tool_args=tool_args,
          tool_context=tool_context,
          result=result,
      )

  def lookup(city: str, tool_context: Context) -> dict[str, str]:
    tool_context.request_confirmation(hint="Look up the city?")
    raise ValueError(f"no data for {city}")

  plugin = _PendingRequestPlugin()

  seen_downstream = await _run_tool_node_with_plugin(
      FunctionTool(func=lookup), plugin
  )

  assert plugin.pending_after == [{}]
  assert seen_downstream == [{"error": "handled"}]


class _ConfirmationWorkflow:
  """Runs start -> confirm-required tool node -> downstream with a runner."""

  def __init__(
      self,
      *,
      resumable: bool,
      require_confirmation: Any = True,
      plugins: list[BasePlugin] | None = None,
  ):
    self.tool_calls: list[str] = []
    self.seen_downstream: list[Any] = []

    def delete_db(name: str) -> dict[str, str]:
      self.tool_calls.append(name)
      return {"deleted": name}

    def start_node():
      return Event(output={"name": "prod"})

    def after(node_input: Any):
      self.seen_downstream.append(node_input)
      return node_input

    tool_node = ToolNode(
        tool=FunctionTool(
            func=delete_db, require_confirmation=require_confirmation
        )
    )
    wf = Workflow(
        name="tool_node_confirmation_wf",
        edges=[
            (START, start_node),
            (start_node, tool_node),
            (tool_node, after),
        ],
    )
    app_instance = testing_utils.App(
        name="test_app",
        root_agent=wf,
        plugins=plugins or [],
        resumability_config=(
            ResumabilityConfig(is_resumable=True) if resumable else None
        ),
    )
    self.runner = testing_utils.InMemoryRunner(app=app_instance)

  async def start(self) -> Event | None:
    """Runs the workflow and returns the confirmation request, if any."""
    events = await self.runner.run_async("start")
    return workflow_testing_utils.find_function_call_event(
        events, REQUEST_INPUT_FUNCTION_CALL_NAME
    )

  async def answer(self, request: Event, response: dict[str, Any]) -> None:
    """Resumes the workflow with the user's answer to the request."""
    interrupt_id = get_request_input_interrupt_ids(request)[0]
    await self.runner.run_async(
        new_message=testing_utils.UserContent(
            create_request_input_response(interrupt_id, response)
        ),
        invocation_id=request.invocation_id,
    )


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.asyncio
async def test_tool_node_confirmation_pauses_before_running_tool(
    resumable: bool,
):
  """Tests that a confirm-required tool asks the user before it runs."""
  wf = _ConfirmationWorkflow(resumable=resumable)

  request = await wf.start()

  assert request is not None
  args = request.content.parts[0].function_call.args
  assert args["payload"] == {"tool_name": "delete_db", "args": {"name": "prod"}}
  assert not wf.tool_calls
  assert not wf.seen_downstream


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.asyncio
async def test_tool_node_confirmation_approved_runs_tool_once(
    resumable: bool,
):
  """Tests that an approved call runs the tool once and continues."""
  wf = _ConfirmationWorkflow(resumable=resumable)
  request = await wf.start()

  await wf.answer(request, {"confirmed": True})

  assert wf.tool_calls == ["prod"]
  assert wf.seen_downstream == [{"deleted": "prod"}]


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.asyncio
async def test_tool_node_confirmation_rejected_skips_tool(resumable: bool):
  """Tests that a rejected call answers with an error and skips the tool."""
  wf = _ConfirmationWorkflow(resumable=resumable)
  request = await wf.start()

  await wf.answer(request, {"confirmed": False})

  assert not wf.tool_calls
  assert wf.seen_downstream == [{"error": "This tool call is rejected."}]


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.asyncio
async def test_tool_node_before_tool_callback_sees_confirmation_on_resume(
    resumable: bool,
):
  """The before-tool plugin sees the user's answer, as in an agent."""

  class _ConfirmationRecordingPlugin(BasePlugin):

    def __init__(self):
      super().__init__(name="confirmation_recording_plugin")
      self.seen: list[bool | None] = []

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
      confirmation = tool_context.tool_confirmation
      self.seen.append(confirmation.confirmed if confirmation else None)

  plugin = _ConfirmationRecordingPlugin()
  wf = _ConfirmationWorkflow(resumable=resumable, plugins=[plugin])
  request = await wf.start()

  await wf.answer(request, {"confirmed": True})

  assert plugin.seen == [None, True]
  assert wf.tool_calls == ["prod"]


@pytest.mark.asyncio
async def test_tool_node_confirmation_not_required_for_args_runs_directly():
  """Tests that a confirmation predicate returning False does not pause."""
  wf = _ConfirmationWorkflow(
      resumable=False, require_confirmation=lambda name: name != "prod"
  )

  request = await wf.start()

  assert request is None
  assert wf.tool_calls == ["prod"]
  assert wf.seen_downstream == [{"deleted": "prod"}]


class _RequestedConfirmationWorkflow(_ConfirmationWorkflow):
  """Runs a tool that calls `tool_context.request_confirmation()` itself."""

  def __init__(self, *, resumable: bool):
    self.tool_calls: list[str] = []
    self.seen_downstream: list[Any] = []

    def transfer(amount: int, tool_context: Context) -> dict[str, Any]:
      confirmation = tool_context.tool_confirmation
      if confirmation is None:
        tool_context.request_confirmation(
            hint="Approve the transfer?", payload={"limit": 0}
        )
        return {"status": "waiting for approval"}
      if not confirmation.confirmed:
        return {"status": "declined"}
      self.tool_calls.append(f"transfer {amount}")
      return {"transferred": amount, "limit": confirmation.payload["limit"]}

    def start_node():
      return Event(output={"amount": 5})

    def after(node_input: Any):
      self.seen_downstream.append(node_input)
      return node_input

    tool_node = ToolNode(tool=FunctionTool(func=transfer))
    wf = Workflow(
        name="tool_node_requested_confirmation_wf",
        edges=[
            (START, start_node),
            (start_node, tool_node),
            (tool_node, after),
        ],
    )
    app_instance = testing_utils.App(
        name="test_app",
        root_agent=wf,
        resumability_config=(
            ResumabilityConfig(is_resumable=True) if resumable else None
        ),
    )
    self.runner = testing_utils.InMemoryRunner(app=app_instance)


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.asyncio
async def test_tool_node_requested_confirmation_pauses(resumable: bool):
  """Tests that request_confirmation() inside a tool pauses the workflow."""
  wf = _RequestedConfirmationWorkflow(resumable=resumable)

  request = await wf.start()

  assert request is not None
  args = request.content.parts[0].function_call.args
  assert args["message"] == "Approve the transfer?"
  assert args["payload"] == {
      "tool_name": "transfer",
      "args": {"amount": 5},
      "confirmation_payload": {"limit": 0},
  }
  assert not wf.tool_calls
  assert not wf.seen_downstream


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.asyncio
async def test_tool_node_requested_confirmation_approved_reruns_tool(
    resumable: bool,
):
  """Tests that the tool reruns with the user's answer on ctx.tool_confirmation."""
  wf = _RequestedConfirmationWorkflow(resumable=resumable)
  request = await wf.start()

  await wf.answer(request, {"confirmed": True, "payload": {"limit": 10}})

  assert wf.tool_calls == ["transfer 5"]
  assert wf.seen_downstream == [{"transferred": 5, "limit": 10}]


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.asyncio
async def test_tool_node_requested_confirmation_rejected_lets_tool_decide(
    resumable: bool,
):
  """Tests that a rejection is handed to the tool, which decides the result."""
  wf = _RequestedConfirmationWorkflow(resumable=resumable)
  request = await wf.start()

  await wf.answer(request, {"confirmed": False})

  assert not wf.tool_calls
  assert wf.seen_downstream == [{"status": "declined"}]


def _make_runner(
    wf: Workflow, *, resumable: bool
) -> testing_utils.InMemoryRunner:
  """Returns a runner for the workflow, resumable or not."""
  app_instance = testing_utils.App(
      name="test_app",
      root_agent=wf,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  return testing_utils.InMemoryRunner(app=app_instance)


async def _start(runner: testing_utils.InMemoryRunner) -> Event | None:
  """Runs the workflow and returns the input request it paused on, if any."""
  events = await runner.run_async("start")
  return workflow_testing_utils.find_function_call_event(
      events, REQUEST_INPUT_FUNCTION_CALL_NAME
  )


async def _answer(
    runner: testing_utils.InMemoryRunner,
    request: Event,
    response: dict[str, Any],
) -> Event | None:
  """Resumes the workflow with the user's answer to the request.

  Returns the next input request the workflow paused on, if any.
  """
  interrupt_id = get_request_input_interrupt_ids(request)[0]
  events = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id, response)
      ),
      invocation_id=request.invocation_id,
  )
  return workflow_testing_utils.find_function_call_event(
      events, REQUEST_INPUT_FUNCTION_CALL_NAME
  )


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.asyncio
async def test_execute_bash_tool_node_runs_command_after_approval(
    tmp_path, resumable: bool
):
  """ExecuteBashTool in a tool node runs its command once the user approves.

  ExecuteBashTool asks for confirmation from inside `run_async` rather than
  through `check_require_confirmation`.
  """
  seen_downstream: list[Any] = []

  def start_node():
    return Event(output={"command": "echo hello_from_bash"})

  def after(node_input: Any):
    seen_downstream.append(node_input)
    return node_input

  tool_node = ToolNode(tool=ExecuteBashTool(workspace=tmp_path))
  wf = Workflow(
      name="tool_node_bash_wf",
      edges=[
          (START, start_node),
          (start_node, tool_node),
          (tool_node, after),
      ],
  )
  runner = _make_runner(wf, resumable=resumable)
  # Given the workflow paused on the bash command's confirmation request
  request = await _start(runner)
  assert request is not None
  assert not seen_downstream

  # When the user approves the command
  await _answer(runner, request, {"confirmed": True})

  # Then the command ran and its result reached the downstream node
  assert len(seen_downstream) == 1
  assert seen_downstream[0]["returncode"] == 0
  assert "hello_from_bash" in seen_downstream[0]["stdout"]


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.asyncio
async def test_tool_node_with_no_output_is_not_rerun_on_resume(
    resumable: bool,
):
  """A tool node whose tool returned None does not run again on resume."""
  tool_calls = 0

  def record_call(tool_context: Context) -> None:
    del tool_context  # Takes a context but never asks for confirmation.
    nonlocal tool_calls
    tool_calls += 1

  def start_node():
    return Event(output={})

  def review_node(node_input: Any):
    return RequestInput(interrupt_id="review", message="Please review")

  tool_node = ToolNode(tool=FunctionTool(func=record_call))
  wf = Workflow(
      name="tool_node_no_output_wf",
      edges=[
          (START, start_node),
          (start_node, tool_node),
          (tool_node, review_node),
      ],
  )
  runner = _make_runner(wf, resumable=resumable)
  # Given the tool ran and the workflow paused on a later node
  request = await _start(runner)
  assert request is not None

  # When the user answers the later node
  await _answer(runner, request, {"status": "ok"})

  # Then the tool ran only in the first turn
  assert tool_calls == 1


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.asyncio
async def test_confirmed_tool_node_with_no_output_is_not_rerun_on_resume(
    resumable: bool,
):
  """A confirmed tool call that returned None does not run again on resume.

  Setup: a confirm-required tool that returns None, followed by a node that
    asks for review in the same invocation.
  Act: approve the call, then answer the review.
  Assert: the tool ran only once, when the call was approved.
  """
  tool_calls = 0

  def record_call(name: str) -> None:
    del name  # Only the number of calls matters.
    nonlocal tool_calls
    tool_calls += 1

  def start_node():
    return Event(output={"name": "prod"})

  def review_node(node_input: Any):
    return RequestInput(interrupt_id="review", message="Please review")

  tool_node = ToolNode(
      tool=FunctionTool(func=record_call, require_confirmation=True)
  )
  wf = Workflow(
      name="confirmed_tool_node_no_output_wf",
      edges=[
          (START, start_node),
          (start_node, tool_node),
          (tool_node, review_node),
      ],
  )
  runner = _make_runner(wf, resumable=resumable)
  # Given the user approved the call and the workflow paused on the review
  confirmation = await _start(runner)
  assert confirmation is not None
  review = await _answer(runner, confirmation, {"confirmed": True})
  assert review is not None
  assert tool_calls == 1

  # When the user answers the review
  await _answer(runner, review, {"status": "ok"})

  # Then the tool did not run again
  assert tool_calls == 1


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.asyncio
async def test_dynamic_tool_node_with_no_output_is_not_rerun_on_resume(
    resumable: bool,
):
  """A tool node run through ctx.run_node() that returned None is not rerun.

  Setup: driver (rerun_on_resume=True) runs a no-output tool node through
    `ctx.run_node()`, then pauses for review.
  Act: answer the review, so the driver reruns and schedules the tool node
    again.
  Assert: the tool ran only in the first turn, and the driver finished.
  """
  tool_calls = 0
  driver_outputs: list[Any] = []

  def record_call(tool_context: Context) -> None:
    del tool_context  # Takes a context but never asks for confirmation.
    nonlocal tool_calls
    tool_calls += 1

  tool_node = ToolNode(tool=FunctionTool(func=record_call))

  @node(rerun_on_resume=True)
  async def driver(*, ctx: Context, node_input: Any):
    del node_input
    await ctx.run_node(tool_node)
    if "review" not in ctx.resume_inputs:
      yield RequestInput(interrupt_id="review", message="Please review")
      return
    driver_outputs.append("done")
    yield Event(output="done")

  wf = Workflow(name="dynamic_tool_node_no_output_wf", edges=[(START, driver)])
  runner = _make_runner(wf, resumable=resumable)
  request = await _start(runner)
  assert request is not None

  await _answer(runner, request, {"status": "ok"})

  assert tool_calls == 1
  assert driver_outputs == ["done"]


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.asyncio
async def test_tool_node_authenticated_function_tool_pause_and_resume(
    resumable: bool,
):
  """Tests that AuthenticatedFunctionTool in _ToolNode pauses for auth and resumes with credential."""
  from fastapi.openapi.models import APIKey
  from fastapi.openapi.models import APIKeyIn
  from google.adk.auth.auth_credential import AuthCredential
  from google.adk.auth.auth_credential import AuthCredentialTypes
  from google.adk.auth.auth_tool import AuthConfig
  from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool
  from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_CREDENTIAL_FUNCTION_CALL_NAME

  auth_config = AuthConfig(
      auth_scheme=APIKey(**{"in": APIKeyIn.header, "name": "X-Api-Key"}),
      credential_key="tool_node_api_key",
  )

  seen_credentials: list[str] = []
  seen_downstream: list[Any] = []
  plugin = _RecordingToolPlugin()

  def fetch_data(query: str, credential: AuthCredential) -> dict[str, str]:
    seen_credentials.append(credential.api_key)
    return {"query": query, "api_key": credential.api_key}

  def start_node():
    return Event(output={"query": "metrics"})

  def after(node_input: Any):
    seen_downstream.append(node_input)
    return node_input

  auth_tool = AuthenticatedFunctionTool(
      func=fetch_data,
      auth_config=auth_config,
  )
  tool_node = ToolNode(tool=auth_tool)
  wf = Workflow(
      name="tool_node_auth_wf",
      edges=[
          (START, start_node),
          (start_node, tool_node),
          (tool_node, after),
      ],
  )
  app_instance = testing_utils.App(
      name="test_app",
      root_agent=wf,
      plugins=[plugin],
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app_instance)

  # Turn 1: pauses for credential before calling fetch_data or after_tool_callback.
  events1 = await runner.run_async("start")
  auth_events = workflow_testing_utils.get_auth_request_events(events1)
  assert len(auth_events) == 1
  fc = auth_events[0].content.parts[0].function_call
  assert fc.name == REQUEST_CREDENTIAL_FUNCTION_CALL_NAME
  assert fc.id == "wf_auth:tool_node_auth_wf@1/fetch_data@1"
  assert not seen_credentials
  assert not seen_downstream
  assert plugin.calls == [("before", "fetch_data", {"query": "metrics"})]

  # Turn 2: supply the credential and verify tool executes and completes.
  auth_response = AuthConfig(
      auth_scheme=auth_config.auth_scheme,
      exchanged_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.API_KEY,
          api_key="secret_key_456",
      ),
      credential_key="tool_node_api_key",
  )
  resume_part = types.Part(
      function_response=types.FunctionResponse(
          id=fc.id,
          name=REQUEST_CREDENTIAL_FUNCTION_CALL_NAME,
          response=auth_response.model_dump(exclude_none=True, by_alias=True),
      )
  )
  await runner.run_async(
      new_message=testing_utils.UserContent(resume_part),
      invocation_id=auth_events[0].invocation_id,
  )

  assert seen_credentials == ["secret_key_456"]
  assert seen_downstream == [{"query": "metrics", "api_key": "secret_key_456"}]
  assert plugin.calls[-1] == (
      "after",
      "fetch_data",
      {"query": "metrics", "api_key": "secret_key_456"},
  )


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.asyncio
async def test_tool_node_dynamic_request_credential_pause_and_resume(
    resumable: bool,
):
  """Tests that a FunctionTool calling tool_context.request_credential pauses and resumes in _ToolNode."""
  from fastapi.openapi.models import APIKey
  from fastapi.openapi.models import APIKeyIn
  from google.adk.auth.auth_credential import AuthCredential
  from google.adk.auth.auth_credential import AuthCredentialTypes
  from google.adk.auth.auth_tool import AuthConfig
  from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_CREDENTIAL_FUNCTION_CALL_NAME

  auth_config = AuthConfig(
      auth_scheme=APIKey(**{"in": APIKeyIn.header, "name": "X-Api-Key"}),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.API_KEY,
          api_key="placeholder",
      ),
      credential_key="dynamic_tool_api_key",
  )

  seen_downstream: list[Any] = []

  def query_service(endpoint: str, tool_context) -> dict[str, str]:
    cred = tool_context.get_auth_response(auth_config)
    if cred is None:
      tool_context.request_credential(auth_config)
      return {"status": "auth_required"}
    return {"endpoint": endpoint, "token": cred.api_key}

  def start_node():
    return Event(output={"endpoint": "/v1/items"})

  def after(node_input: Any):
    seen_downstream.append(node_input)
    return node_input

  tool_node = ToolNode(tool=FunctionTool(func=query_service))
  wf = Workflow(
      name="tool_node_dyn_auth_wf",
      edges=[
          (START, start_node),
          (start_node, tool_node),
          (tool_node, after),
      ],
  )
  app_instance = testing_utils.App(
      name="test_app",
      root_agent=wf,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app_instance)

  # Turn 1: pauses with adk_request_credential and cleans up requested_auth_configs.
  events1 = await runner.run_async("start")
  auth_events = workflow_testing_utils.get_auth_request_events(events1)
  assert len(auth_events) == 1
  assert not auth_events[0].actions.requested_auth_configs
  fc = auth_events[0].content.parts[0].function_call
  assert fc.id == "wf_auth:tool_node_dyn_auth_wf@1/query_service@1"
  assert not seen_downstream

  # Turn 2: supply the credential and verify tool_context.get_auth_response succeeds.
  auth_response = AuthConfig(
      auth_scheme=auth_config.auth_scheme,
      raw_auth_credential=auth_config.raw_auth_credential,
      exchanged_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.API_KEY,
          api_key="dyn_secret_789",
      ),
      credential_key="dynamic_tool_api_key",
  )
  resume_part = types.Part(
      function_response=types.FunctionResponse(
          id=fc.id,
          name=REQUEST_CREDENTIAL_FUNCTION_CALL_NAME,
          response=auth_response.model_dump(exclude_none=True, by_alias=True),
      )
  )
  await runner.run_async(
      new_message=testing_utils.UserContent(resume_part),
      invocation_id=auth_events[0].invocation_id,
  )

  assert seen_downstream == [
      {"endpoint": "/v1/items", "token": "dyn_secret_789"}
  ]


@pytest.mark.asyncio
async def test_tool_node_malformed_auth_response_is_not_a_tool_error():
  """A malformed auth response fails the node instead of becoming a tool error.

  Like a malformed confirmation answer, it is handled before the tool runs, so
  the on-tool-error callback never sees it and cannot answer it.
  """
  from fastapi.openapi.models import APIKey
  from fastapi.openapi.models import APIKeyIn
  from google.adk.auth.auth_credential import AuthCredential
  from google.adk.auth.auth_tool import AuthConfig
  from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool
  from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_CREDENTIAL_FUNCTION_CALL_NAME

  auth_config = AuthConfig(
      auth_scheme=APIKey(**{"in": APIKeyIn.header, "name": "X-Api-Key"}),
      credential_key="tool_node_api_key",
  )
  tool_calls: list[str] = []
  plugin = _RecordingToolPlugin(error_response={"error": "handled"})

  def fetch_data(query: str, credential: AuthCredential) -> dict[str, str]:
    tool_calls.append(query)
    return {"query": query}

  def start_node():
    return Event(output={"query": "metrics"})

  tool_node = ToolNode(
      tool=AuthenticatedFunctionTool(func=fetch_data, auth_config=auth_config)
  )
  wf = Workflow(
      name="tool_node_malformed_auth_wf",
      edges=[(START, start_node), (start_node, tool_node)],
  )
  app_instance = testing_utils.App(
      name="test_app", root_agent=wf, plugins=[plugin]
  )
  runner = testing_utils.InMemoryRunner(app=app_instance)
  # Given the tool node paused for a credential
  auth_events = workflow_testing_utils.get_auth_request_events(
      await runner.run_async("start")
  )
  assert len(auth_events) == 1
  fc = auth_events[0].content.parts[0].function_call

  # When the client answers with something that is not a credential
  resume_part = types.Part(
      function_response=types.FunctionResponse(
          id=fc.id,
          name=REQUEST_CREDENTIAL_FUNCTION_CALL_NAME,
          response={"result": "not a credential"},
      )
  )

  # Then the node fails without running the tool or the error callback
  with pytest.raises(ValueError):
    await runner.run_async(
        new_message=testing_utils.UserContent(resume_part),
        invocation_id=auth_events[0].invocation_id,
    )
  assert not tool_calls
  assert not [call for call in plugin.calls if call[0] == "error"]


@pytest.mark.parametrize(
    "auth_response",
    [
        {"result": "some_api_key"},
        {
            "authScheme": {"type": "apiKey", "in": "header", "name": "X-Key"},
            "credentialKey": "client_chosen_key",
            "exchangedAuthCredential": {
                "authType": "apiKey",
                "apiKey": "some_api_key",
            },
        },
    ],
)
@pytest.mark.asyncio
async def test_tool_node_auth_resume_missing_auth_config_raises_workflow_data_error(
    auth_response: dict[str, Any],
):
  """Tests that resuming auth when AuthConfig cannot be resolved raises WorkflowDataError.

  A full auth config in the client's response is not used either, so the
  client cannot pick the state slot the credential is stored in.
  """
  from unittest.mock import MagicMock

  from google.adk.workflow._errors import WorkflowDataError

  class SimpleTool(BaseTool):

    def __init__(self):
      super().__init__(name="simple_tool", description="Simple tool")

    async def run_async(self, *, args: dict[str, Any], tool_context) -> Any:
      return "done"

  tool_node = ToolNode(tool=SimpleTool())
  ctx = MagicMock(spec=Context)
  ctx.node_path = "simple_tool@1"
  interrupt_id = "wf_auth:simple_tool@1"
  ctx.resume_inputs = {interrupt_id: auth_response}
  ctx.session = MagicMock()
  ctx.session.events = []

  with pytest.raises(WorkflowDataError, match="no AuthConfig found"):
    await tool_node._apply_auth_resume(ctx=ctx)


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.asyncio
async def test_tool_node_non_function_tool_no_output_fast_forwards_on_resume(
    resumable: bool,
):
  """Tests that a non-FunctionTool returning None fast-forwards and does not rerun on resume."""
  execution_count = 0

  class NoOutputTool(BaseTool):

    def __init__(self):
      super().__init__(
          name="no_output_tool", description="Tool producing no output"
      )

    async def run_async(self, *, args: dict[str, Any], tool_context) -> None:
      nonlocal execution_count
      execution_count += 1
      return None

  def start_node():
    return Event(output={})

  def pause_node(node_input: Any):
    return RequestInput(
        interrupt_id="pause_after_tool",
        message="pause",
        response_schema=dict,
    )

  tool_node = ToolNode(tool=NoOutputTool())
  wf = Workflow(
      name="no_output_wf",
      edges=[
          (START, start_node),
          (start_node, tool_node),
          (tool_node, pause_node),
      ],
  )
  app_instance = testing_utils.App(
      name="test_app",
      root_agent=wf,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app_instance)

  # Turn 1: tool runs once, returns None, then pause_node pauses.
  events1 = await runner.run_async("start")
  assert execution_count == 1
  request = workflow_testing_utils.find_function_call_event(
      events1, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  assert request is not None

  # Turn 2: resume pause_node. Tool should fast-forward and NOT re-execute.
  interrupt_id = get_request_input_interrupt_ids(request)[0]
  await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id, {"status": "ok"})
      ),
      invocation_id=request.invocation_id,
  )
  assert execution_count == 1
