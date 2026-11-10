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

"""Tests for @node decorator and behavior."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest import mock

from google.adk.agents import context as context_lib
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps import App
from google.adk.events._interrupts import extract_event_interrupt_ids
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.tools.base_tool import BaseTool
from google.adk.utils.context_utils import Aclosing
from google.adk.workflow import FunctionNode
from google.adk.workflow import START
from google.adk.workflow._node import Node
from google.adk.workflow._node import node
from google.adk.workflow._parallel_worker import _ParallelWorker as ParallelWorker
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._tool_node import _ToolNode as ToolNode
from google.adk.workflow._workflow import Workflow
from google.genai import types
from pydantic import BaseModel
import pytest

from .. import testing_utils

ANY = mock.ANY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from .._invariants import InvariantPlugin


async def _run_workflow(wf, message="start", *, check_invariants: bool = True):
  """Run a Workflow through Runner, return collected events."""
  ss = InMemorySessionService()
  runner = Runner(app_name="test", node=wf, session_service=ss)
  if check_invariants:
    runner.plugin_manager.plugins.insert(0, InvariantPlugin())
  session = await ss.create_session(app_name="test", user_id="u")
  msg = types.Content(parts=[types.Part(text=message)], role="user")
  events = []
  async for event in runner.run_async(
      user_id="u", session_id=session.id, new_message=msg
  ):
    events.append(event)
  return events, ss, session


def _output_by_node(events):
  """Extract (node_name_from_path, output) for child node events."""
  results = []
  for e in events:
    if e.output is not None and e.node_info.path and "/" in e.node_info.path:
      node_name = e.node_info.path.rsplit("/", 1)[-1]
      if "@" in node_name:
        node_name = node_name.rsplit("@", 1)[0]
      results.append((node_name, e.output))
  return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_decorator():
  """Tests that @node decorator can wrap a function and override its name."""

  @node(name="decorated_node")
  def my_func():
    return "Hello from decorated_func"

  assert my_func.name == "decorated_node"

  wf = Workflow(
      name="test_agent",
      edges=[
          (START, my_func),
      ],
  )
  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  assert ("decorated_node", "Hello from decorated_func") in by_node


def test_node_parallel_worker_instance():
  """Tests that node() can wrap a node in ParallelWorker."""

  @node(parallel_worker=True)
  def my_func(node_input):
    return node_input

  assert isinstance(my_func, ParallelWorker)
  assert my_func.name == "my_func"

  def other_func(x):
    return x

  parallel_node = node(other_func, parallel_worker=True)
  assert isinstance(parallel_node, ParallelWorker)
  assert parallel_node.name == "other_func"


@pytest.mark.asyncio
async def test_node_parallel_worker_execution():
  """Tests that a node with parallel_worker=True correctly processes inputs."""

  @node(parallel_worker=True)
  async def my_func(node_input):
    return node_input * 2

  async def producer_func() -> list[int]:
    return [1, 2, 3]

  wf = Workflow(
      name="test_agent",
      edges=[
          (START, producer_func),
          (producer_func, my_func),
      ],
  )
  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  assert ("producer_func", [1, 2, 3]) in by_node
  assert ("my_func", [2, 4, 6]) in by_node


def test_node_decorator_rerun_on_resume():
  """Tests that @node decorator can override rerun_on_resume."""

  @node(name="decorated_node", rerun_on_resume=True)
  def my_func():
    return "Hello from decorated_func"

  assert isinstance(my_func, FunctionNode)
  assert my_func.rerun_on_resume

  @node()
  def my_func2():
    return "Hello from decorated_func2"

  assert isinstance(my_func2, FunctionNode)
  assert not my_func2.rerun_on_resume


def test_node_decorator_parameter_binding():
  """Tests that @node decorator can configure parameter_binding."""

  @node(parameter_binding="node_input")
  def my_func(foo: str):
    return f"Hello {foo}"

  assert isinstance(my_func, FunctionNode)
  assert my_func.parameter_binding == "node_input"


@pytest.mark.asyncio
async def test_node_decorator_parameter_binding_execution():
  """Tests execution of a node with parameter_binding='node_input'."""

  async def producer_func() -> dict[str, Any]:
    return {"foo": "hello", "bar": 42}

  @node(parameter_binding="node_input")
  def my_func(foo: str, bar: int):
    return f"{foo}:{bar}"

  wf = Workflow(
      name="test_agent",
      edges=[
          (START, producer_func),
          (producer_func, my_func),
      ],
  )
  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  assert ("my_func", "hello:42") in by_node


def test_node_function_with_base_node():
  """Tests that node() function returns a copied node when given a BaseNode."""

  @node(name="original")
  def original():
    pass

  wrapped = node(original, name="overridden", rerun_on_resume=True)

  assert isinstance(wrapped, FunctionNode)
  assert wrapped is not original
  assert wrapped.name == "overridden"
  assert wrapped.rerun_on_resume


class MyTool(BaseTool):
  name = "tool"
  description = "desc"

  async def _run_async_impl(self):
    return "done"


def test_node_no_unnecessary_wrap():
  """Tests that node() does not wrap LlmAgent, Agent, Tool, or func in OverridingNode."""

  llm_agent = LlmAgent(name="llm")
  llm_node = node(llm_agent, name="overridden_llm")

  assert isinstance(llm_node, LlmAgent)
  assert llm_node.name == "overridden_llm"
  assert llm_node.mode == "single_turn"

  agent = BaseAgent(name="agent")
  agent_node_inst = node(agent, name="overridden_agent", rerun_on_resume=True)
  assert isinstance(agent_node_inst, BaseAgent)
  assert agent_node_inst.name == "overridden_agent"
  assert agent_node_inst.rerun_on_resume

  tool_inst = MyTool(name="tool", description="desc")
  t_node = node(tool_inst, name="overridden_tool")
  assert isinstance(t_node, ToolNode)
  assert t_node.name == "overridden_tool"

  def my_func():
    pass

  f_node = node(my_func, name="overridden_func", rerun_on_resume=True)
  assert isinstance(f_node, FunctionNode)
  assert f_node.name == "overridden_func"
  assert f_node.rerun_on_resume


class StatefulTool(BaseTool):
  """A tool that modifies state via tool_context."""

  async def run_async(self, *, args, tool_context):
    tool_context.state["tool_key"] = "tool_value"
    tool_context.state["tool_count"] = 10
    return {"status": "ok"}


from .workflow_testing_utils import simplify_events_with_node


@pytest.mark.asyncio
async def test_tool_node_state_delta():
  """Tests that state set via tool_context.state in ToolNode is persisted."""

  tool_node = ToolNode(
      tool=StatefulTool(name="stateful_tool", description="Sets state values"),
  )

  def read_state(tool_key: str, tool_count: int) -> str:
    return f"tool_key={tool_key}, tool_count={tool_count}"

  def start_node():
    return {}

  wf = Workflow(
      name="test_tool_node_state_delta",
      edges=[
          (START, start_node),
          (start_node, tool_node),
          (tool_node, read_state),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  simplified = simplify_events_with_node(
      events, include_workflow_output=True, include_state_delta=True
  )

  assert (
      "test_tool_node_state_delta@1/stateful_tool@1",
      {"output": {"status": "ok"}},
  ) in [(e[0], {"output": e[1].get("output")}) for e in simplified]

  assert (
      "test_tool_node_state_delta@1/read_state@1",
      {
          "output": "tool_key=tool_value, tool_count=10",
      },
  ) in [(e[0], {"output": e[1].get("output")}) for e in simplified]


class _CustomNode(Node):
  custom_val: str = "hello"
  rerun_on_resume: bool = True

  async def run_node_impl(self, *, ctx, node_input):
    yield f"subclass: {self.custom_val} -> {node_input}"


def test_node_subclassing_model_copy_preserves_identity():
  """Tests that Node.model_copy preserves the subclass class identity."""
  node_inst = _CustomNode(
      name="subclass", parallel_worker=True, custom_val="barrier"
  )
  assert node_inst.parallel_worker is True

  cloned = node_inst.model_copy()
  assert isinstance(cloned, _CustomNode)
  assert cloned.custom_val == "barrier"
  assert cloned.parallel_worker is True
  assert isinstance(cloned._inner_node, ParallelWorker)
  # Confirm inner node wraps a clone of _CustomNode preserving identity!
  assert isinstance(cloned._inner_node._node, _CustomNode)
  assert cloned._inner_node._node.custom_val == "barrier"
  assert cloned._inner_node._node.parallel_worker is False


@pytest.mark.asyncio
async def test_node_subclassing_execution_with_parallel_worker():
  """Tests that a subclassed Node with parallel_worker=True executes successfully."""
  subclass_node = _CustomNode(
      name="subclass", parallel_worker=True, custom_val="workflow"
  )

  async def producer():
    return ["input1", "input2"]

  wf = Workflow(
      name="test_agent",
      edges=[
          (START, producer),
          (producer, subclass_node),
      ],
  )

  events, _, _ = await _run_workflow(wf)
  by_node = _output_by_node(events)

  assert ("producer", ["input1", "input2"]) in by_node
  assert (
      "subclass",
      ["subclass: workflow -> input1", "subclass: workflow -> input2"],
  ) in by_node


def test_node_decorator_parallel_worker_max_parallel_workers():
  """Tests that node() correctly sets max_parallel_workers on ParallelWorker."""

  @node(parallel_worker=True, max_parallel_workers=3)
  def my_func(node_input):
    return node_input

  assert isinstance(my_func, ParallelWorker)
  assert my_func.max_parallel_workers == 3


def test_node_decorator_invalid_max_parallel_workers():
  """Tests that node() raises ValueError if max_parallel_workers is set without parallel_worker."""
  with pytest.raises(
      ValueError,
      match="max_parallel_workers can only be set when parallel_worker is True",
  ):

    @node(parallel_worker=False, max_parallel_workers=3)
    def my_func(node_input):
      return node_input


def test_node_subclass_invalid_max_parallel_workers():
  """Tests that Node subclass raises ValidationError if max_parallel_workers is set without parallel_worker."""
  from pydantic import ValidationError

  with pytest.raises(ValidationError) as exc_info:
    _CustomNode(name="subclass", parallel_worker=False, max_parallel_workers=3)

  assert (
      "max_parallel_workers can only be set when parallel_worker is True"
      in str(exc_info.value)
  )


def test_node_subclassing_model_copy_preserves_max_parallel_workers():
  """Tests that Node.model_copy preserves max_parallel_workers."""
  node_inst = _CustomNode(
      name="subclass",
      parallel_worker=True,
      max_parallel_workers=5,
      custom_val="barrier",
  )
  assert node_inst.parallel_worker is True
  assert node_inst.max_parallel_workers == 5
  assert node_inst._inner_node.max_parallel_workers == 5

  cloned = node_inst.model_copy()
  assert isinstance(cloned, _CustomNode)
  assert cloned.parallel_worker is True
  assert cloned.max_parallel_workers == 5
  assert isinstance(cloned._inner_node, ParallelWorker)
  assert cloned._inner_node.max_parallel_workers == 5
  assert cloned._inner_node._node.parallel_worker is False


def test_node_decorator_invalid_max_parallel_workers_less_than_one():
  """Tests that node() raises ValueError if max_parallel_workers is less than 1."""
  with pytest.raises(
      ValueError,
      match="max_parallel_workers must be greater than or equal to 1",
  ):

    @node(parallel_worker=True, max_parallel_workers=0)
    def my_func(node_input):
      return node_input


def test_node_subclass_invalid_max_parallel_workers_less_than_one():
  """Tests that Node subclass raises ValidationError if max_parallel_workers is less than 1."""
  from pydantic import ValidationError

  with pytest.raises(ValidationError) as exc_info:
    _CustomNode(name="subclass", parallel_worker=True, max_parallel_workers=0)

  assert "max_parallel_workers must be greater than or equal to 1" in str(
      exc_info.value
  )


def test_parallel_worker_invalid_max_parallel_workers_less_than_one():
  """Tests that ParallelWorker constructor raises ValueError if max_parallel_workers is less than 1."""

  def dummy_func(x):
    return x

  with pytest.raises(
      ValueError,
      match="max_parallel_workers must be greater than or equal to 1",
  ):
    ParallelWorker(node=dummy_func, max_parallel_workers=0)


def test_node_require_confirmation_configuration():
  """Tests @node(require_confirmation=...) defaults and configuration validation."""
  from google.adk.workflow._errors import WorkflowConfigurationError

  @node(require_confirmation=True)
  def remove_file():
    return "removed"

  assert isinstance(remove_file, FunctionNode)
  assert remove_file.require_confirmation is True
  assert remove_file.rerun_on_resume is True

  def check_fn(path: str) -> bool:
    return path.startswith("/prod")

  @node(require_confirmation=check_fn)
  def remove_path(path: str):
    return path

  assert remove_path.require_confirmation is check_fn
  assert remove_path.rerun_on_resume is True
  assert remove_path._as_tool_node().require_confirmation is check_fn

  for fn in (
      lambda: node(require_confirmation=True, rerun_on_resume=False)(
          lambda: None
      ),
      lambda: FunctionNode(
          func=lambda: "ok",
          name="direct_node",
          require_confirmation=True,
          rerun_on_resume=False,
      ),
      lambda: node(remove_file, rerun_on_resume=False),
  ):
    with pytest.raises(
        ValueError, match="require_confirmation requires rerun_on_resume=True"
    ):
      fn()

  agent = LlmAgent(name="test_agent")
  with pytest.raises(
      WorkflowConfigurationError,
      match="require_confirmation is only supported on FunctionNode",
  ):
    node(agent, require_confirmation=True)
  assert node(agent, require_confirmation=False) is not None


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.parametrize("confirmed", [True, False])
@pytest.mark.asyncio
async def test_node_require_confirmation_workflow_execution(
    resumable: bool, confirmed: bool
):
  """Tests @node(require_confirmation=True) pauses in Workflow and handles approval/rejection."""
  from google.adk.apps.app import ResumabilityConfig
  from google.adk.utils._function_call_names import REQUEST_INPUT_FUNCTION_CALL_NAME
  from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response

  from . import workflow_testing_utils

  calls: list[str] = []
  downstream_seen: list[object] = []

  class RemoveResult(BaseModel):
    removed: bool

  @node(require_confirmation=True)
  def remove_file() -> RemoveResult:
    calls.append("removed")
    return RemoveResult(removed=True)

  def after_node(node_input):
    downstream_seen.append(node_input)
    return node_input

  wf = Workflow(
      name="confirm_wf",
      edges=[(START, remove_file), (remove_file, after_node)],
  )
  app_instance = App(
      name="test_confirm_wf",
      root_agent=wf,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app_instance)

  events1 = await runner.run_async("start")
  req_event = workflow_testing_utils.find_function_call_event(
      events1, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  assert req_event is not None
  fc_args = req_event.content.parts[0].function_call.args
  assert fc_args["payload"] == {"tool_name": "remove_file", "args": {}}
  assert not calls
  assert not downstream_seen

  [interrupt_id] = extract_event_interrupt_ids(req_event)
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id, {"confirmed": confirmed})
      ),
      invocation_id=req_event.invocation_id,
  )

  if confirmed:
    assert calls == ["removed"]
    assert downstream_seen == [{"removed": True}]
  else:
    assert not calls
    assert downstream_seen == [{"error": "This tool call is rejected."}]
    assert any(
        e.output == {"error": "This tool call is rejected."} for e in events2
    )


@pytest.mark.parametrize(
    "target_path,should_pause", [("/dev/tmp", False), ("/prod/db", True)]
)
@pytest.mark.asyncio
async def test_node_require_confirmation_callable_predicate(
    target_path: str, should_pause: bool
):
  """Tests @node(require_confirmation=callable) evaluates predicate with bound arguments."""
  from google.adk.agents.context import Context
  from google.adk.utils._function_call_names import REQUEST_INPUT_FUNCTION_CALL_NAME
  from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response

  from . import workflow_testing_utils

  calls: list[str] = []
  predicate_calls: list[str] = []

  def setup_state(ctx: Context):
    ctx.state["path"] = target_path

  def check_path(path: str) -> str:
    predicate_calls.append(path)
    return path if path.startswith("/prod") else ""

  @node(require_confirmation=check_path)
  def remove_file(path: str):
    calls.append(path)
    return f"deleted:{path}"

  wf = Workflow(
      name="predicate_wf",
      edges=[(START, setup_state), (setup_state, remove_file)],
  )
  runner = testing_utils.InMemoryRunner(
      app=App(name="test_predicate_wf", root_agent=wf)
  )

  events1 = await runner.run_async("start")
  req_event = workflow_testing_utils.find_function_call_event(
      events1, REQUEST_INPUT_FUNCTION_CALL_NAME
  )

  if not should_pause:
    assert req_event is None
    assert predicate_calls == [target_path]
    assert calls == [target_path]
  else:
    assert req_event is not None
    assert req_event.content.parts[0].function_call.args["payload"] == {
        "tool_name": "remove_file",
        "args": {"path": target_path},
    }
    assert predicate_calls == [target_path]
    assert not calls
    [interrupt_id] = extract_event_interrupt_ids(req_event)
    await runner.run_async(
        new_message=testing_utils.UserContent(
            create_request_input_response(interrupt_id, {"confirmed": True})
        ),
        invocation_id=req_event.invocation_id,
    )
    assert predicate_calls == [target_path]
    assert calls == [target_path]


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.asyncio
async def test_node_require_confirmation_returning_none_not_rerun_downstream(
    resumable: bool,
):
  """Tests @node(require_confirmation=True) returning None does not rerun on downstream resume."""
  from google.adk.apps.app import ResumabilityConfig
  from google.adk.events.request_input import RequestInput
  from google.adk.utils._function_call_names import REQUEST_INPUT_FUNCTION_CALL_NAME
  from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response

  from . import workflow_testing_utils

  call_count = 0

  @node(require_confirmation=True)
  def remove_file() -> None:
    nonlocal call_count
    call_count += 1
    return None

  def review_step():
    return RequestInput(interrupt_id="review", message="Review?")

  wf = Workflow(
      name="confirm_none_wf",
      edges=[(START, remove_file), (remove_file, review_step)],
  )
  runner = testing_utils.InMemoryRunner(
      app=App(
          name="test_confirm_none_wf",
          root_agent=wf,
          resumability_config=(
              ResumabilityConfig(is_resumable=True) if resumable else None
          ),
      )
  )

  events1 = await runner.run_async("start")
  req1 = workflow_testing_utils.find_function_call_event(
      events1, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  assert req1 is not None
  assert call_count == 0

  # Approve remove_file -> runs once and pauses on review_step
  [id1] = extract_event_interrupt_ids(req1)
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(id1, {"confirmed": True})
      ),
      invocation_id=req1.invocation_id,
  )
  assert call_count == 1
  req2 = workflow_testing_utils.find_function_call_event(
      events2, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  assert req2 is not None

  # Resume review_step -> remove_file must not run again
  [id2] = extract_event_interrupt_ids(req2)
  await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(id2, {"ok": True})
      ),
      invocation_id=req2.invocation_id,
  )
  assert call_count == 1


@pytest.mark.parametrize("child_confirmed", [True, False])
@pytest.mark.asyncio
async def test_node_require_confirmation_child_run_node_not_skipped(
    child_confirmed: bool,
):
  """Tests child @node(require_confirmation=True) called via ctx.run_node handles approval and rejection."""
  from google.adk.agents.context import Context
  from google.adk.utils._function_call_names import REQUEST_INPUT_FUNCTION_CALL_NAME
  from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response

  from . import workflow_testing_utils

  calls: list[str] = []
  child_results: list[Any] = []

  @node(require_confirmation=True)
  def child_action() -> str:
    calls.append("child")
    return "child_done"

  @node(require_confirmation=True)
  async def parent_action(ctx: Context) -> Any:
    calls.append("parent")
    res = await ctx.run_node(child_action)
    child_results.append(res)
    return res

  wf = Workflow(name="nested_confirm_wf", edges=[(START, parent_action)])
  runner = testing_utils.InMemoryRunner(
      app=App(name="test_nested_confirm_wf", root_agent=wf)
  )

  # 1. Pauses on parent_action
  events1 = await runner.run_async("start")
  req1 = workflow_testing_utils.find_function_call_event(
      events1, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  assert req1 is not None
  assert not calls

  # 2. Approve parent_action -> parent runs and pauses on child_action
  [id1] = extract_event_interrupt_ids(req1)
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(id1, {"confirmed": True})
      ),
      invocation_id=req1.invocation_id,
  )
  assert calls == ["parent"]
  req2 = workflow_testing_utils.find_function_call_event(
      events2, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  assert req2 is not None

  # 3. Respond to child_action -> parent reruns to reach ctx.run_node
  [id2] = extract_event_interrupt_ids(req2)
  await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(id2, {"confirmed": child_confirmed})
      ),
      invocation_id=req2.invocation_id,
  )
  if child_confirmed:
    assert calls == ["parent", "parent", "child"]
    assert child_results == ["child_done"]
  else:
    assert calls == ["parent", "parent"]
    assert child_results == [{"error": "This tool call is rejected."}]


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.asyncio
async def test_node_require_confirmation_rejected_survives_replay_across_turns(
    resumable: bool,
):
  """Tests that a rejected node's output survives replay when a sibling branch resumes later."""
  from google.adk.apps.app import ResumabilityConfig
  from google.adk.events.request_input import RequestInput
  from google.adk.workflow._join_node import JoinNode
  from google.adk.workflow.utils._workflow_hitl_utils import confirmation_interrupt_id
  from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response

  node_b_calls: list[str] = []
  downstream_b_calls: list[Any] = []
  join_calls: list[Any] = []

  @node(rerun_on_resume=False)
  def node_a():
    return RequestInput(interrupt_id="ask_a", message="Input for A?")

  @node(require_confirmation=True)
  def node_b() -> str:
    node_b_calls.append("ran_b")
    return "result_b"

  @node
  def downstream_b(node_input: Any) -> str:
    downstream_b_calls.append(node_input)
    return "after_b"

  node_join = JoinNode(name="node_join")

  @node
  def downstream_join(node_input: Any) -> str:
    join_calls.append(node_input)
    return "done"

  wf = Workflow(
      name="replay_cancel_wf",
      edges=[
          (START, node_a),
          (START, node_b),
          (node_b, downstream_b),
          (node_a, node_join),
          (downstream_b, node_join),
          (node_join, downstream_join),
      ],
  )
  runner = testing_utils.InMemoryRunner(
      app=App(
          name="test_replay_cancel_wf",
          root_agent=wf,
          resumability_config=(
              ResumabilityConfig(is_resumable=True) if resumable else None
          ),
      )
  )

  # Turn 1: Both node_a and node_b pause for input/confirmation.
  events1 = await runner.run_async("start")
  inv_id = events1[0].invocation_id
  all_interrupts: list[str] = []
  for ev in events1:
    all_interrupts.extend(extract_event_interrupt_ids(ev))
  confirm_id = confirmation_interrupt_id("replay_cancel_wf@1/node_b@1")
  assert "ask_a" in all_interrupts
  assert confirm_id in all_interrupts

  # Turn 2: Reject node_b while node_a remains WAITING.
  await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(confirm_id, {"confirmed": False})
      ),
      invocation_id=inv_id,
  )
  assert not node_b_calls
  assert downstream_b_calls == [{"error": "This tool call is rejected."}]
  assert join_calls == []

  # Turn 3: Resume node_a -> node_b is replayed from history without re-running.
  await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response("ask_a", {"result": "a_done"})
      ),
      invocation_id=inv_id,
  )
  assert not node_b_calls
  assert downstream_b_calls == [{"error": "This tool call is rejected."}]
  assert join_calls == [{
      "node_a": "a_done",
      "downstream_b": "after_b",
  }]


@pytest.mark.parametrize("confirmed", [True, False])
@pytest.mark.asyncio
async def test_node_require_confirmation_as_llm_agent_tool(confirmed: bool):
  """Tests @node(require_confirmation=True) in LlmAgent.tools pauses and resumes via RequestInput."""
  from google.adk.utils._function_call_names import REQUEST_INPUT_FUNCTION_CALL_NAME
  from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response

  from . import workflow_testing_utils

  calls: list[str] = []

  @node(require_confirmation=True)
  def remove_file(path: str) -> dict[str, str]:
    """Removes a file."""
    calls.append(path)
    return {"removed": path}

  agent = LlmAgent(
      name="file_agent",
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name="remove_file", args={"path": "/tmp/a.txt"}
              ),
              types.Part.from_text(text="Done."),
          ]
      ),
      tools=[remove_file],
  )
  runner = testing_utils.InMemoryRunner(
      app=App(name="test_node_tool_confirm", root_agent=agent)
  )

  events1 = await runner.run_async("delete /tmp/a.txt")
  req_event = workflow_testing_utils.find_function_call_event(
      events1, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  assert req_event is not None
  assert not calls

  [interrupt_id] = extract_event_interrupt_ids(req_event)
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id, {"confirmed": confirmed})
      ),
      invocation_id=req_event.invocation_id,
  )

  tool_responses = [
      p.function_response.response
      for e in events2
      for p in (e.content.parts if e.content and e.content.parts else [])
      if p.function_response and p.function_response.name == "remove_file"
  ]
  if confirmed:
    assert calls == ["/tmp/a.txt"]
    assert tool_responses == [{"removed": "/tmp/a.txt"}]
  else:
    assert not calls
    assert tool_responses == [{"error": "This tool call is rejected."}]


@pytest.mark.asyncio
async def test_node_require_confirmation_dumps_nested_models_in_payload():
  """Tests @node(require_confirmation=True) recursively dumps nested BaseModels in payload args."""
  from google.adk.agents.context import Context
  from google.adk.utils._function_call_names import REQUEST_INPUT_FUNCTION_CALL_NAME

  from . import workflow_testing_utils

  class Item(BaseModel):
    name: str

  def seed_items(ctx: Context):
    ctx.state["items"] = [{"name": "a"}, {"name": "b"}]

  @node(require_confirmation=True)
  def batch_delete(items: list[Item]) -> int:
    return len(items)

  wf = Workflow(
      name="nested_model_payload_wf",
      edges=[(START, seed_items), (seed_items, batch_delete)],
  )
  runner = testing_utils.InMemoryRunner(
      app=App(name="test_nested_model_payload_wf", root_agent=wf)
  )

  events = await runner.run_async("start")
  req_event = workflow_testing_utils.find_function_call_event(
      events, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  assert req_event is not None
  fc_args = req_event.content.parts[0].function_call.args
  assert fc_args["payload"] == {
      "tool_name": "batch_delete",
      "args": {"items": [{"name": "a"}, {"name": "b"}]},
  }


@pytest.mark.parametrize("confirmed", [True, False])
@pytest.mark.asyncio
async def test_node_require_confirmation_runs_before_auth_gate(confirmed: bool):
  """Tests @node(require_confirmation=True, auth_config=...) confirms before collecting credentials."""
  from fastapi.openapi.models import APIKey
  from fastapi.openapi.models import APIKeyIn
  from google.adk.agents.context import Context
  from google.adk.auth.auth_credential import AuthCredential
  from google.adk.auth.auth_credential import AuthCredentialTypes
  from google.adk.auth.auth_tool import AuthConfig
  from google.adk.utils._function_call_names import REQUEST_EUC_FUNCTION_CALL_NAME
  from google.adk.utils._function_call_names import REQUEST_INPUT_FUNCTION_CALL_NAME
  from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response

  from . import workflow_testing_utils

  auth_config = AuthConfig(
      auth_scheme=APIKey(**{"in": APIKeyIn.header, "name": "X-Api-Key"}),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.API_KEY,
          api_key="placeholder",
      ),
      credential_key="test_api_key",
  )

  calls: list[str] = []

  @node(require_confirmation=True, auth_config=auth_config)
  def authed_action(ctx: Context) -> str:
    cred = ctx.get_auth_response(auth_config)
    calls.append(cred.api_key)
    return "authed_done"

  wf = Workflow(name="confirm_then_auth_wf", edges=[(START, authed_action)])
  runner = testing_utils.InMemoryRunner(
      app=App(name="test_confirm_then_auth_wf", root_agent=wf)
  )

  # Turn 1: Must pause on confirmation first, NOT auth.
  events1 = await runner.run_async("start")
  assert not workflow_testing_utils.get_auth_request_events(events1)
  req_event = workflow_testing_utils.find_function_call_event(
      events1, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  assert req_event is not None
  [confirm_id] = extract_event_interrupt_ids(req_event)

  # Turn 2: Respond to confirmation.
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(confirm_id, {"confirmed": confirmed})
      ),
      invocation_id=req_event.invocation_id,
  )

  if not confirmed:
    assert not workflow_testing_utils.get_auth_request_events(events2)
    assert not calls
    assert any(
        e.output == {"error": "This tool call is rejected."} for e in events2
    )
    return

  # Approved: now the auth gate requests credentials.
  auth_events = workflow_testing_utils.get_auth_request_events(events2)
  assert len(auth_events) == 1
  assert not calls
  [auth_id] = extract_event_interrupt_ids(auth_events[0])

  auth_response = AuthConfig(
      auth_scheme=auth_config.auth_scheme,
      raw_auth_credential=auth_config.raw_auth_credential,
      exchanged_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.API_KEY,
          api_key="secret_key_123",
      ),
      credential_key="test_api_key",
  )
  events3 = await runner.run_async(
      new_message=testing_utils.UserContent(
          types.Part(
              function_response=types.FunctionResponse(
                  id=auth_id,
                  name=REQUEST_EUC_FUNCTION_CALL_NAME,
                  response=auth_response.model_dump(mode="json", by_alias=True),
              )
          )
      ),
      invocation_id=req_event.invocation_id,
  )
  assert calls == ["secret_key_123"]
  assert any(e.output == "authed_done" for e in events3)


def test_node_iter_static_children_parallel_worker():
  """Node._iter_static_children yields _inner_node when parallel_worker is True."""
  node_inst = _CustomNode(name="subclass", parallel_worker=True)
  children = list(node_inst._iter_static_children())
  assert children == [node_inst._inner_node]


def test_node_iter_static_children_plain_node():
  """Node._iter_static_children yields nothing when parallel_worker is False."""
  node_inst = _CustomNode(name="subclass", parallel_worker=False)
  assert list(node_inst._iter_static_children()) == []


@pytest.mark.asyncio
async def test_node_run_impl_early_close_closes_inner_generator():
  """Closing Node._run_impl early closes run_node_impl."""
  cleaned_up = False
  first_event = Event(author="cleanup_node")
  second_event = Event(author="cleanup_node")

  class _CleanupNode(Node):

    async def run_node_impl(
        self, *, ctx: context_lib.Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      nonlocal cleaned_up
      try:
        yield first_event
        yield second_event
      finally:
        await asyncio.sleep(0)
        cleaned_up = True

  node_inst = _CleanupNode(name="cleanup_node")
  session = Session(app_name="test", user_id="user", id="session")
  ic = InvocationContext(
      invocation_id="inv",
      session=session,
      session_service=InMemorySessionService(),
  )
  ctx = context_lib.Context(ic, node=node_inst)

  async with Aclosing(node_inst.run(ctx=ctx, node_input=None)) as agen:
    first = await anext(agen)
    assert first is first_event

  assert cleaned_up
