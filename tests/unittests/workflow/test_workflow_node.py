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

from unittest import mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.adk.workflow import FunctionNode
from google.adk.workflow import START
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._node import node
from google.adk.workflow._node import Node
from google.adk.workflow._parallel_worker import _ParallelWorker as ParallelWorker
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._tool_node import _ToolNode as ToolNode
from google.adk.workflow._workflow import Workflow
from google.genai import types
import pytest

from .. import testing_utils

ANY = mock.ANY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_workflow(wf, message="start"):
  """Run a Workflow through Runner, return collected events."""
  ss = InMemorySessionService()
  runner = Runner(app_name="test", node=wf, session_service=ss)
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
