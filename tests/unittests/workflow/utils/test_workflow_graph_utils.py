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

from __future__ import annotations

from unittest.mock import Mock

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.events.event import Event
from google.adk.tools._node_tool import NodeTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._base_node import START
from google.adk.workflow._function_node import FunctionNode
from google.adk.workflow._tool_node import _ToolNode
from google.adk.workflow.utils._workflow_graph_utils import build_node
from google.adk.workflow.utils._workflow_graph_utils import is_node_like
from google.genai import types
import pytest

from ... import testing_utils


class TestIsNodeLike:

  def test_returns_true_for_base_node(self):
    """is_node_like returns True for BaseNode instances."""

    class DummyNode(BaseNode):

      async def _run_impl(self, *, ctx, node_input):
        yield node_input

    node = DummyNode(name="test")

    assert is_node_like(node) is True

  def test_returns_true_for_base_tool(self):
    """is_node_like returns True for BaseTool instances."""

    class DummyTool(BaseTool):

      def execute(self, **kwargs):
        return "done"

    tool = DummyTool(name="test", description="test")

    assert is_node_like(tool) is True

  def test_returns_true_for_callable(self):
    """is_node_like returns True for callables."""

    def my_func():
      pass

    assert is_node_like(my_func) is True

  def test_returns_true_for_start_string(self):
    """is_node_like returns True for 'START' string."""
    assert is_node_like("START") is True

  def test_returns_false_for_invalid_types(self):
    """is_node_like returns False for invalid types."""
    assert is_node_like(123) is False
    assert is_node_like("NOT_START") is False


class TestBuildNode:

  def test_returns_start_when_node_like_is_start(self):
    """build_node returns START sentinel when input is 'START'."""
    assert build_node("START") == START

  def test_returns_copy_of_base_node_with_overrides(self):
    """build_node returns a copy of BaseNode with provided overrides."""

    class DummyNode(BaseNode):

      async def _run_impl(self, *, ctx, node_input):
        yield node_input

    node = DummyNode(name="original")

    built = build_node(node, name="new_name")

    assert built != node
    assert built.name == "new_name"

  def test_returns_tool_node_for_base_tool(self):
    """build_node wraps BaseTool in a _ToolNode."""

    class DummyTool(BaseTool):

      def execute(self, **kwargs):
        return "done"

    tool = DummyTool(name="test", description="test")

    built = build_node(tool)

    assert isinstance(built, _ToolNode)

  def test_unwraps_node_tool_to_underlying_node(self):
    """build_node unwraps NodeTool and returns the underlying BaseNode."""

    class DummyNode(BaseNode):

      async def _run_impl(self, *, ctx, node_input):
        yield node_input

    inner_node = DummyNode(name="inner", input_schema=str)
    node_tool = NodeTool(node=inner_node)

    built = build_node(node_tool)

    assert built is inner_node
    assert not isinstance(built, _ToolNode)

  def test_unwraps_node_tool_with_overrides(self):
    """build_node unwraps NodeTool and applies property overrides."""

    class DummyNode(BaseNode):

      async def _run_impl(self, *, ctx, node_input):
        yield node_input

    inner_node = DummyNode(name="original_name", input_schema=str)
    node_tool = NodeTool(node=inner_node)

    built = build_node(node_tool, name="overridden_name", timeout=12.5)

    assert built.name == "overridden_name"
    assert built.timeout == 12.5
    assert not isinstance(built, _ToolNode)

  def test_unwraps_node_tool_preserves_tool_name(self):
    """build_node unwraps NodeTool and preserves custom tool name."""

    class DummyNode(BaseNode):

      async def _run_impl(self, *, ctx, node_input):
        yield node_input

    inner_node = DummyNode(name="inner", input_schema=str)
    node_tool = NodeTool(node=inner_node, name="custom_tool_name")

    built = build_node(node_tool)

    assert built.name == "custom_tool_name"
    assert not isinstance(built, _ToolNode)

  def test_wraps_function_tool_in_tool_node(self):
    """build_node wraps FunctionTool in _ToolNode."""

    def custom_func(x: int) -> int:
      return x * 2

    func_tool = FunctionTool(func=custom_func)

    built = build_node(func_tool)

    assert isinstance(built, _ToolNode)
    assert built.tool is func_tool

  def test_returns_function_node_for_callable(self):
    """build_node wraps callable in a FunctionNode."""

    def my_func(x):
      return x

    built = build_node(my_func)

    assert isinstance(built, FunctionNode)

  def test_raises_value_error_for_invalid_type(self):
    """build_node raises ValueError for invalid types."""
    with pytest.raises(ValueError, match="Invalid node type"):
      build_node(123)

  def test_llm_agent_mode_defaults(self):
    """build_node sets correct default mode for LlmAgent."""
    root_agent = LlmAgent(name="root", instruction="test")
    sub_agent = LlmAgent(name="sub", description="test")
    # Dynamic subagent attachment without model_post_init normalization
    sub_agent.parent_agent = root_agent

    # Subagent with parent_agent should default to chat mode
    built_sub = build_node(sub_agent)
    assert built_sub.mode == "chat"

    # Standalone agent without parent_agent should default to single_turn
    standalone = LlmAgent(name="standalone", instruction="test")
    built_standalone = build_node(standalone)
    assert built_standalone.mode == "single_turn"

    # Standalone agent with default_llm_mode='chat' should default to chat
    built_chat = build_node(standalone, default_llm_mode="chat")
    assert built_chat.mode == "chat"
    assert standalone.mode is None

    # Agent with sub_agents can have default_llm_mode='chat'
    coord_child = LlmAgent(name="coord_child", description="test")
    coordinator = LlmAgent(name="coord", sub_agents=[coord_child])
    built_coord = build_node(coordinator, default_llm_mode="chat")
    assert built_coord.mode == "chat"
    assert coordinator.mode is None

  def test_build_node_remote_a2a_agent_non_task(self):
    """build_node does not wrap RemoteA2aAgent in task wrapper if mode is not task."""

    class DummyRemoteAgent(RemoteA2aAgent):

      def __init__(self, mode=None):
        super().__init__(name="dummy", agent_card="dummy_card", mode=mode)
        self.parent_agent = None

      def clone(self, *args, **kwargs):
        raise AssertionError("clone should not be called")

    agent = DummyRemoteAgent(mode=None)
    built = build_node(agent)
    assert built == agent

  @pytest.mark.parametrize(
      "has_parent, kwargs, expected",
      [
          (True, {}, True),
          (False, {}, False),
          (True, {"rerun_on_resume": False}, False),
      ],
      ids=["transfer_target", "standalone", "explicit_opt_out"],
  )
  def test_build_node_remote_a2a_agent_rerun_on_resume(
      self, has_parent, kwargs, expected
  ):
    """Only a transfer target reruns by default to answer the remote agent."""
    agent = RemoteA2aAgent(name="dummy", agent_card="dummy_card", **kwargs)
    if has_parent:
      LlmAgent(name="coordinator", sub_agents=[agent])

    assert build_node(agent).rerun_on_resume is expected

  @pytest.mark.asyncio
  async def test_remote_a2a_transfer_target_receives_interrupt_response(self):
    """A transfer target runs again when the call it left open is answered."""
    runs = []

    class FakeRemoteAgent(RemoteA2aAgent):

      async def _run_async_impl(self, ctx):
        runs.append(ctx.session.events[-1].get_function_responses())
        if len(runs) == 1:
          call = types.Part.from_function_call(name="long_task", args={})
          call.function_call.id = "call_long"
          yield Event(
              invocation_id=ctx.invocation_id,
              author=self.name,
              branch=ctx.branch,
              content=types.ModelContent([call]),
              long_running_tool_ids={"call_long"},
          )
        else:
          yield Event(
              invocation_id=ctx.invocation_id,
              author=self.name,
              branch=ctx.branch,
              content=types.ModelContent("done"),
          )

    coordinator = LlmAgent(
        name="coordinator",
        model=testing_utils.MockModel.create(
            responses=[
                types.Part.from_function_call(
                    name="transfer_to_agent", args={"agent_name": "remote"}
                )
            ]
        ),
        sub_agents=[FakeRemoteAgent(name="remote", agent_card="dummy_card")],
    )
    runner = testing_utils.InMemoryRunner(coordinator)
    await runner.run_async("Hi")

    answer = types.Part.from_function_response(
        name="long_task", response={"result": "ok"}
    )
    answer.function_response.id = "call_long"
    events = await runner.run_async(types.UserContent([answer]))

    assert len(runs) == 2
    assert [response.id for response in runs[1]] == ["call_long"]
    assert [e.content.parts[0].text for e in events if e.content] == ["done"]

  def test_build_node_remote_a2a_agent_task(self):
    """build_node raises ValueError if RemoteA2aAgent mode is task."""

    class DummyRemoteAgent(RemoteA2aAgent):

      def __init__(self, mode="task"):
        super().__init__(name="dummy", agent_card="dummy_card", mode=mode)
        self.parent_agent = None

    agent = DummyRemoteAgent(mode="task")
    with pytest.raises(
        ValueError,
        match=(
            "RemoteA2aAgent in task mode is not supported as a standalone"
            " workflow node. It is only supported in tool-delegation mode."
        ),
    ):
      build_node(agent)

  def test_build_node_remote_a2a_agent_task_with_parent(self):
    """build_node allows task-mode RemoteA2aAgent if parent_agent is set."""

    class DummyRemoteAgent(RemoteA2aAgent):

      def __init__(self, mode="task"):
        super().__init__(name="dummy", agent_card="dummy_card", mode=mode)
        self.parent_agent = Mock()

    agent = DummyRemoteAgent(mode="task")
    built = build_node(agent)
    assert built is not agent
    assert built.mode == "task"
    assert built.wait_for_output is True
