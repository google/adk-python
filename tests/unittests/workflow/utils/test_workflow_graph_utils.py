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

from google.adk.tools.base_tool import BaseTool
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._base_node import START
from google.adk.workflow._function_node import FunctionNode
from google.adk.workflow._tool_node import _ToolNode
from google.adk.workflow.utils._workflow_graph_utils import build_node
from google.adk.workflow.utils._workflow_graph_utils import is_node_like
import pytest


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
