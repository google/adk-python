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

"""Tests for CodingAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.coding_agent import CodingAgent
from google.adk.agents.coding_agent import CodingAgentState
from google.adk.agents.coding_agent_config import CodingAgentConfig
from google.adk.agents.coding_agent_config import DEFAULT_SAFE_IMPORTS
from google.adk.tools.base_tool import BaseTool
import pytest


class TestCodingAgentConfig:
  """Tests for CodingAgentConfig."""

  def test_default_values(self):
    """Test that default values are set correctly."""
    config = CodingAgentConfig(name="test_agent")

    assert config.name == "test_agent"
    assert config.agent_class == "CodingAgent"
    assert config.max_iterations == 10
    assert config.error_retry_attempts == 2
    assert config.stateful is False
    assert config.tool_server_port == 8765
    assert config.authorized_imports == DEFAULT_SAFE_IMPORTS

  def test_custom_values(self):
    """Test that custom values can be set."""
    custom_imports = frozenset({"json", "math"})
    config = CodingAgentConfig(
        name="custom_agent",
        model="gemini-2.0-flash",
        max_iterations=20,
        error_retry_attempts=5,
        stateful=True,
        tool_server_port=9999,
        authorized_imports=custom_imports,
    )

    assert config.name == "custom_agent"
    assert config.model == "gemini-2.0-flash"
    assert config.max_iterations == 20
    assert config.error_retry_attempts == 5
    assert config.stateful is True
    assert config.tool_server_port == 9999
    assert config.authorized_imports == custom_imports

  def test_max_iterations_bounds(self):
    """Test max_iterations validation."""
    # Valid bounds
    config = CodingAgentConfig(name="test", max_iterations=1)
    assert config.max_iterations == 1

    config = CodingAgentConfig(name="test", max_iterations=100)
    assert config.max_iterations == 100

    # Invalid bounds
    with pytest.raises(ValueError):
      CodingAgentConfig(name="test", max_iterations=0)

    with pytest.raises(ValueError):
      CodingAgentConfig(name="test", max_iterations=101)

  def test_port_bounds(self):
    """Test tool_server_port validation."""
    # Valid bounds
    config = CodingAgentConfig(name="test", tool_server_port=1024)
    assert config.tool_server_port == 1024

    config = CodingAgentConfig(name="test", tool_server_port=65535)
    assert config.tool_server_port == 65535

    # Invalid bounds
    with pytest.raises(ValueError):
      CodingAgentConfig(name="test", tool_server_port=1023)

    with pytest.raises(ValueError):
      CodingAgentConfig(name="test", tool_server_port=65536)


class TestCodingAgentState:
  """Tests for CodingAgentState."""

  def test_default_state(self):
    """Test default state values."""
    state = CodingAgentState()

    assert state.iteration_count == 0
    assert state.error_count == 0
    assert state.execution_history == []

  def test_state_with_history(self):
    """Test state with execution history."""
    history = [
        {"iteration": 1, "code": "print('hello')", "success": True},
        {"iteration": 2, "code": "print('world')", "success": True},
    ]
    state = CodingAgentState(
        iteration_count=2,
        error_count=0,
        execution_history=history,
    )

    assert state.iteration_count == 2
    assert len(state.execution_history) == 2

  def test_state_serialization(self):
    """Test state can be serialized and deserialized."""
    state = CodingAgentState(
        iteration_count=5,
        error_count=1,
        execution_history=[{"iteration": 1, "code": "x = 1"}],
    )

    dumped = state.model_dump()
    restored = CodingAgentState.model_validate(dumped)

    assert restored.iteration_count == 5
    assert restored.error_count == 1
    assert len(restored.execution_history) == 1


class TestCodingAgent:
  """Tests for CodingAgent."""

  def test_agent_creation(self):
    """Test basic agent creation."""
    agent = CodingAgent(
        name="test_coding_agent",
        description="A test coding agent",
    )

    assert agent.name == "test_coding_agent"
    assert agent.description == "A test coding agent"
    assert agent.max_iterations == 10
    assert agent.error_retry_attempts == 2

  def test_agent_with_custom_config(self):
    """Test agent with custom configuration."""
    agent = CodingAgent(
        name="custom_agent",
        model="gemini-2.0-flash",
        max_iterations=5,
        error_retry_attempts=3,
        stateful=True,
    )

    assert agent.name == "custom_agent"
    assert agent.model == "gemini-2.0-flash"
    assert agent.max_iterations == 5
    assert agent.error_retry_attempts == 3
    assert agent.stateful is True

  def test_extract_code_block_tool_code(self):
    """Test code extraction from tool_code blocks."""
    agent = CodingAgent(name="test")

    response = """Here's some code:
```tool_code
result = search(query="test")
print(result)
```
That should work."""

    code = agent._extract_code_block(response)
    assert code == 'result = search(query="test")\nprint(result)'

  def test_extract_code_block_python(self):
    """Test code extraction from python blocks."""
    agent = CodingAgent(name="test")

    response = """Here's some code:
```python
x = 1 + 2
print(x)
```
Done."""

    code = agent._extract_code_block(response)
    assert code == "x = 1 + 2\nprint(x)"

  def test_extract_code_block_prefers_tool_code(self):
    """Test that tool_code blocks are preferred over python blocks."""
    agent = CodingAgent(name="test")

    response = """Code:
```tool_code
tool_result = tool_call()
```
Also:
```python
python_code = True
```"""

    code = agent._extract_code_block(response)
    assert code == "tool_result = tool_call()"

  def test_extract_code_block_no_code(self):
    """Test code extraction when no code block present."""
    agent = CodingAgent(name="test")

    response = "This is just text without any code blocks."
    code = agent._extract_code_block(response)
    assert code is None

  def test_build_error_feedback(self):
    """Test error feedback formatting."""
    agent = CodingAgent(name="test")

    error = "NameError: name 'undefined_var' is not defined"
    code = "print(undefined_var)"

    feedback = agent._build_error_feedback(error, code)

    assert "NameError" in feedback
    assert "undefined_var" in feedback
    assert code in feedback
    assert "fix the error" in feedback.lower()

  def test_default_model(self):
    """Test that default model is used when not specified."""
    agent = CodingAgent(name="test")

    # canonical_model property should return a BaseLlm
    model = agent.canonical_model
    assert model is not None

  def test_cleanup(self):
    """Test that cleanup releases resources."""
    agent = CodingAgent(name="test")
    agent._resolved_tools = [MagicMock()]
    agent._coding_executor = MagicMock()

    agent.cleanup()

    assert agent._resolved_tools is None
    assert agent._coding_executor is None


class TestCodingAgentTools:
  """Tests for CodingAgent tool handling."""

  def test_agent_with_function_tools(self):
    """Test agent with function tools."""

    def my_tool(query: str) -> dict:
      """A test tool."""
      return {"result": query}

    agent = CodingAgent(
        name="test",
        tools=[my_tool],
    )

    assert len(agent.tools) == 1

  def test_agent_with_base_tool(self):
    """Test agent with BaseTool instances."""

    class MockTool(BaseTool):

      def __init__(self):
        super().__init__(name="mock_tool", description="A mock tool")

      async def run_async(self, *, args, tool_context):
        return {"result": "mock"}

    tool = MockTool()
    agent = CodingAgent(
        name="test",
        tools=[tool],
    )

    assert len(agent.tools) == 1

  @pytest.mark.asyncio
  async def test_resolve_tools(self):
    """Test tool resolution."""

    def test_func(x: int) -> int:
      """Test function."""
      return x * 2

    agent = CodingAgent(
        name="test",
        tools=[test_func],
    )

    tools = await agent._resolve_tools()
    assert len(tools) == 1
    assert tools[0].name == "test_func"
