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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from google.adk.models.llm_request import LlmRequest
# Use the actual ComputerEnvironment enum from the code
from google.adk.tools.computer_use.base_computer import BaseComputer
from google.adk.tools.computer_use.base_computer import ComputerEnvironment
from google.adk.tools.computer_use.base_computer import ComputerState
from google.adk.tools.computer_use.computer_use_tool import ComputerUseTool
from google.adk.tools.computer_use.computer_use_toolset import ComputerUseToolset
from google.genai import types
import pytest


class MockComputer(BaseComputer):
  """Mock Computer implementation for testing."""

  def __init__(self):
    self.initialize_called = False
    self.close_called = False
    self._screen_size = (1920, 1080)
    self._environment = ComputerEnvironment.ENVIRONMENT_BROWSER

  async def initialize(self):
    self.initialize_called = True

  async def close(self):
    self.close_called = True

  async def screen_size(self) -> tuple[int, int]:
    return self._screen_size

  async def environment(self) -> ComputerEnvironment:
    return self._environment

  # Implement all abstract methods to make this a concrete class
  async def open_web_browser(self) -> ComputerState:
    return ComputerState(screenshot=b"test", url="https://example.com")

  async def click_at(self, x: int, y: int) -> ComputerState:
    return ComputerState(screenshot=b"test", url="https://example.com")

  async def hover_at(self, x: int, y: int) -> ComputerState:
    return ComputerState(screenshot=b"test", url="https://example.com")

  async def type_text_at(
      self,
      x: int,
      y: int,
      text: str,
      press_enter: bool = True,
      clear_before_typing: bool = True,
  ) -> ComputerState:
    return ComputerState(screenshot=b"test", url="https://example.com")

  async def scroll_document(self, direction: str) -> ComputerState:
    return ComputerState(screenshot=b"test", url="https://example.com")

  async def scroll_at(
      self, x: int, y: int, direction: str, magnitude: int
  ) -> ComputerState:
    return ComputerState(screenshot=b"test", url="https://example.com")

  async def wait(self, seconds: int) -> ComputerState:
    return ComputerState(screenshot=b"test", url="https://example.com")

  async def go_back(self) -> ComputerState:
    return ComputerState(screenshot=b"test", url="https://example.com")

  async def go_forward(self) -> ComputerState:
    return ComputerState(screenshot=b"test", url="https://example.com")

  async def search(self) -> ComputerState:
    return ComputerState(screenshot=b"test", url="https://example.com")

  async def navigate(self, url: str) -> ComputerState:
    return ComputerState(screenshot=b"test", url=url)

  async def key_combination(self, keys: list[str]) -> ComputerState:
    return ComputerState(screenshot=b"test", url="https://example.com")

  async def drag_and_drop(
      self, x: int, y: int, destination_x: int, destination_y: int
  ) -> ComputerState:
    return ComputerState(screenshot=b"test", url="https://example.com")

  async def current_state(self) -> ComputerState:
    return ComputerState(screenshot=b"test", url="https://example.com")


class TestComputerUseToolset:
  """Test cases for ComputerUseToolset class."""

  @pytest.fixture
  def mock_computer(self):
    """Fixture providing a mock computer."""
    return MockComputer()

  @pytest.fixture
  def toolset(self, mock_computer):
    """Fixture providing a ComputerUseToolset instance."""
    return ComputerUseToolset(computer=mock_computer)

  def test_init(self, mock_computer):
    """Test ComputerUseToolset initialization."""
    toolset = ComputerUseToolset(computer=mock_computer)

    assert toolset._computer == mock_computer
    assert toolset._initialized is False

  @pytest.mark.asyncio
  async def test_ensure_initialized(self, toolset, mock_computer):
    """Test that _ensure_initialized calls computer.initialize()."""
    assert not mock_computer.initialize_called
    assert not toolset._initialized

    await toolset._ensure_initialized()

    assert mock_computer.initialize_called
    assert toolset._initialized

  @pytest.mark.asyncio
  async def test_ensure_initialized_only_once(self, toolset, mock_computer):
    """Test that _ensure_initialized only calls initialize once."""
    await toolset._ensure_initialized()

    # Reset the flag to test it's not called again
    mock_computer.initialize_called = False

    await toolset._ensure_initialized()

    # Should not be called again
    assert not mock_computer.initialize_called
    assert toolset._initialized

  @pytest.mark.asyncio
  async def test_get_tools(self, toolset, mock_computer):
    """Test that get_tools returns ComputerUseTool instances."""
    tools = await toolset.get_tools()

    # Should initialize the computer
    assert mock_computer.initialize_called

    # Should return a list of ComputerUseTool instances
    assert isinstance(tools, list)
    assert len(tools) > 0
    assert all(isinstance(tool, ComputerUseTool) for tool in tools)

    # Each tool should have the correct configuration
    for tool in tools:
      assert tool._screen_size == (1920, 1080)
      # Should use default virtual screen size
      assert tool._coordinate_space == (1000, 1000)

  @pytest.mark.asyncio
  async def test_get_tools_excludes_utility_methods(self, toolset):
    """Test that get_tools excludes utility methods like screen_size, environment, close."""
    tools = await toolset.get_tools()

    # Get tool function names
    tool_names = [tool.func.__name__ for tool in tools]

    # Should exclude utility methods
    excluded_methods = {"screen_size", "environment", "close"}
    for method in excluded_methods:
      assert method not in tool_names

    # initialize might be included since it's a concrete method, not just abstract
    # This is acceptable behavior

    # Should include action methods
    expected_methods = {
        "open_web_browser",
        "click_at",
        "hover_at",
        "type_text_at",
        "scroll_document",
        "scroll_at",
        "wait",
        "go_back",
        "go_forward",
        "search",
        "navigate",
        "key_combination",
        "drag_and_drop",
        "current_state",
    }
    for method in expected_methods:
      assert method in tool_names

  @pytest.mark.asyncio
  async def test_get_tools_with_readonly_context(self, toolset):
    """Test get_tools with readonly_context parameter."""
    from google.adk.agents.readonly_context import ReadonlyContext

    readonly_context = MagicMock(spec=ReadonlyContext)

    tools = await toolset.get_tools(readonly_context=readonly_context)

    # Should still return tools (readonly_context doesn't affect behavior currently)
    assert isinstance(tools, list)
    assert len(tools) > 0

  @pytest.mark.asyncio
  async def test_close(self, toolset, mock_computer):
    """Test that close calls computer.close()."""
    await toolset.close()

    assert mock_computer.close_called

  @pytest.mark.asyncio
  async def test_get_tools_creates_tools_with_correct_methods(
      self, toolset, mock_computer
  ):
    """Test that get_tools creates tools with the correct underlying methods."""
    tools = await toolset.get_tools()

    # Find the click_at tool
    click_tool = None
    for tool in tools:
      if tool.func.__name__ == "click_at":
        click_tool = tool
        break

    assert click_tool is not None

    # The tool's function should be bound to the mock computer instance
    assert click_tool.func.__self__ == mock_computer

  @pytest.mark.asyncio
  async def test_get_tools_handles_custom_screen_size(self, mock_computer):
    """Test get_tools with custom screen size."""
    mock_computer._screen_size = (2560, 1440)

    toolset = ComputerUseToolset(computer=mock_computer)
    tools = await toolset.get_tools()

    # All tools should have the custom screen size
    for tool in tools:
      assert tool._screen_size == (2560, 1440)

  @pytest.mark.asyncio
  async def test_get_tools_handles_custom_environment(self, mock_computer):
    """Test get_tools with custom environment."""
    mock_computer._environment = ComputerEnvironment.ENVIRONMENT_UNSPECIFIED

    toolset = ComputerUseToolset(computer=mock_computer)
    tools = await toolset.get_tools()

    # Should still return tools regardless of environment
    assert isinstance(tools, list)
    assert len(tools) > 0

  @pytest.mark.asyncio
  async def test_multiple_get_tools_calls_return_cached_instances(
      self, toolset
  ):
    """Test that multiple get_tools calls return the same cached instances."""
    tools1 = await toolset.get_tools()
    tools2 = await toolset.get_tools()

    # Should return the same list instance
    assert tools1 is tools2

  def test_inheritance(self, toolset):
    """Test that ComputerUseToolset inherits from BaseToolset."""
    from google.adk.tools.base_toolset import BaseToolset

    assert isinstance(toolset, BaseToolset)

  @pytest.mark.asyncio
  async def test_get_tools_method_filtering(self, toolset):
    """Test that get_tools properly filters methods from BaseComputer."""
    tools = await toolset.get_tools()

    # Get all method names from the tools
    tool_method_names = [tool.func.__name__ for tool in tools]

    # Should not include private methods (starting with _)
    for name in tool_method_names:
      assert not name.startswith("_")

    # Should not include excluded methods
    excluded_methods = {"screen_size", "environment", "close"}
    for excluded in excluded_methods:
      assert excluded not in tool_method_names

  @pytest.mark.asyncio
  async def test_computer_method_binding(self, toolset, mock_computer):
    """Test that tools are properly bound to the computer instance."""
    tools = await toolset.get_tools()

    # All tools should be bound to the mock computer
    for tool in tools:
      assert tool.func.__self__ == mock_computer

  @pytest.mark.asyncio
  async def test_toolset_handles_computer_initialization_failure(
      self, mock_computer
  ):
    """Test that toolset handles computer initialization failure gracefully."""

    # Make initialize raise an exception
    async def failing_initialize():
      raise Exception("Initialization failed")

    mock_computer.initialize = failing_initialize

    toolset = ComputerUseToolset(computer=mock_computer)

    # Should raise the exception when trying to get tools
    with pytest.raises(Exception, match="Initialization failed"):
      await toolset.get_tools()

  @pytest.mark.asyncio
  async def test_process_llm_request(self, toolset, mock_computer):
    """Test that process_llm_request adds tools and computer use configuration."""
    llm_request = LlmRequest(
        model="gemini-1.5-flash",
        config=types.GenerateContentConfig(),
    )

    await toolset.process_llm_request(
        tool_context=MagicMock(), llm_request=llm_request
    )

    # Should add tools to the request
    assert len(llm_request.tools_dict) > 0

    # Should add computer use configuration
    assert llm_request.config.tools is not None
    assert len(llm_request.config.tools) > 0

    # Should have computer use tool
    computer_use_tools = [
        tool
        for tool in llm_request.config.tools
        if hasattr(tool, "computer_use") and tool.computer_use
    ]
    assert len(computer_use_tools) == 1

    # Should have correct environment
    computer_use_tool = computer_use_tools[0]
    assert (
        computer_use_tool.computer_use.environment
        == types.Environment.ENVIRONMENT_BROWSER
    )

  @pytest.mark.asyncio
  async def test_process_llm_request_with_existing_computer_use(
      self, toolset, mock_computer
  ):
    """Test that process_llm_request doesn't add duplicate computer use configuration."""
    llm_request = LlmRequest(
        model="gemini-1.5-flash",
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    computer_use=types.ToolComputerUse(
                        environment=types.Environment.ENVIRONMENT_BROWSER
                    )
                )
            ]
        ),
    )

    original_tools_count = len(llm_request.config.tools)

    await toolset.process_llm_request(
        tool_context=MagicMock(), llm_request=llm_request
    )

    # Should not add duplicate computer use configuration
    assert len(llm_request.config.tools) == original_tools_count

    # Should still add the actual tools
    assert len(llm_request.tools_dict) > 0

  @pytest.mark.asyncio
  async def test_process_llm_request_error_handling(self, mock_computer):
    """Test that process_llm_request handles errors gracefully."""

    # Make environment raise an exception
    async def failing_environment():
      raise Exception("Environment failed")

    mock_computer.environment = failing_environment

    toolset = ComputerUseToolset(computer=mock_computer)

    llm_request = LlmRequest(
        model="gemini-1.5-flash",
        config=types.GenerateContentConfig(),
    )

    # Should raise the exception
    with pytest.raises(Exception, match="Environment failed"):
      await toolset.process_llm_request(
          tool_context=MagicMock(), llm_request=llm_request
      )

  @pytest.mark.asyncio
  async def test_adapt_computer_use_tool_sync_adapter(self):
    """Test adapt_computer_use_tool with sync adapter function."""
    # Create a mock tool
    mock_func = AsyncMock()
    original_tool = ComputerUseTool(
        func=mock_func,
        screen_size=(1920, 1080),
        virtual_screen_size=(1000, 1000),
    )

    llm_request = LlmRequest(
        model="gemini-1.5-flash",
        config=types.GenerateContentConfig(),
    )
    llm_request.tools_dict["wait"] = original_tool

    # Create a sync adapter function
    def sync_adapter(original_func):
      async def adapted_func():
        return await original_func(5)

      return adapted_func

    # Call the adaptation method
    await ComputerUseToolset.adapt_computer_use_tool(
        "wait", sync_adapter, llm_request
    )

    # Verify the original tool was replaced
    assert "wait" not in llm_request.tools_dict
    assert "adapted_func" in llm_request.tools_dict

    # Verify the new tool has correct properties
    adapted_tool = llm_request.tools_dict["adapted_func"]
    assert isinstance(adapted_tool, ComputerUseTool)
    assert adapted_tool._screen_size == (1920, 1080)
    assert adapted_tool._coordinate_space == (1000, 1000)

  @pytest.mark.asyncio
  async def test_adapt_computer_use_tool_async_adapter(self):
    """Test adapt_computer_use_tool with async adapter function."""
    # Create a mock tool
    mock_func = AsyncMock()
    original_tool = ComputerUseTool(
        func=mock_func,
        screen_size=(1920, 1080),
        virtual_screen_size=(1000, 1000),
    )

    llm_request = LlmRequest(
        model="gemini-1.5-flash",
        config=types.GenerateContentConfig(),
    )
    llm_request.tools_dict["wait"] = original_tool

    # Create an async adapter function
    async def async_adapter(original_func):
      async def adapted_func():
        return await original_func(5)

      return adapted_func

    # Call the adaptation method
    await ComputerUseToolset.adapt_computer_use_tool(
        "wait", async_adapter, llm_request
    )

    # Verify the original tool was replaced
    assert "wait" not in llm_request.tools_dict
    assert "adapted_func" in llm_request.tools_dict

    # Verify the new tool has correct properties
    adapted_tool = llm_request.tools_dict["adapted_func"]
    assert isinstance(adapted_tool, ComputerUseTool)
    assert adapted_tool._screen_size == (1920, 1080)
    assert adapted_tool._coordinate_space == (1000, 1000)

  @pytest.mark.asyncio
  async def test_adapt_computer_use_tool_invalid_method(self):
    """Test adapt_computer_use_tool with invalid method name."""
    llm_request = LlmRequest(
        model="gemini-1.5-flash",
        config=types.GenerateContentConfig(),
    )

    def adapter(original_func):
      async def adapted_func():
        return await original_func()

      return adapted_func

    # Should not raise an exception, just log a warning
    await ComputerUseToolset.adapt_computer_use_tool(
        "invalid_method", adapter, llm_request
    )

    # Should not add any tools
    assert len(llm_request.tools_dict) == 0

  @pytest.mark.asyncio
  async def test_adapt_computer_use_tool_excluded_method(self):
    """Test adapt_computer_use_tool with excluded method name."""
    llm_request = LlmRequest(
        model="gemini-1.5-flash",
        config=types.GenerateContentConfig(),
    )

    def adapter(original_func):
      async def adapted_func():
        return await original_func()

      return adapted_func

    # Should not raise an exception, just log a warning
    await ComputerUseToolset.adapt_computer_use_tool(
        "screen_size", adapter, llm_request
    )

    # Should not add any tools
    assert len(llm_request.tools_dict) == 0

  @pytest.mark.asyncio
  async def test_adapt_computer_use_tool_method_not_in_tools_dict(self):
    """Test adapt_computer_use_tool when method is not in tools_dict."""
    llm_request = LlmRequest(
        model="gemini-1.5-flash",
        config=types.GenerateContentConfig(),
    )

    def adapter(original_func):
      async def adapted_func():
        return await original_func()

      return adapted_func

    # Should not raise an exception, just log a warning
    await ComputerUseToolset.adapt_computer_use_tool(
        "wait", adapter, llm_request
    )

    # Should not add any tools
    assert len(llm_request.tools_dict) == 0
