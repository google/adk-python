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

import asyncio
import base64
from dataclasses import dataclass
from dataclasses import field
import inspect
import json
import pickle
import socket
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock

from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.models.llm_request import LlmRequest
from google.adk.tools import _url_validator
# Use the actual ComputerEnvironment enum from the code
from google.adk.tools.computer_use.base_computer import BaseComputer
from google.adk.tools.computer_use.base_computer import ComputerEnvironment
from google.adk.tools.computer_use.base_computer import ComputerState
from google.adk.tools.computer_use.computer_use_tool import ComputerUseTool
from google.adk.tools.computer_use.computer_use_toolset import _COMPUTER_EXECUTION_LOCKS
from google.adk.tools.computer_use.computer_use_toolset import _COMPUTER_EXECUTION_LOCKS_GUARD
from google.adk.tools.computer_use.computer_use_toolset import _get_computer_execution_lock
from google.adk.tools.computer_use.computer_use_toolset import ComputerUseToolset
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest


def _assert_no_secret_screenshot(response: object, secret: bytes) -> None:
  """Check the model-facing response for raw and encoded screenshot content."""
  assert isinstance(response, dict)
  assert "image" not in response
  assert "screenshot" not in response
  serialized = json.dumps(response)
  assert secret.decode() not in serialized
  assert base64.b64encode(secret).decode() not in serialized


@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
  """Keep computer state validation independent of real DNS/network access."""
  monkeypatch.setattr(
      socket,
      "getaddrinfo",
      Mock(
          return_value=[(
              socket.AF_INET,
              socket.SOCK_STREAM,
              socket.IPPROTO_TCP,
              "",
              ("93.184.216.34", 0),
          )]
      ),
  )


class MockComputer(BaseComputer):
  """Mock Computer implementation for testing."""

  def __init__(self):
    self.initialize_called = False
    self.close_called = False
    self.navigate_calls: list[str] = []
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
    self.navigate_calls.append(url)
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
    excluded_methods = {"screen_size", "environment", "close", "type_text"}
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
  async def test_get_tools_filters_excluded_functions(self, mock_computer):
    """Test that get_tools filters out excluded functions."""
    excluded_funcs = ["drag_and_drop", "key_combination"]
    toolset = ComputerUseToolset(
        computer=mock_computer,
        excluded_predefined_functions=excluded_funcs,
    )

    tools = await toolset.get_tools()
    tool_names = [tool.func.__name__ for tool in tools]

    for func in excluded_funcs:
      assert func not in tool_names

    assert "click_at" in tool_names

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

    # The tool's function should have the correct name (wrapped method)
    assert click_tool.func.__name__ == "click_at"

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

    # All tools should have wrapped functions with correct names
    for tool in tools:
      # Wrapped functions preserve the original method name via functools.wraps
      assert callable(tool.func)
      assert not tool.func.__name__.startswith("_")

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
        model="gemini-2.5-flash",
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
  async def test_process_llm_request_with_excluded_functions(
      self, mock_computer
  ):
    """Test that process_llm_request passes excluded_predefined_functions."""
    excluded_funcs = ["drag_and_drop", "key_combination"]
    toolset = ComputerUseToolset(
        computer=mock_computer,
        excluded_predefined_functions=excluded_funcs,
    )

    llm_request = LlmRequest(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(),
    )

    await toolset.process_llm_request(
        tool_context=MagicMock(), llm_request=llm_request
    )

    # Should have computer use tool
    computer_use_tools = [
        tool
        for tool in llm_request.config.tools
        if hasattr(tool, "computer_use") and tool.computer_use
    ]
    assert len(computer_use_tools) == 1

    # Should have correct excluded functions
    computer_use_tool = computer_use_tools[0]
    assert (
        computer_use_tool.computer_use.excluded_predefined_functions
        == excluded_funcs
    )

  @pytest.mark.asyncio
  async def test_process_llm_request_with_existing_computer_use(
      self, toolset, mock_computer
  ):
    """Test that process_llm_request doesn't add duplicate computer use configuration."""
    llm_request = LlmRequest(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    computer_use=types.ComputerUse(
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
        model="gemini-2.5-flash",
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
        model="gemini-2.5-flash",
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
        model="gemini-2.5-flash",
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
        model="gemini-2.5-flash",
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
        model="gemini-2.5-flash",
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
        model="gemini-2.5-flash",
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


class TestNavigateUrlSafety:
  """Test cases for the navigate() url safety guard."""

  @pytest.fixture
  def mock_computer(self):
    """Fixture providing a mock computer."""
    return MockComputer()

  @pytest.fixture
  def resolver(self, monkeypatch) -> Mock:
    """Records every DNS lookup, resolving whatever is asked for to a public ip.

    Tests assert on this to pin down whether a url was refused before or after
    its hostname was resolved.
    """
    resolver = Mock(
        return_value=[(
            socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            ("93.184.216.34", 0),
        )]
    )
    monkeypatch.setattr(_url_validator.socket, "getaddrinfo", resolver)
    return resolver

  @staticmethod
  async def _build_navigate_tool(
      computer: MockComputer, **toolset_kwargs
  ) -> ComputerUseTool:
    """Returns the navigate tool of a toolset built over `computer`.

    Toolset settings differ per test, so they are passed in rather than fixed
    by a fixture.
    """
    toolset = ComputerUseToolset(computer=computer, **toolset_kwargs)
    for tool in await toolset.get_tools():
      if tool.func.__name__ == "navigate":
        return tool
    raise AssertionError("No navigate tool in toolset")

  @pytest.mark.parametrize(
      "url",
      [
          (
              "http://169.254.169.254/computeMetadata/v1/instance/"
              "service-accounts/default/token"
          ),
          # Parser-divergence regression test, not a duplicate: `urlparse`
          # reads the host as 'example.com' while a browser reads
          # '169.254.169.254', so the url is refused outright.
          r"http://169.254.169.254\@example.com/",
          "file:///etc/passwd",
          "http://localhost:3000/",
      ],
      ids=["cloud_metadata", "backslash_authority", "file_scheme", "localhost"],
  )
  @pytest.mark.asyncio
  async def test_navigate_refuses_unsafe_url(
      self, url, mock_computer, resolver
  ):
    """Unsafe urls are refused before the driver or a resolver is touched."""
    navigate_tool = await self._build_navigate_tool(mock_computer)
    url_before = (await mock_computer.current_state()).url

    result = await navigate_tool.func(url=url)

    # The security-critical assertion: the driver was never asked to navigate.
    assert mock_computer.navigate_calls == []
    assert result["error"]
    # The computer-use model rejects a function response carrying no url, so a
    # refusal reports the page the browser is still on
    assert result["url"] == url_before
    # The unsafe input never reaches DNS; the existing page is checked before
    # reporting its URL so a previous asynchronous navigation cannot leak it.
    assert [call.args[0] for call in resolver.call_args_list] == ["example.com"]

  @pytest.mark.asyncio
  async def test_navigate_allows_public_url_unmodified(
      self, mock_computer, resolver
  ):
    """A public url reaches the computer exactly as the model provided it."""
    navigate_tool = await self._build_navigate_tool(mock_computer)

    result = await navigate_tool.func(url="https://example.com/search?q=adk")

    assert mock_computer.navigate_calls == ["https://example.com/search?q=adk"]
    assert result.url == "https://example.com/search?q=adk"
    # Validate both the requested URL and the browser's resulting state.
    assert [call.args[0] for call in resolver.call_args_list] == [
        "example.com",
        "example.com",
    ]

  @pytest.mark.asyncio
  async def test_navigate_allows_loopback_with_private_network_access(
      self, mock_computer, resolver
  ):
    """allow_private_network_access=True lets loopback urls through."""
    navigate_tool = await self._build_navigate_tool(
        mock_computer, allow_private_network_access=True
    )

    result = await navigate_tool.func(url="http://127.0.0.1:8000/")

    assert mock_computer.navigate_calls == ["http://127.0.0.1:8000/"]
    assert result.url == "http://127.0.0.1:8000/"
    # The flag short-circuits before resolution, so no lookup happens.
    resolver.assert_not_called()

  @pytest.mark.asyncio
  async def test_navigate_guard_preserves_tool_context(
      self, mock_computer, resolver
  ):
    """The url guard must not cut navigate off from tool_context."""
    mock_computer.prepare = AsyncMock()
    tool_context = MagicMock(tool_confirmation=None)
    navigate_tool = await self._build_navigate_tool(mock_computer)

    result = await navigate_tool.run_async(
        args={"url": "https://example.com"}, tool_context=tool_context
    )

    mock_computer.prepare.assert_awaited_once_with(tool_context)
    assert mock_computer.navigate_calls == ["https://example.com"]
    assert result["url"] == "https://example.com"
    # Validate both the requested URL and the browser's resulting state.
    assert [call.args[0] for call in resolver.call_args_list] == [
        "example.com",
        "example.com",
    ]

  @pytest.mark.parametrize(
      "navigation", ["http_redirect", "meta_refresh", "javascript"]
  )
  @pytest.mark.parametrize(
      "destination",
      [
          "http://127.0.0.1/secret",
          "http://10.0.0.1/secret",
          "http://169.254.1.1/secret",
          "http://169.254.169.254/latest/meta-data/",
          "http://[::1]/secret",
          "http://localhost/secret",
      ],
  )
  async def test_private_navigation_landing_is_refused(
      self,
      navigation: str,
      destination: str,
      mock_computer: MockComputer,
      resolver: Mock,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """Never expose a private landing page's screenshot after navigation."""
    # The backend abstracts HTTP redirects and document-driven navigation into
    # the same final state. No browser or network is involved in these tests.
    monkeypatch.setattr(
        mock_computer,
        "navigate",
        AsyncMock(
            return_value=ComputerState(
                url=destination, screenshot=b"private secret"
            )
        ),
    )
    navigate = await self._build_navigate_tool(mock_computer)

    result = await navigate.func(url=f"https://example.com/{navigation}")

    assert isinstance(result, dict)
    _assert_no_secret_screenshot(result, b"private secret")
    assert result["error"]
    assert "image" not in result
    assert "screenshot" not in result
    assert mock_computer.close_called

  @pytest.mark.parametrize("allow_private", [False, True])
  async def test_navigation_landing_allowed_by_policy(
      self,
      allow_private: bool,
      mock_computer: MockComputer,
      resolver: Mock,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """Public redirects and explicitly permitted private landings work."""
    destination = (
        "http://127.0.0.1/" if allow_private else "https://other.example/"
    )
    state = ComputerState(url=destination, screenshot=b"allowed")
    monkeypatch.setattr(
        mock_computer, "navigate", AsyncMock(return_value=state)
    )
    navigate = await self._build_navigate_tool(
        mock_computer, allow_private_network_access=allow_private
    )

    result = await navigate.func(url="https://example.com/redirect")

    assert result == state
    assert not mock_computer.close_called

  async def test_delayed_private_navigation_is_refused(
      self,
      mock_computer: MockComputer,
      resolver: Mock,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """A JS/meta navigation completed during a later action is also refused."""
    monkeypatch.setattr(
        mock_computer,
        "wait",
        AsyncMock(
            return_value=ComputerState(
                url="http://169.254.169.254/", screenshot=b"private secret"
            )
        ),
    )
    toolset = ComputerUseToolset(computer=mock_computer)
    tools = {tool.name: tool for tool in await toolset.get_tools()}
    await tools["navigate"].func(url="https://example.com/")

    result = await tools["wait"].func(seconds=1)

    assert isinstance(result, dict)
    _assert_no_secret_screenshot(result, b"private secret")
    assert result["error"]
    assert mock_computer.close_called

  async def test_refused_landing_disables_subsequent_actions(
      self,
      mock_computer: MockComputer,
      resolver: Mock,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """A backend with a no-op close cannot expose the private page later."""
    private = ComputerState(url="http://127.0.0.1/", screenshot=b"secret")
    monkeypatch.setattr(
        mock_computer, "navigate", AsyncMock(return_value=private)
    )
    current_state = AsyncMock(return_value=private)
    monkeypatch.setattr(mock_computer, "current_state", current_state)
    toolset = ComputerUseToolset(computer=mock_computer)
    tools = {tool.name: tool for tool in await toolset.get_tools()}
    initial_result = await tools["navigate"].func(url="https://example.com/")
    _assert_no_secret_screenshot(initial_result, b"secret")

    result = await tools["current_state"].func()

    assert isinstance(result, dict)
    _assert_no_secret_screenshot(result, b"secret")
    assert result["error"]
    assert not current_state.called

  async def test_private_dns_landing_is_refused(
      self,
      mock_computer: MockComputer,
      resolver: Mock,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """A public-looking redirect hostname resolving to private IPs is refused."""
    resolver.side_effect = [
        [(
            socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            ("93.184.216.34", 0),
        )],
        [(
            socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            ("10.0.0.1", 0),
        )],
    ]
    monkeypatch.setattr(
        mock_computer,
        "navigate",
        AsyncMock(
            return_value=ComputerState(
                url="https://attacker.example/", screenshot=b"private secret"
            )
        ),
    )
    navigate = await self._build_navigate_tool(mock_computer)

    result = await navigate.func(url="https://example.com/")

    _assert_no_secret_screenshot(result, b"private secret")
    assert result["error"]
    assert mock_computer.close_called

  async def test_unsafe_current_page_is_not_reported_on_input_refusal(
      self,
      mock_computer: MockComputer,
      resolver: Mock,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """Rejecting an input must also reject an already-private current page."""
    monkeypatch.setattr(
        mock_computer,
        "current_state",
        AsyncMock(
            return_value=ComputerState(
                url="http://127.0.0.1/secret", screenshot=b"secret"
            )
        ),
    )
    navigate = await self._build_navigate_tool(mock_computer)

    result = await navigate.func(url="file:///etc/passwd")

    _assert_no_secret_screenshot(result, b"secret")
    assert result["error"]
    assert result["url"] == "about:blank"
    assert mock_computer.close_called
    assert mock_computer.navigate_calls == []

  async def test_cleanup_failure_keeps_computer_disabled(
      self,
      mock_computer: MockComputer,
      resolver: Mock,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """Failing cleanup must not allow subsequent actions to read private data."""
    monkeypatch.setattr(
        mock_computer,
        "navigate",
        AsyncMock(
            return_value=ComputerState(
                url="http://127.0.0.1/secret", screenshot=b"secret"
            )
        ),
    )
    monkeypatch.setattr(
        mock_computer,
        "close",
        AsyncMock(side_effect=RuntimeError("cleanup failed")),
    )
    current_state = AsyncMock()
    monkeypatch.setattr(mock_computer, "current_state", current_state)
    toolset = ComputerUseToolset(computer=mock_computer)
    tools = {tool.name: tool for tool in await toolset.get_tools()}

    with pytest.raises(RuntimeError, match="cleanup failed"):
      await tools["navigate"].func(url="https://example.com/")
    result = await tools["current_state"].func()

    _assert_no_secret_screenshot(result, b"secret")
    assert result["error"]
    assert not current_state.called

  async def test_private_landing_is_not_serialized_as_an_image(
      self,
      mock_computer: MockComputer,
      resolver: Mock,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """The public tool runner never encodes a refused screenshot for the LLM."""

    async def navigate_to_private(url: str) -> ComputerState:
      return ComputerState(url="http://169.254.169.254/", screenshot=b"secret")

    monkeypatch.setattr(mock_computer, "navigate", navigate_to_private)
    navigate = await self._build_navigate_tool(mock_computer)

    result = await navigate.run_async(
        args={"url": "https://example.com/"},
        tool_context=MagicMock(tool_confirmation=None),
    )

    _assert_no_secret_screenshot(result, b"secret")
    assert result["error"]
    assert "image" not in result
    assert result["url"] == "about:blank"

  @pytest.mark.parametrize("url", [None, "about:blank"])
  async def test_non_web_computer_state_remains_available(
      self,
      url: str | None,
      mock_computer: MockComputer,
      resolver: Mock,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """Desktop and blank browser states remain usable without a web URL."""
    state = ComputerState(url=url, screenshot=b"desktop")
    monkeypatch.setattr(
        mock_computer, "current_state", AsyncMock(return_value=state)
    )
    toolset = ComputerUseToolset(computer=mock_computer)
    tools = {tool.name: tool for tool in await toolset.get_tools()}

    result = await tools["current_state"].func()

    assert result == state
    assert not mock_computer.close_called
    resolver.assert_not_called()

  @pytest.mark.parametrize(
      "url",
      [
          None,
          "",
          "about:blank",
          "about:blank#anchor",
          "about:srcdoc",
          "chrome-error://chromewebdata/",
          "data:text/html,hello",
      ],
  )
  async def test_benign_landing_remains_usable(
      self,
      url: str | None,
      mock_computer: MockComputer,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """Benign browser states expose their image and allow later actions."""
    state = ComputerState(url=url, screenshot=b"benign pixels")

    async def current_state() -> ComputerState:
      return state

    monkeypatch.setattr(mock_computer, "current_state", current_state)
    toolset = ComputerUseToolset(computer=mock_computer)
    tools = {tool.name: tool for tool in await toolset.get_tools()}

    result = await tools["current_state"].run_async(
        args={}, tool_context=MagicMock(tool_confirmation=None)
    )
    later = await tools["click_at"].func(x=1, y=1)

    assert (
        result["image"]["data"] == base64.b64encode(b"benign pixels").decode()
    )
    assert isinstance(later, ComputerState)
    assert not mock_computer.close_called

  @pytest.mark.parametrize(
      "url",
      [
          "http://localhost/",
          "https://10.0.0.1/",
          "ws://127.0.0.1/",
          "wss://169.254.169.254/",
          "ftp://[::1]/",
          "gopher://metadata.goog/",
          "smb://192.168.1.1/",
          "file:///etc/passwd",
          "http://0177.0.0.1/",
          "http://0x7f.1/",
          "http://2130706433/",
      ],
  )
  async def test_confirmed_private_or_file_landing_latches(
      self,
      url: str,
      mock_computer: MockComputer,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """Private network and file pixels never reach the model or later calls."""

    async def current_state() -> ComputerState:
      return ComputerState(url=url, screenshot=b"secret landing pixels")

    monkeypatch.setattr(mock_computer, "current_state", current_state)
    toolset = ComputerUseToolset(computer=mock_computer)
    tools = {tool.name: tool for tool in await toolset.get_tools()}

    result = await tools["current_state"].run_async(
        args={}, tool_context=MagicMock(tool_confirmation=None)
    )
    later = await tools["click_at"].func(x=1, y=1)

    _assert_no_secret_screenshot(result, b"secret landing pixels")
    _assert_no_secret_screenshot(later, b"secret landing pixels")
    assert result["error"]
    assert later["error"]
    assert mock_computer.close_called

  @pytest.mark.parametrize(
      "url",
      [
          "chrome://settings/",
          "chrome-untrusted://new-tab-page/",
          "unknown://opaque-page/",
          "devtools://devtools/bundled/inspector.html",
          "javascript:void(0)",
          "custom:opaque-page",
          "about:config",
      ],
  )
  async def test_unknown_non_network_landing_is_withheld_without_latching(
      self,
      url: str,
      mock_computer: MockComputer,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """Unsupported internal pages are withheld without disabling later tools."""

    async def current_state() -> ComputerState:
      return ComputerState(url=url, screenshot=b"withheld pixels")

    monkeypatch.setattr(mock_computer, "current_state", current_state)
    toolset = ComputerUseToolset(computer=mock_computer)
    tools = {tool.name: tool for tool in await toolset.get_tools()}

    result = await tools["current_state"].run_async(
        args={}, tool_context=MagicMock(tool_confirmation=None)
    )
    later = await tools["click_at"].func(x=1, y=1)

    _assert_no_secret_screenshot(result, b"withheld pixels")
    assert result["error"]
    assert isinstance(later, ComputerState)
    assert not mock_computer.close_called

  @pytest.mark.parametrize(
      "error",
      [socket.gaierror("temporary DNS failure"), TimeoutError("DNS timeout")],
  )
  async def test_landing_dns_failure_is_withheld_without_latching(
      self,
      error: OSError,
      mock_computer: MockComputer,
      resolver: Mock,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """A transient resolver failure suppresses pixels but permits recovery."""

    async def current_state() -> ComputerState:
      return ComputerState(
          url="https://unavailable.example/", screenshot=b"secret DNS pixels"
      )

    monkeypatch.setattr(mock_computer, "current_state", current_state)
    toolset = ComputerUseToolset(computer=mock_computer)
    tools = {tool.name: tool for tool in await toolset.get_tools()}
    resolver.side_effect = error

    result = await tools["current_state"].run_async(
        args={}, tool_context=MagicMock(tool_confirmation=None)
    )
    resolver.side_effect = None
    later = await tools["click_at"].func(x=1, y=1)

    _assert_no_secret_screenshot(result, b"secret DNS pixels")
    assert result["error"]
    assert isinstance(later, ComputerState)
    assert not mock_computer.close_called

  async def test_private_dns_landing_latches_without_exposing_pixels(
      self,
      mock_computer: MockComputer,
      resolver: Mock,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """Positive private DNS resolution disables tools as well as hiding pixels."""

    async def current_state() -> ComputerState:
      return ComputerState(
          url="https://private.example/", screenshot=b"secret DNS pixels"
      )

    monkeypatch.setattr(mock_computer, "current_state", current_state)
    toolset = ComputerUseToolset(computer=mock_computer)
    tools = {tool.name: tool for tool in await toolset.get_tools()}
    resolver.return_value = [(
        socket.AF_INET,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
        "",
        ("169.254.169.254", 0),
    )]

    result = await tools["current_state"].run_async(
        args={}, tool_context=MagicMock(tool_confirmation=None)
    )
    later = await tools["click_at"].func(x=1, y=1)

    _assert_no_secret_screenshot(result, b"secret DNS pixels")
    _assert_no_secret_screenshot(later, b"secret DNS pixels")
    assert later["error"]
    assert mock_computer.close_called

  async def test_private_landing_cleanup_runs_only_once(
      self,
      mock_computer: MockComputer,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """Normal toolset cleanup does not close an already closed backend again."""

    async def current_state() -> ComputerState:
      return ComputerState(url="http://127.0.0.1/", screenshot=b"secret pixels")

    close = AsyncMock()
    monkeypatch.setattr(mock_computer, "current_state", current_state)
    monkeypatch.setattr(mock_computer, "close", close)
    toolset = ComputerUseToolset(computer=mock_computer)
    tools = {tool.name: tool for tool in await toolset.get_tools()}

    result = await tools["current_state"].run_async(
        args={}, tool_context=MagicMock(tool_confirmation=None)
    )
    await toolset.close()
    await toolset.close()

    _assert_no_secret_screenshot(result, b"secret pixels")
    close.assert_awaited_once()

  @pytest.mark.parametrize("host", ["0177.0.0.1", "0x7f.1", "2130706433"])
  async def test_numeric_private_navigation_input_is_refused(
      self,
      host: str,
      mock_computer: MockComputer,
  ) -> None:
    """Legacy numeric loopback forms cannot reach the navigation backend."""
    navigate = await self._build_navigate_tool(mock_computer)

    result = await navigate.run_async(
        args={"url": f"http://{host}/"},
        tool_context=MagicMock(tool_confirmation=None),
    )

    assert result["error"]
    assert mock_computer.navigate_calls == []

  @pytest.mark.parametrize("scheme", ["ws", "wss", "ftp"])
  async def test_public_network_landing_is_allowed(
      self,
      scheme: str,
      mock_computer: MockComputer,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """Other network schemes follow the same public-address policy."""
    state = ComputerState(
        url=f"{scheme}://8.8.8.8/", screenshot=b"public pixels"
    )

    async def current_state() -> ComputerState:
      return state

    monkeypatch.setattr(mock_computer, "current_state", current_state)
    toolset = ComputerUseToolset(computer=mock_computer)
    tools = {tool.name: tool for tool in await toolset.get_tools()}

    result = await tools["current_state"].run_async(
        args={}, tool_context=MagicMock(tool_confirmation=None)
    )

    assert (
        result["image"]["data"] == base64.b64encode(b"public pixels").decode()
    )
    assert not mock_computer.close_called


class _AsyncCountingComputer(MockComputer):
  """Computer that yields to the event loop during initialize/screen_size."""

  def __init__(self) -> None:
    super().__init__()
    self.init_count = 0
    self.screen_size_count = 0
    self.close_count = 0

  async def initialize(self) -> None:
    await asyncio.sleep(0)
    self.init_count += 1
    await super().initialize()

  async def close(self) -> None:
    await asyncio.sleep(0)
    self.close_count += 1
    await super().close()

  async def screen_size(self) -> tuple[int, int]:
    await asyncio.sleep(0)
    self.screen_size_count += 1
    return await super().screen_size()


@pytest.mark.asyncio
async def test_computer_use_toolset_concurrent_initialization():
  """Concurrent initialization and get_tools initialize computer once."""
  computer = _AsyncCountingComputer()
  toolset = ComputerUseToolset(computer=computer)

  await asyncio.gather(*(toolset._ensure_initialized() for _ in range(16)))
  assert computer.init_count == 1

  results = await asyncio.gather(*(toolset.get_tools() for _ in range(16)))
  assert computer.init_count == 1
  assert computer.screen_size_count == 1
  for tools in results:
    assert tools is results[0]


@pytest.mark.asyncio
async def test_computer_use_toolset_pickle_roundtrip():
  """Pickled toolset unpickles cleanly and re-initializes via public calls."""
  computer = _AsyncCountingComputer()
  toolset = ComputerUseToolset(computer=computer)
  tools = await toolset.get_tools_with_prefix()
  assert computer.init_count == 1

  restored = pickle.loads(pickle.dumps(toolset))
  restored_tools = await restored.get_tools_with_prefix()
  assert len(restored_tools) == len(tools)
  assert restored._computer.init_count == 2


@pytest.mark.asyncio
async def test_computer_use_toolset_close_resets_and_allows_reinitialization():
  """close() cleans up tools and resets initialization for subsequent calls."""
  computer = _AsyncCountingComputer()
  toolset = ComputerUseToolset(computer=computer)
  tools = await toolset.get_tools_with_prefix()
  assert computer.init_count == 1

  await asyncio.gather(*(toolset.close() for _ in range(8)))
  assert computer.close_count == 1

  new_tools = await toolset.get_tools_with_prefix()
  assert len(new_tools) == len(tools)
  assert computer.init_count == 2


class PointerRecordingComputer(MockComputer):
  """MockComputer that records pointer calls, with real method signatures.

  A bare AsyncMock cannot stand in for a BaseComputer method that is registered
  as a tool: get_tools() introspects each method with inspect.signature(),
  which raises TypeError on a Mock under Python 3.10.
  """

  def __init__(self, url: str = "https://example.com"):
    super().__init__()
    self._url = url
    self.click_at_calls: list[tuple[int, int]] = []
    self.hover_at_calls: list[tuple[int, int]] = []
    self.scroll_at_calls: list[tuple[int, int, str, int]] = []
    self.key_combination_calls: list[list[str]] = []
    self.type_text_at_calls: list[dict[str, object]] = []
    self.wait_calls: list[int] = []

  async def wait(self, seconds: int) -> ComputerState:
    self.wait_calls.append(seconds)
    return ComputerState(screenshot=b"test", url=self._url)

  async def click_at(self, x: int, y: int) -> ComputerState:
    self.click_at_calls.append((x, y))
    return ComputerState(screenshot=b"test", url=self._url)

  async def hover_at(self, x: int, y: int) -> ComputerState:
    self.hover_at_calls.append((x, y))
    return ComputerState(screenshot=b"test", url=self._url)

  async def scroll_at(
      self, x: int, y: int, direction: str, magnitude: int
  ) -> ComputerState:
    self.scroll_at_calls.append((x, y, direction, magnitude))
    return ComputerState(screenshot=b"test", url=self._url)

  async def key_combination(self, keys: list[str]) -> ComputerState:
    self.key_combination_calls.append(keys)
    return ComputerState(screenshot=b"test", url=self._url)

  async def type_text_at(
      self,
      x: int,
      y: int,
      text: str,
      press_enter: bool = True,
      clear_before_typing: bool = True,
  ) -> ComputerState:
    self.type_text_at_calls.append({
        "x": x,
        "y": y,
        "text": text,
        "press_enter": press_enter,
        "clear_before_typing": clear_before_typing,
    })
    return ComputerState(screenshot=b"test", url=self._url)


class TestGemini3ModernComputerUseTranslation:
  """Tests for Gemini 3.x Computer Use modern browser action translation."""

  @staticmethod
  def _make_tool_context(
      state: dict | None = None,
      function_call_id: str = "fc_id",
      tool_confirmation: ToolConfirmation | None = None,
  ) -> MagicMock:
    ctx = MagicMock(spec=ToolContext)
    ctx.state = state if state is not None else {}
    ctx.function_call_id = function_call_id
    ctx.tool_confirmation = tool_confirmation
    ctx.actions = MagicMock()
    ctx.actions.requested_tool_confirmations = {}

    def _request_confirmation(hint: str = ""):
      ctx.actions.requested_tool_confirmations[ctx.function_call_id] = (
          MagicMock(hint=hint)
      )

    ctx.request_confirmation.side_effect = _request_confirmation
    return ctx

  @pytest.mark.asyncio
  async def test_callable_spec_signatures_and_float_wait_validation(self):
    """Verifies shims expose modern signatures and forward arguments."""
    computer = PointerRecordingComputer()
    toolset = ComputerUseToolset(computer=computer)
    tools = await toolset.get_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    # Verify all registered tools are ComputerUseTool instances
    for tool in tools:
      assert isinstance(tool, ComputerUseTool)

    # Check modern parameter names are preserved in inspect.signature(tool.func)
    scroll_params = inspect.signature(tools_by_name["scroll"].func).parameters
    assert "magnitude_in_pixels" in scroll_params
    assert "direction" in scroll_params

    press_key_params = inspect.signature(
        tools_by_name["press_key"].func
    ).parameters
    assert "key" in press_key_params

    wait_params = inspect.signature(tools_by_name["wait"].func).parameters
    assert "seconds" in wait_params
    assert wait_params["seconds"].annotation in (float, "float")
    assert wait_params["seconds"].default == 1

    # Verify move, scroll, press_key, hotkey, and wait(seconds=0.4) pass
    # FUNCTION_TOOL_ARG_VALIDATION and forward translated args to BaseComputer.
    ctx = self._make_tool_context()
    with temporary_feature_override(
        FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True
    ):
      await tools_by_name["move"].run_async(
          args={"x": 500, "y": 500, "intent": "hover over target"},
          tool_context=ctx,
      )
      await tools_by_name["scroll"].run_async(
          args={
              "x": 500,
              "y": 500,
              "direction": "down",
              "magnitude_in_pixels": 450,
              "intent": "scroll down",
          },
          tool_context=ctx,
      )
      await tools_by_name["press_key"].run_async(
          args={"key": "Enter", "intent": "submit"},
          tool_context=ctx,
      )
      await tools_by_name["hotkey"].run_async(
          args={"keys": ["Control", "c"], "intent": "copy"},
          tool_context=ctx,
      )
      await tools_by_name["wait"].run_async(
          args={"seconds": 0.4, "intent": "wait for animation"},
          tool_context=ctx,
      )
    assert computer.hover_at_calls == [(960, 540)]
    assert computer.scroll_at_calls == [(960, 540, "down", 450)]
    assert computer.key_combination_calls == [["Enter"], ["Control", "c"]]
    assert computer.wait_calls == [1]

  @pytest.mark.asyncio
  async def test_type_dispatches_to_type_text_without_clicking_or_mutating_state(
      self,
  ):
    """Verifies `type` dispatches to `type_text` without clicking or state writes.

    Setup:
      Initialize a PointerRecordingComputer (using BaseComputer's default
      type_text) and a computer overriding type_text.

    Act & Assert:
      1. Default BaseComputer.type_text types via key_combination without
         calling click_at or type_text_at.
      2. Subclass overriding type_text receives (text, press_enter) directly.
      3. Neither click_at, click, nor type writes anything to
      tool_context.state.
    """
    computer = PointerRecordingComputer()
    toolset = ComputerUseToolset(computer=computer)
    tools_by_name = {tool.name: tool for tool in await toolset.get_tools()}

    state: dict[str, object] = {}
    ctx = self._make_tool_context(state=state)

    await tools_by_name["click"].run_async(
        args={"x": 200, "y": 300, "intent": "open search box"},
        tool_context=ctx,
    )
    await tools_by_name["type"].run_async(
        args={"text": "cats", "press_enter": True, "intent": "search cats"},
        tool_context=ctx,
    )

    assert computer.click_at_calls == [(384, 324)]
    assert computer.type_text_at_calls == []
    assert computer.key_combination_calls == [
        ["c"],
        ["a"],
        ["t"],
        ["s"],
        ["Enter"],
    ]
    assert state == {}

    class OverriddenTypeTextComputer(PointerRecordingComputer):

      def __init__(self):
        super().__init__()
        self.type_text_calls: list[tuple[str, bool]] = []

      async def type_text(
          self, text: str, press_enter: bool = False
      ) -> ComputerState:
        self.type_text_calls.append((text, press_enter))
        return ComputerState(screenshot=b"typed", url=self._url)

    custom_computer = OverriddenTypeTextComputer()
    custom_toolset = ComputerUseToolset(computer=custom_computer)
    custom_tools = {
        tool.name: tool for tool in await custom_toolset.get_tools()
    }

    await custom_tools["type"].run_async(
        args={"text": "hello world", "press_enter": False},
        tool_context=ctx,
    )
    assert custom_computer.type_text_calls == [("hello world", False)]
    assert custom_computer.click_at_calls == []
    assert custom_computer.type_text_at_calls == []
    assert state == {}

  @pytest.mark.asyncio
  async def test_unimplemented_modern_actions_excluded_in_computer_use(self):
    """Unimplemented 3.x actions are excluded in ComputerUse instead of shimmed."""
    computer = PointerRecordingComputer()
    toolset = ComputerUseToolset(computer=computer)
    tools_by_name = {tool.name: tool for tool in await toolset.get_tools()}

    unimplemented = (
        "double_click",
        "triple_click",
        "middle_click",
        "right_click",
        "mouse_down",
        "mouse_up",
        "key_down",
        "key_up",
    )
    for action_name in unimplemented:
      assert action_name not in tools_by_name

    llm_request = LlmRequest(
        model="gemini-3-pro-preview",
        config=types.GenerateContentConfig(),
    )
    await toolset.process_llm_request(
        tool_context=self._make_tool_context(), llm_request=llm_request
    )
    assert llm_request.config.tools[
        0
    ].computer_use.excluded_predefined_functions == list(unimplemented)

    class PartiallyImplementedComputer(PointerRecordingComputer):

      def __init__(self):
        super().__init__()
        self.double_clicks: list[tuple[int, int]] = []
        self.keys_down: list[str] = []

      async def double_click(self, x: int, y: int) -> ComputerState:
        self.double_clicks.append((x, y))
        return ComputerState(screenshot=b"test", url=self._url)

      async def key_down(self, key: str) -> ComputerState:
        self.keys_down.append(key)
        return ComputerState(screenshot=b"test", url=self._url)

    partial_computer = PartiallyImplementedComputer()
    partial_toolset = ComputerUseToolset(computer=partial_computer)
    partial_tools = {
        tool.name: tool for tool in await partial_toolset.get_tools()
    }
    assert "double_click" in partial_tools
    assert "key_down" in partial_tools
    assert "right_click" not in partial_tools

    ctx = self._make_tool_context()
    await partial_tools["double_click"].run_async(
        args={"x": 500, "y": 500}, tool_context=ctx
    )
    await partial_tools["key_down"].run_async(
        args={"key": "Shift"}, tool_context=ctx
    )
    assert partial_computer.double_clicks == [(960, 540)]
    assert partial_computer.keys_down == ["Shift"]

    partial_req = LlmRequest(
        model="gemini-3-pro-preview",
        config=types.GenerateContentConfig(),
    )
    await partial_toolset.process_llm_request(
        tool_context=ctx, llm_request=partial_req
    )
    assert partial_req.config.tools[
        0
    ].computer_use.excluded_predefined_functions == [
        "triple_click",
        "middle_click",
        "right_click",
        "mouse_down",
        "mouse_up",
        "key_up",
    ]

  @pytest.mark.asyncio
  async def test_safety_decision_and_resume_after_confirmation_on_alias(self):
    """Verifies safety_decision pauses alias execution and resumes cleanly.

    Setup:
      ToolContext with function_call_id and safety_decision on click action.

    Act & Assert:
      1. First call requests confirmation and returns an error without clicking.
      2. Second call (with confirmation) runs click_at and acknowledges safety.
    """
    computer = PointerRecordingComputer(url="https://example.com/pay")
    toolset = ComputerUseToolset(computer=computer)
    tools_by_name = {tool.name: tool for tool in await toolset.get_tools()}
    click_tool = tools_by_name["click"]

    state = {}
    ctx_first = self._make_tool_context(
        state=state, function_call_id="fc_safety_1"
    )
    call_args = {
        "x": 500,
        "y": 500,
        "intent": "submit payment",
        "safety_decision": {
            "decision": "require_confirmation",
            "explanation": "Confirm payment submission",
        },
    }

    # First call requests confirmation and does not execute click_at
    res_unconfirmed = await click_tool.run_async(
        args=dict(call_args), tool_context=ctx_first
    )
    assert "fc_safety_1" in ctx_first.actions.requested_tool_confirmations
    assert (
        ctx_first.actions.requested_tool_confirmations["fc_safety_1"].hint
        == "Confirm payment submission"
    )
    assert "error" in res_unconfirmed
    assert computer.click_at_calls == []

    # Resume with confirmation executes click_at and acknowledges safety
    ctx_resumed = self._make_tool_context(
        state=state,
        function_call_id="fc_safety_1",
        tool_confirmation=ToolConfirmation(confirmed=True),
    )
    res_confirmed = await click_tool.run_async(
        args=dict(call_args), tool_context=ctx_resumed
    )
    assert computer.click_at_calls == [(960, 540)]
    assert res_confirmed["safety_acknowledgement"] == "true"
    assert res_confirmed["url"] == "https://example.com/pay"

  @pytest.mark.asyncio
  async def test_class_level_override_probe_ignores_getattr_proxy(self):
    """Verifies class-level probe ignores __getattr__ proxies and helpers.

    Setup:
      Mock computer with dynamic __getattr__ and non-contract helpers.

    Act & Assert:
      Asserts unsupported actions are not exposed from __getattr__ proxies.
    """

    class ProxyComputer(MockComputer):

      def __init__(self):
        super().__init__()
        self.proxy_called = False
        self.click_at_calls = []

      async def click_at(self, x: int, y: int) -> ComputerState:
        self.click_at_calls.append((x, y))
        return ComputerState(screenshot=b"test", url="https://example.com")

      def __getattr__(self, name: str):
        async def _dynamic_rpc(*args, **kwargs):
          self.proxy_called = True
          return ComputerState(screenshot=b"proxy", url="https://proxy.example")

        return _dynamic_rpc

    proxy_comp = ProxyComputer()
    proxy_toolset = ComputerUseToolset(computer=proxy_comp)
    proxy_tools = {tool.name: tool for tool in await proxy_toolset.get_tools()}
    ctx = self._make_tool_context()

    await proxy_tools["click"].run_async(
        args={"x": 500, "y": 500}, tool_context=ctx
    )
    assert not proxy_comp.proxy_called
    assert proxy_comp.click_at_calls == [(960, 540)]
    assert "right_click" not in proxy_tools

    class NonContractHelperComputer(MockComputer):

      def __init__(self):
        super().__init__()
        self.helper_called = False

      def click(self, x: int, y: int) -> ComputerState:
        # Sync method must not be used as an async tool override
        self.helper_called = True
        return ComputerState(screenshot=b"sync", url="https://example.com")

      async def take_screenshot(
          self,
          bboxes: list[list[int]] | None = None,
          timeout: float | None = None,
      ) -> str | None:
        # Helper returning str | None (like BrowserServiceComputer)
        # must not be exposed as the Gemini 3.x take_screenshot tool.
        del bboxes, timeout
        self.helper_called = True
        return "/cns/path/screenshot.png"

    helper_comp = NonContractHelperComputer()
    helper_toolset = ComputerUseToolset(computer=helper_comp)
    helper_tools = {
        tool.name: tool for tool in await helper_toolset.get_tools()
    }
    click_res = await helper_tools["click"].run_async(
        args={"x": 500, "y": 500}, tool_context=ctx
    )
    assert not helper_comp.helper_called
    assert click_res["url"] == "https://example.com"
    assert "image" in click_res

    screenshot_res = await helper_tools["take_screenshot"].run_async(
        args={}, tool_context=ctx
    )
    assert not helper_comp.helper_called
    assert screenshot_res["url"] == "https://example.com"
    assert "image" in screenshot_res

    class NativeModernComputer(MockComputer):

      def __init__(self):
        super().__init__()
        self.native_clicks = []
        self.native_scrolls = []
        self.native_navigates = []

      async def click(
          self,
          x: int,
          y: int,
          button: str = "left",
          tool_context: ToolContext | None = None,
      ) -> ComputerState:
        self.native_clicks.append((x, y, button, tool_context))
        return ComputerState(screenshot=b"native", url="https://example.com")

      async def scroll(
          self,
          x: int,
          y: int,
          direction: str,
          magnitude_in_pixels: int = 300,
      ) -> ComputerState:
        self.native_scrolls.append((x, y, direction, magnitude_in_pixels))
        return ComputerState(screenshot=b"native", url="https://example.com")

      async def navigate(
          self,
          url: str,
          tool_context: ToolContext | None = None,
      ) -> ComputerState:
        self.native_navigates.append((url, tool_context))
        return ComputerState(screenshot=b"native", url=url)

    native_comp = NativeModernComputer()
    native_toolset = ComputerUseToolset(
        computer=native_comp, allow_private_network_access=True
    )
    native_tools = {
        tool.name: tool for tool in await native_toolset.get_tools()
    }
    await native_tools["click"].run_async(
        args={"x": 500, "y": 500, "button": "right"}, tool_context=ctx
    )
    assert native_comp.native_clicks == [(960, 540, "right", ctx)]

    await native_tools["scroll"].run_async(
        args={
            "x": 500,
            "y": 500,
            "direction": "down",
            "magnitude_in_pixels": 450,
        },
        tool_context=ctx,
    )
    assert native_comp.native_scrolls == [(960, 540, "down", 450)]

    await native_tools["navigate"].run_async(
        args={"url": "https://example.com/native"}, tool_context=ctx
    )
    assert native_comp.native_navigates == [("https://example.com/native", ctx)]

  @pytest.mark.asyncio
  async def test_concurrent_click_and_type_in_single_response(self):
    """Concurrent click and type in one response execute sequentially."""
    events: list[str] = []

    class OrderRecordingComputer(PointerRecordingComputer):

      async def click_at(self, x: int, y: int) -> ComputerState:
        events.append("click_start")
        await asyncio.sleep(0.01)
        events.append("click_end")
        return await super().click_at(x, y)

      async def type_text(
          self, text: str, press_enter: bool = False
      ) -> ComputerState:
        events.append(f"type:{text}:{press_enter}")
        return ComputerState(screenshot=b"test", url=self._url)

    computer = OrderRecordingComputer()
    toolset = ComputerUseToolset(computer=computer)
    tools_by_name = {tool.name: tool for tool in await toolset.get_tools()}
    ctx = self._make_tool_context(state={})

    click_res, type_res = await asyncio.gather(
        tools_by_name["click"].run_async(
            args={"x": 200, "y": 300}, tool_context=ctx
        ),
        tools_by_name["type"].run_async(
            args={"text": "first field", "press_enter": False},
            tool_context=ctx,
        ),
    )
    assert "error" not in click_res
    assert "error" not in type_res
    assert computer.click_at_calls == [(384, 324)]
    assert events == ["click_start", "click_end", "type:first field:False"]

  @pytest.mark.asyncio
  async def test_directional_and_conjunctive_exclusions_and_same_name_adapter(
      self,
  ):
    """Verifies directional, conjunctive, and model-aware exclusions.

    Setup:
      MockComputer configured with various excluded_predefined_functions lists.

    Act & Assert:
      1. Excluding hotkey leaves press_key and key_combination registered.
      2. Excluding both press_key and hotkey drops key_combination.
      3. Excluding scroll keeps scroll_document registered.
      4. Model-aware translation maps names for Gemini 3.x and Gemini 2.5.
      5. adapt_computer_use_tool retains the tool when the adapter returns a
         function with the same name.
    """
    computer = MockComputer()

    # 1. Excluding only hotkey leaves press_key and key_combination registered
    ts_hotkey_only = ComputerUseToolset(
        computer=computer, excluded_predefined_functions=["hotkey"]
    )
    names_hotkey_only = {t.name for t in await ts_hotkey_only.get_tools()}
    assert "hotkey" not in names_hotkey_only
    assert "press_key" in names_hotkey_only
    assert "key_combination" in names_hotkey_only

    # 2. Conjunctive: excluding both press_key and hotkey drops key_combination
    ts_both_keys = ComputerUseToolset(
        computer=computer,
        excluded_predefined_functions=["press_key", "hotkey"],
    )
    names_both_keys = {t.name for t in await ts_both_keys.get_tools()}
    assert "press_key" not in names_both_keys
    assert "hotkey" not in names_both_keys
    assert "key_combination" not in names_both_keys

    # 3. Directional: excluding scroll drops scroll_at, keeps scroll_document
    ts_scroll = ComputerUseToolset(
        computer=computer, excluded_predefined_functions=["scroll"]
    )
    names_scroll = {t.name for t in await ts_scroll.get_tools()}
    assert "scroll" not in names_scroll
    assert "scroll_at" not in names_scroll
    assert "scroll_document" in names_scroll

    # 4. Legacy exclusion of key_combination drops both press_key and hotkey,
    # and translates to ['press_key', 'hotkey'] (plus unimplemented 3.x actions)
    # on Gemini 3.x LLM requests.
    ts_legacy_key = ComputerUseToolset(
        computer=computer,
        excluded_predefined_functions=["click_at", "key_combination"],
    )
    names_legacy_key = {t.name for t in await ts_legacy_key.get_tools()}
    assert "click" not in names_legacy_key
    assert "click_at" not in names_legacy_key
    assert "key_combination" not in names_legacy_key
    assert "press_key" not in names_legacy_key
    assert "hotkey" not in names_legacy_key

    expected_modern_exclusions = [
        "click",
        "press_key",
        "hotkey",
        "double_click",
        "triple_click",
        "middle_click",
        "right_click",
        "mouse_down",
        "mouse_up",
        "key_down",
        "key_up",
    ]
    req_v2 = LlmRequest(
        model="gemini-3-pro-preview",
        config=types.GenerateContentConfig(),
    )
    await ts_legacy_key.process_llm_request(
        tool_context=self._make_tool_context(), llm_request=req_v2
    )
    assert (
        req_v2.config.tools[0].computer_use.excluded_predefined_functions
        == expected_modern_exclusions
    )

    # Unparseable and custom model names translate configured exclusions to 3.x
    # without auto-injecting unimplemented 3.x exclusions (which would fail on a
    # Gemini 2.5 model deployed behind a custom endpoint or alias).
    ts_no_exclusions = ComputerUseToolset(computer=computer)
    for unparseable_model in (
        "gemini-flash-latest",
        "projects/123/locations/us-central1/endpoints/456",
        None,
    ):
      req_unparseable = LlmRequest(
          model=unparseable_model,
          config=types.GenerateContentConfig(),
      )
      await ts_legacy_key.process_llm_request(
          tool_context=self._make_tool_context(), llm_request=req_unparseable
      )
      assert req_unparseable.config.tools[
          0
      ].computer_use.excluded_predefined_functions == [
          "click",
          "press_key",
          "hotkey",
      ]

      req_unparseable_default = LlmRequest(
          model=unparseable_model,
          config=types.GenerateContentConfig(),
      )
      await ts_no_exclusions.process_llm_request(
          tool_context=self._make_tool_context(),
          llm_request=req_unparseable_default,
      )
      assert (
          req_unparseable_default.config.tools[
              0
          ].computer_use.excluded_predefined_functions
          is None
      )

    req_v1 = LlmRequest(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(),
    )
    await ts_both_keys.process_llm_request(
        tool_context=self._make_tool_context(), llm_request=req_v1
    )
    assert req_v1.config.tools[
        0
    ].computer_use.excluded_predefined_functions == [
        "key_combination",
    ]

    # 5. Same-name adapter replaces 'wait' in place without deleting it
    ts_default = ComputerUseToolset(computer=computer)
    llm_request = LlmRequest(
        model="gemini-3-pro-preview",
        config=types.GenerateContentConfig(),
    )
    await ts_default.process_llm_request(
        tool_context=self._make_tool_context(), llm_request=llm_request
    )

    def same_name_wait_adapter(original_func):
      async def wait(seconds: int = 3):
        return await original_func(seconds)

      return wait

    await ComputerUseToolset.adapt_computer_use_tool(
        "wait", same_name_wait_adapter, llm_request
    )
    assert "wait" in llm_request.tools_dict

  @pytest.mark.asyncio
  async def test_unhashable_dataclass_computer_executes_without_type_error(
      self,
  ):
    """Unhashable @dataclass BaseComputer executes actions without TypeError."""

    @dataclass
    class DataclassComputer(PointerRecordingComputer):
      label: str = "dataclass-computer"
      extra_tags: list[str] = field(default_factory=list)

      def __post_init__(self):
        super().__init__()

    computer = DataclassComputer()
    with pytest.raises(TypeError):
      hash(computer)

    toolset = ComputerUseToolset(computer=computer)
    tools_by_name = {tool.name: tool for tool in await toolset.get_tools()}
    ctx = self._make_tool_context(state={})

    click_res, type_res = await asyncio.gather(
        tools_by_name["click"].run_async(
            args={"x": 200, "y": 300}, tool_context=ctx
        ),
        tools_by_name["type"].run_async(
            args={"text": "hi", "press_enter": False},
            tool_context=ctx,
        ),
    )
    assert "error" not in click_res
    assert "error" not in type_res
    assert computer.click_at_calls == [(384, 324)]
    assert computer.key_combination_calls == [["h"], ["i"]]

  @pytest.mark.asyncio
  async def test_gc_weakref_cleanup_while_guard_held_does_not_deadlock(self):
    """Weakref cleanup running while guard is held on same thread succeeds."""
    computer: MockComputer | None = MockComputer()
    computer_id = id(computer)
    _get_computer_execution_lock(computer)
    assert computer_id in _COMPUTER_EXECUTION_LOCKS

    with _COMPUTER_EXECUTION_LOCKS_GUARD:
      del computer

    assert computer_id not in _COMPUTER_EXECUTION_LOCKS
