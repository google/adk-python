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

import base64
import inspect

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.computer_use.base_computer import ComputerState
from google.adk.tools.computer_use.computer_use_tool import ComputerUseTool
from google.adk.tools.tool_context import ToolContext
import pytest


class TestComputerUseTool:
  """Test cases for ComputerUseTool class."""

  @pytest.fixture
  async def tool_context(self):
    """Fixture providing a tool context."""
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="test_app", user_id="test_user"
    )
    agent = SequentialAgent(name="test_agent")
    invocation_context = InvocationContext(
        invocation_id="invocation_id",
        agent=agent,
        session=session,
        session_service=session_service,
    )
    return ToolContext(invocation_context=invocation_context)

  @pytest.fixture
  def mock_computer_function(self):
    """Fixture providing a mock computer function."""
    # Create a real async function instead of AsyncMock for Python 3.9 compatibility
    calls = []

    async def mock_func(*args, **kwargs):
      calls.append((args, kwargs))
      # Return a default ComputerState - this will be overridden in individual tests
      return ComputerState(screenshot=b"default", url="https://default.com")

    # Add attributes that tests expect
    mock_func.__name__ = "test_function"
    mock_func.__doc__ = "Test function documentation"
    mock_func.calls = calls

    # Add assertion methods for compatibility with Mock
    def assert_called_once_with(*args, **kwargs):
      assert len(calls) == 1, f"Expected 1 call, got {len(calls)}"
      assert calls[0] == (
          args,
          kwargs,
      ), f"Expected {(args, kwargs)}, got {calls[0]}"

    def assert_called_once():
      assert len(calls) == 1, f"Expected 1 call, got {len(calls)}"

    mock_func.assert_called_once_with = assert_called_once_with
    mock_func.assert_called_once = assert_called_once

    return mock_func

  def test_init(self, mock_computer_function):
    """Test ComputerUseTool initialization."""
    screen_size = (1920, 1080)
    tool = ComputerUseTool(func=mock_computer_function, screen_size=screen_size)

    assert tool._screen_size == screen_size
    assert tool.func == mock_computer_function

  def test_init_with_invalid_screen_size(self, mock_computer_function):
    """Test ComputerUseTool initialization with invalid screen size."""
    with pytest.raises(ValueError, match="screen_size must be a tuple"):
      ComputerUseTool(func=mock_computer_function, screen_size=[1920, 1080])

    with pytest.raises(ValueError, match="screen_size must be a tuple"):
      ComputerUseTool(func=mock_computer_function, screen_size=(1920,))

    with pytest.raises(
        ValueError, match="screen_size dimensions must be positive"
    ):
      ComputerUseTool(func=mock_computer_function, screen_size=(0, 1080))

    with pytest.raises(
        ValueError, match="screen_size dimensions must be positive"
    ):
      ComputerUseTool(func=mock_computer_function, screen_size=(1920, -1))

  def test_init_with_invalid_virtual_screen_size(self, mock_computer_function):
    """Test ComputerUseTool initialization with invalid virtual_screen_size."""
    with pytest.raises(ValueError, match="virtual_screen_size must be a tuple"):
      ComputerUseTool(
          func=mock_computer_function,
          screen_size=(1920, 1080),
          virtual_screen_size=[1000, 1000],
      )

    with pytest.raises(ValueError, match="virtual_screen_size must be a tuple"):
      ComputerUseTool(
          func=mock_computer_function,
          screen_size=(1920, 1080),
          virtual_screen_size=(1000,),
      )

    with pytest.raises(
        ValueError, match="virtual_screen_size dimensions must be positive"
    ):
      ComputerUseTool(
          func=mock_computer_function,
          screen_size=(1920, 1080),
          virtual_screen_size=(0, 1000),
      )

    with pytest.raises(
        ValueError, match="virtual_screen_size dimensions must be positive"
    ):
      ComputerUseTool(
          func=mock_computer_function,
          screen_size=(1920, 1080),
          virtual_screen_size=(1000, -1),
      )

  def test_init_with_custom_virtual_screen_size(self, mock_computer_function):
    """Test ComputerUseTool initialization with custom virtual_screen_size."""
    screen_size = (1920, 1080)
    virtual_screen_size = (2000, 2000)
    tool = ComputerUseTool(
        func=mock_computer_function,
        screen_size=screen_size,
        virtual_screen_size=virtual_screen_size,
    )

    assert tool._screen_size == screen_size
    assert tool._coordinate_space == virtual_screen_size
    assert tool.func == mock_computer_function

  def test_normalize_x(self, mock_computer_function):
    """Test x coordinate normalization with default virtual screen size (1000x1000)."""
    tool = ComputerUseTool(
        func=mock_computer_function, screen_size=(1920, 1080)
    )

    # Test normal cases
    assert tool._normalize_x(0) == 0
    assert tool._normalize_x(500) == 960  # 500/1000 * 1920
    assert tool._normalize_x(1000) == 1919  # Clamped to screen bounds

    # Test edge cases
    assert tool._normalize_x(-100) == 0  # Clamped to 0
    assert tool._normalize_x(1500) == 1919  # Clamped to max

  def test_normalize_y(self, mock_computer_function):
    """Test y coordinate normalization with default virtual screen size (1000x1000)."""
    tool = ComputerUseTool(
        func=mock_computer_function, screen_size=(1920, 1080)
    )

    # Test normal cases
    assert tool._normalize_y(0) == 0
    assert tool._normalize_y(500) == 540  # 500/1000 * 1080
    assert tool._normalize_y(1000) == 1079  # Clamped to screen bounds

    # Test edge cases
    assert tool._normalize_y(-100) == 0  # Clamped to 0
    assert tool._normalize_y(1500) == 1079  # Clamped to max

  def test_normalize_with_custom_virtual_screen_size(
      self, mock_computer_function
  ):
    """Test coordinate normalization with custom virtual screen size."""
    tool = ComputerUseTool(
        func=mock_computer_function,
        screen_size=(1920, 1080),
        virtual_screen_size=(2000, 2000),
    )

    # Test x coordinate normalization with 2000x2000 virtual space
    assert tool._normalize_x(0) == 0
    assert tool._normalize_x(1000) == 960  # 1000/2000 * 1920
    assert tool._normalize_x(2000) == 1919  # Clamped to screen bounds

    # Test y coordinate normalization with 2000x2000 virtual space
    assert tool._normalize_y(0) == 0
    assert tool._normalize_y(1000) == 540  # 1000/2000 * 1080
    assert tool._normalize_y(2000) == 1079  # Clamped to screen bounds

    # Test edge cases
    assert tool._normalize_x(-100) == 0  # Clamped to 0
    assert tool._normalize_x(3000) == 1919  # Clamped to max
    assert tool._normalize_y(-100) == 0  # Clamped to 0
    assert tool._normalize_y(3000) == 1079  # Clamped to max

  def test_normalize_with_invalid_coordinates(self, mock_computer_function):
    """Test coordinate normalization with invalid inputs."""
    tool = ComputerUseTool(
        func=mock_computer_function, screen_size=(1920, 1080)
    )

    with pytest.raises(ValueError, match="x coordinate must be numeric"):
      tool._normalize_x("invalid")

    with pytest.raises(ValueError, match="y coordinate must be numeric"):
      tool._normalize_y("invalid")

  @pytest.mark.asyncio
  async def test_run_async_with_coordinates(
      self, mock_computer_function, tool_context
  ):
    """Test run_async with coordinate normalization."""

    # Set up a proper signature for the mock function
    def dummy_func(x: int, y: int):
      pass

    mock_computer_function.__name__ = "dummy_func"
    mock_computer_function.__signature__ = inspect.signature(dummy_func)

    # Create a specific mock function for this test that returns the expected state
    calls = []
    mock_state = ComputerState(
        screenshot=b"test_screenshot", url="https://example.com"
    )

    async def specific_mock_func(x: int, y: int):
      calls.append((x, y))
      return mock_state

    specific_mock_func.__name__ = "dummy_func"
    specific_mock_func.__signature__ = inspect.signature(dummy_func)
    specific_mock_func.calls = calls

    def assert_called_once_with(x, y):
      assert len(calls) == 1, f"Expected 1 call, got {len(calls)}"
      assert calls[0] == (x, y), f"Expected ({x}, {y}), got {calls[0]}"

    specific_mock_func.assert_called_once_with = assert_called_once_with

    tool = ComputerUseTool(func=specific_mock_func, screen_size=(1920, 1080))

    args = {"x": 500, "y": 300}
    result = await tool.run_async(args=args, tool_context=tool_context)

    # Check that coordinates were normalized
    specific_mock_func.assert_called_once_with(x=960, y=324)

    # Check return format for ComputerState
    expected_result = {
        "image": {
            "mimetype": "image/png",
            "data": base64.b64encode(b"test_screenshot").decode("utf-8"),
        },
        "url": "https://example.com",
    }
    assert result == expected_result

  @pytest.mark.asyncio
  async def test_run_async_with_drag_and_drop_coordinates(
      self, mock_computer_function, tool_context
  ):
    """Test run_async with drag and drop coordinate normalization."""

    # Set up a proper signature for the mock function
    def dummy_func(x: int, y: int, destination_x: int, destination_y: int):
      pass

    # Create a specific mock function for this test
    calls = []
    mock_state = ComputerState(
        screenshot=b"test_screenshot", url="https://example.com"
    )

    async def specific_mock_func(
        x: int, y: int, destination_x: int, destination_y: int
    ):
      calls.append((x, y, destination_x, destination_y))
      return mock_state

    specific_mock_func.__name__ = "dummy_func"
    specific_mock_func.__signature__ = inspect.signature(dummy_func)
    specific_mock_func.calls = calls

    def assert_called_once_with(x, y, destination_x, destination_y):
      assert len(calls) == 1, f"Expected 1 call, got {len(calls)}"
      assert calls[0] == (x, y, destination_x, destination_y), (
          f"Expected ({x}, {y}, {destination_x}, {destination_y}), got"
          f" {calls[0]}"
      )

    specific_mock_func.assert_called_once_with = assert_called_once_with

    tool = ComputerUseTool(func=specific_mock_func, screen_size=(1920, 1080))

    args = {"x": 100, "y": 200, "destination_x": 800, "destination_y": 600}
    result = await tool.run_async(args=args, tool_context=tool_context)

    # Check that all coordinates were normalized
    specific_mock_func.assert_called_once_with(
        x=192,  # 100/1000 * 1920
        y=216,  # 200/1000 * 1080
        destination_x=1536,  # 800/1000 * 1920
        destination_y=648,  # 600/1000 * 1080
    )

  @pytest.mark.asyncio
  async def test_run_async_with_non_computer_state_result(
      self, mock_computer_function, tool_context
  ):
    """Test run_async when function returns non-ComputerState result."""
    # Create a specific mock function that returns non-ComputerState
    calls = []

    async def specific_mock_func(*args, **kwargs):
      calls.append((args, kwargs))
      return {"status": "success"}

    specific_mock_func.__name__ = "test_function"
    specific_mock_func.calls = calls

    tool = ComputerUseTool(func=specific_mock_func, screen_size=(1920, 1080))

    args = {"text": "hello"}
    result = await tool.run_async(args=args, tool_context=tool_context)

    # Should return the result as-is
    assert result == {"status": "success"}

  @pytest.mark.asyncio
  async def test_run_async_without_coordinates(
      self, mock_computer_function, tool_context
  ):
    """Test run_async with no coordinate parameters."""

    # Set up a proper signature for the mock function
    def dummy_func(direction: str):
      pass

    # Create a specific mock function for this test
    calls = []
    mock_state = ComputerState(
        screenshot=b"test_screenshot", url="https://example.com"
    )

    async def specific_mock_func(direction: str):
      calls.append((direction,))
      return mock_state

    specific_mock_func.__name__ = "dummy_func"
    specific_mock_func.__signature__ = inspect.signature(dummy_func)
    specific_mock_func.calls = calls

    def assert_called_once_with(direction):
      assert len(calls) == 1, f"Expected 1 call, got {len(calls)}"
      assert calls[0] == (
          direction,
      ), f"Expected ({direction},), got {calls[0]}"

    specific_mock_func.assert_called_once_with = assert_called_once_with

    tool = ComputerUseTool(func=specific_mock_func, screen_size=(1920, 1080))

    args = {"direction": "down"}
    result = await tool.run_async(args=args, tool_context=tool_context)

    # Should call function with original args
    specific_mock_func.assert_called_once_with(direction="down")

  @pytest.mark.asyncio
  async def test_run_async_with_error(
      self, mock_computer_function, tool_context
  ):
    """Test run_async when underlying function raises an error."""
    # Create a specific mock function that raises an error
    calls = []

    async def specific_mock_func(*args, **kwargs):
      calls.append((args, kwargs))
      raise ValueError("Test error")

    specific_mock_func.__name__ = "test_function"
    specific_mock_func.calls = calls

    tool = ComputerUseTool(func=specific_mock_func, screen_size=(1920, 1080))

    args = {"x": 500, "y": 300}

    with pytest.raises(ValueError, match="Test error"):
      await tool.run_async(args=args, tool_context=tool_context)

  @pytest.mark.asyncio
  async def test_process_llm_request(
      self, mock_computer_function, tool_context
  ):
    """Test process_llm_request method."""
    tool = ComputerUseTool(
        func=mock_computer_function, screen_size=(1920, 1080)
    )
    llm_request = LlmRequest()

    # Should not raise any exceptions and should do nothing
    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    # Verify llm_request is unchanged (process_llm_request is now a no-op)
    assert llm_request.tools_dict == {}

  def test_inheritance(self, mock_computer_function):
    """Test that ComputerUseTool inherits from FunctionTool."""
    from google.adk.tools.function_tool import FunctionTool

    tool = ComputerUseTool(
        func=mock_computer_function, screen_size=(1920, 1080)
    )
    assert isinstance(tool, FunctionTool)

  def test_custom_screen_size(self, mock_computer_function):
    """Test ComputerUseTool with custom screen size and default virtual screen size."""
    custom_size = (2560, 1440)
    tool = ComputerUseTool(func=mock_computer_function, screen_size=custom_size)

    # Test normalization with custom screen size and default 1000x1000 virtual space
    assert tool._normalize_x(500) == 1280  # 500/1000 * 2560
    assert tool._normalize_y(500) == 720  # 500/1000 * 1440

  def test_custom_screen_size_with_custom_virtual_screen_size(
      self, mock_computer_function
  ):
    """Test ComputerUseTool with both custom screen size and custom virtual screen size."""
    screen_size = (2560, 1440)
    virtual_screen_size = (800, 600)
    tool = ComputerUseTool(
        func=mock_computer_function,
        screen_size=screen_size,
        virtual_screen_size=virtual_screen_size,
    )

    # Test normalization: 400/800 * 2560 = 1280, 300/600 * 1440 = 720
    assert tool._normalize_x(400) == 1280  # 400/800 * 2560
    assert tool._normalize_y(300) == 720  # 300/600 * 1440

    # Test bounds
    assert (
        tool._normalize_x(800) == 2559
    )  # 800/800 * 2560, clamped to screen bounds
    assert (
        tool._normalize_y(600) == 1439
    )  # 600/600 * 1440, clamped to screen bounds

  @pytest.mark.asyncio
  async def test_coordinate_logging(
      self, mock_computer_function, tool_context, caplog
  ):
    """Test that coordinate normalization is logged."""
    import logging

    # Set up a proper signature for the mock function
    def dummy_func(x: int, y: int):
      pass

    # Create a specific mock function for this test
    calls = []
    mock_state = ComputerState(
        screenshot=b"test_screenshot", url="https://example.com"
    )

    async def specific_mock_func(x: int, y: int):
      calls.append((x, y))
      return mock_state

    specific_mock_func.__name__ = "dummy_func"
    specific_mock_func.__signature__ = inspect.signature(dummy_func)
    specific_mock_func.calls = calls

    tool = ComputerUseTool(func=specific_mock_func, screen_size=(1920, 1080))

    # Set the specific logger used by ComputerUseTool to DEBUG level
    logger_name = "google_adk.google.adk.tools.computer_use.computer_use_tool"
    with caplog.at_level(logging.DEBUG, logger=logger_name):
      args = {"x": 500, "y": 300}
      await tool.run_async(args=args, tool_context=tool_context)

    # Check that normalization was logged
    assert "Normalized x: 500 -> 960" in caplog.text
    assert "Normalized y: 300 -> 324" in caplog.text
