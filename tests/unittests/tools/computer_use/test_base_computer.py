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

"""Unit tests for base_computer module."""

from typing import Literal

from google.adk.tools.computer_use.base_computer import BaseComputer
from google.adk.tools.computer_use.base_computer import ComputerEnvironment
from google.adk.tools.computer_use.base_computer import ComputerState
import pytest


class TestComputerEnvironment:
  """Test cases for ComputerEnvironment enum."""

  def test_valid_environments(self):
    """Test valid environment values."""
    assert (
        ComputerEnvironment.ENVIRONMENT_UNSPECIFIED == "ENVIRONMENT_UNSPECIFIED"
    )
    assert ComputerEnvironment.ENVIRONMENT_BROWSER == "ENVIRONMENT_BROWSER"

  def test_invalid_environment_raises(self):
    """Test that invalid environment values raise ValueError."""

    with pytest.raises(ValueError):
      ComputerEnvironment("INVALID_ENVIRONMENT")

  def test_string_representation(self):
    """Test string representation of enum values."""
    assert (
        ComputerEnvironment.ENVIRONMENT_BROWSER.value == "ENVIRONMENT_BROWSER"
    )
    assert (
        ComputerEnvironment.ENVIRONMENT_UNSPECIFIED.value
        == "ENVIRONMENT_UNSPECIFIED"
    )


class TestComputerState:
  """Test cases for ComputerState Pydantic model."""

  def test_default_initialization(self):
    """Test ComputerState with default values."""
    state = ComputerState()
    assert state.screenshot is None
    assert state.url is None

  def test_initialization_with_screenshot(self):
    """Test ComputerState with screenshot data."""
    screenshot_data = b"fake_png_data"
    state = ComputerState(screenshot=screenshot_data)
    assert state.screenshot == screenshot_data
    assert state.url is None

  def test_initialization_with_url(self):
    """Test ComputerState with URL."""
    url = "https://example.com"
    state = ComputerState(url=url)
    assert state.screenshot is None
    assert state.url == url

  def test_initialization_with_all_fields(self):
    """Test ComputerState with all fields provided."""
    screenshot_data = b"fake_png_data"
    url = "https://example.com"
    state = ComputerState(screenshot=screenshot_data, url=url)
    assert state.screenshot == screenshot_data
    assert state.url == url

  def test_field_validation(self):
    """Test field validation for ComputerState."""
    # Test that bytes are accepted for screenshot
    state = ComputerState(screenshot=b"test_data")
    assert state.screenshot == b"test_data"

    # Test that string is accepted for URL
    state = ComputerState(url="https://test.com")
    assert state.url == "https://test.com"

  def test_model_serialization(self):
    """Test that ComputerState can be serialized."""
    state = ComputerState(screenshot=b"test", url="https://example.com")
    # Should not raise an exception
    model_dict = state.model_dump()
    assert "screenshot" in model_dict
    assert "url" in model_dict


class MockComputer(BaseComputer):
  """Mock implementation of BaseComputer for testing."""

  def __init__(self):
    self.initialized = False
    self.closed = False

  async def screen_size(self) -> tuple[int, int]:
    return (1920, 1080)

  async def open_web_browser(self) -> ComputerState:
    return ComputerState(url="https://example.com")

  async def click_at(self, x: int, y: int) -> ComputerState:
    return ComputerState(url="https://example.com")

  async def hover_at(self, x: int, y: int) -> ComputerState:
    return ComputerState(url="https://example.com")

  async def type_text_at(
      self,
      x: int,
      y: int,
      text: str,
      press_enter: bool = True,
      clear_before_typing: bool = True,
  ) -> ComputerState:
    return ComputerState(url="https://example.com")

  async def scroll_document(
      self, direction: Literal["up", "down", "left", "right"]
  ) -> ComputerState:
    return ComputerState(url="https://example.com")

  async def scroll_at(
      self,
      x: int,
      y: int,
      direction: Literal["up", "down", "left", "right"],
      magnitude: int,
  ) -> ComputerState:
    return ComputerState(url="https://example.com")

  async def wait(self, seconds: int) -> ComputerState:
    return ComputerState(url="https://example.com")

  async def go_back(self) -> ComputerState:
    return ComputerState(url="https://example.com")

  async def go_forward(self) -> ComputerState:
    return ComputerState(url="https://example.com")

  async def search(self) -> ComputerState:
    return ComputerState(url="https://search.example.com")

  async def navigate(self, url: str) -> ComputerState:
    return ComputerState(url=url)

  async def key_combination(self, keys: list[str]) -> ComputerState:
    return ComputerState(url="https://example.com")

  async def drag_and_drop(
      self, x: int, y: int, destination_x: int, destination_y: int
  ) -> ComputerState:
    return ComputerState(url="https://example.com")

  async def current_state(self) -> ComputerState:
    return ComputerState(
        url="https://example.com", screenshot=b"screenshot_data"
    )

  async def initialize(self) -> None:
    self.initialized = True

  async def close(self) -> None:
    self.closed = True

  async def environment(self) -> ComputerEnvironment:
    return ComputerEnvironment.ENVIRONMENT_BROWSER


class TestBaseComputer:
  """Test cases for BaseComputer abstract base class."""

  @pytest.fixture
  def mock_computer(self) -> MockComputer:
    """Fixture providing a mock computer implementation."""
    return MockComputer()

  def test_cannot_instantiate_abstract_class(self):
    """Test that BaseComputer cannot be instantiated directly."""
    import pytest

    with pytest.raises(TypeError):
      BaseComputer()  # Should raise TypeError because it's abstract

  @pytest.mark.asyncio
  async def test_screen_size(self, mock_computer):
    """Test screen_size method."""
    size = await mock_computer.screen_size()
    assert size == (1920, 1080)
    assert isinstance(size, tuple)
    assert len(size) == 2

  @pytest.mark.asyncio
  async def test_open_web_browser(self, mock_computer):
    """Test open_web_browser method."""
    state = await mock_computer.open_web_browser()
    assert isinstance(state, ComputerState)
    assert state.url == "https://example.com"

  @pytest.mark.asyncio
  async def test_click_at(self, mock_computer):
    """Test click_at method."""
    state = await mock_computer.click_at(100, 200)
    assert isinstance(state, ComputerState)

  @pytest.mark.asyncio
  async def test_hover_at(self, mock_computer):
    """Test hover_at method."""
    state = await mock_computer.hover_at(150, 250)
    assert isinstance(state, ComputerState)

  @pytest.mark.asyncio
  async def test_type_text_at(self, mock_computer):
    """Test type_text_at method with different parameters."""
    # Test with default parameters
    state = await mock_computer.type_text_at(100, 200, "Hello World")
    assert isinstance(state, ComputerState)

    # Test with custom parameters
    state = await mock_computer.type_text_at(
        100, 200, "Hello", press_enter=False, clear_before_typing=False
    )
    assert isinstance(state, ComputerState)

  @pytest.mark.asyncio
  async def test_scroll_document(self, mock_computer):
    """Test scroll_document method with different directions."""
    directions = ["up", "down", "left", "right"]
    for direction in directions:
      state = await mock_computer.scroll_document(direction)
      assert isinstance(state, ComputerState)

  @pytest.mark.asyncio
  async def test_scroll_at(self, mock_computer):
    """Test scroll_at method."""
    state = await mock_computer.scroll_at(100, 200, "down", 5)
    assert isinstance(state, ComputerState)

  @pytest.mark.asyncio
  async def test_wait(self, mock_computer):
    """Test wait method."""
    state = await mock_computer.wait(5)
    assert isinstance(state, ComputerState)

  @pytest.mark.asyncio
  async def test_go_back(self, mock_computer):
    """Test go_back method."""
    state = await mock_computer.go_back()
    assert isinstance(state, ComputerState)

  @pytest.mark.asyncio
  async def test_go_forward(self, mock_computer):
    """Test go_forward method."""
    state = await mock_computer.go_forward()
    assert isinstance(state, ComputerState)

  @pytest.mark.asyncio
  async def test_search(self, mock_computer):
    """Test search method."""
    state = await mock_computer.search()
    assert isinstance(state, ComputerState)
    assert state.url == "https://search.example.com"

  @pytest.mark.asyncio
  async def test_navigate(self, mock_computer):
    """Test navigate method."""
    test_url = "https://test.example.com"
    state = await mock_computer.navigate(test_url)
    assert isinstance(state, ComputerState)
    assert state.url == test_url

  @pytest.mark.asyncio
  async def test_key_combination(self, mock_computer):
    """Test key_combination method."""
    state = await mock_computer.key_combination(["ctrl", "c"])
    assert isinstance(state, ComputerState)

  @pytest.mark.asyncio
  async def test_drag_and_drop(self, mock_computer):
    """Test drag_and_drop method."""
    state = await mock_computer.drag_and_drop(100, 200, 300, 400)
    assert isinstance(state, ComputerState)

  @pytest.mark.asyncio
  async def test_current_state(self, mock_computer):
    """Test current_state method."""
    state = await mock_computer.current_state()
    assert isinstance(state, ComputerState)
    assert state.url == "https://example.com"
    assert state.screenshot == b"screenshot_data"

  @pytest.mark.asyncio
  async def test_initialize(self, mock_computer):
    """Test initialize method."""
    assert not mock_computer.initialized
    await mock_computer.initialize()
    assert mock_computer.initialized

  @pytest.mark.asyncio
  async def test_close(self, mock_computer):
    """Test close method."""
    assert not mock_computer.closed
    await mock_computer.close()
    assert mock_computer.closed

  @pytest.mark.asyncio
  async def test_environment(self, mock_computer):
    """Test environment method."""
    env = await mock_computer.environment()
    assert env == ComputerEnvironment.ENVIRONMENT_BROWSER
    assert isinstance(env, ComputerEnvironment)

  @pytest.mark.asyncio
  async def test_lifecycle_methods(self, mock_computer):
    """Test the lifecycle of a computer instance."""
    # Initially not initialized or closed
    assert not mock_computer.initialized
    assert not mock_computer.closed

    # Initialize
    await mock_computer.initialize()
    assert mock_computer.initialized
    assert not mock_computer.closed

    # Close
    await mock_computer.close()
    assert mock_computer.initialized
    assert mock_computer.closed
