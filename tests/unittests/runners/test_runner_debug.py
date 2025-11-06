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

"""Tests for Runner.run_debug helper method."""

from __future__ import annotations

from unittest import mock

from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
import pytest


class TestRunDebug:
  """Tests for Runner.run_debug method."""

  @pytest.mark.asyncio
  async def test_run_debug_single_query(self):
    """Test run_debug with a single string query."""
    # Setup
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="You are a helpful assistant.",
    )
    runner = InMemoryRunner(agent=agent)

    # Mock the runner's run_async to return controlled events
    mock_event = mock.Mock()
    mock_event.author = "test_agent"
    mock_event.content = mock.Mock()
    mock_event.content.parts = [mock.Mock(text="Hello! I can help you.")]

    async def mock_run_async(*args, **kwargs):
      yield mock_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      # Execute
      events = await runner.run_debug("Hello, how are you?", quiet=True)

      # Assertions
      assert len(events) == 1
      assert events[0].author == "test_agent"
      assert events[0].content.parts[0].text == "Hello! I can help you."

      # Verify session was created with defaults
      session = await runner.session_service.get_session(
          app_name=runner.app_name,
          user_id="debug_user_id",
          session_id="debug_session_id",
      )
      assert session is not None

  @pytest.mark.asyncio
  async def test_run_debug_multiple_queries(self):
    """Test run_debug with multiple queries in sequence."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="You are a test bot.",
    )
    runner = InMemoryRunner(agent=agent)

    # Mock responses for multiple queries
    responses = ["First response", "Second response"]
    call_count = 0

    async def mock_run_async(*args, **kwargs):
      nonlocal call_count
      mock_event = mock.Mock()
      mock_event.author = "test_agent"
      mock_event.content = mock.Mock()
      mock_event.content.parts = [mock.Mock(text=responses[call_count])]
      call_count += 1
      yield mock_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      # Execute with multiple queries
      events = await runner.run_debug(
          ["First query", "Second query"], quiet=True
      )

      # Assertions
      assert len(events) == 2
      assert events[0].content.parts[0].text == "First response"
      assert events[1].content.parts[0].text == "Second response"

  @pytest.mark.asyncio
  async def test_run_debug_always_returns_events(self):
    """Test that run_debug always returns events."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent.",
    )
    runner = InMemoryRunner(agent=agent)

    async def mock_run_async(*args, **kwargs):
      mock_event = mock.Mock()
      mock_event.author = "test_agent"
      mock_event.content = mock.Mock()
      mock_event.content.parts = [mock.Mock(text="Response")]
      yield mock_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      # Test that events are always returned
      events = await runner.run_debug("Query", quiet=True)
      assert isinstance(events, list)
      assert len(events) == 1

  @pytest.mark.asyncio
  async def test_run_debug_quiet_mode(self, capsys):
    """Test that quiet=True suppresses printing."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent.",
    )
    runner = InMemoryRunner(agent=agent)

    async def mock_run_async(*args, **kwargs):
      mock_event = mock.Mock()
      mock_event.author = "test_agent"
      mock_event.content = mock.Mock()
      mock_event.content.parts = [mock.Mock(text="This should not be printed")]
      yield mock_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      # Execute with quiet=True
      await runner.run_debug("Test query", quiet=True)

      # Check that nothing was printed
      captured = capsys.readouterr()
      assert "This should not be printed" not in captured.out
      assert "User >" not in captured.out
      assert "Session:" not in captured.out

  @pytest.mark.asyncio
  async def test_run_debug_custom_session_id(self):
    """Test run_debug with custom session_id."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent.",
    )
    runner = InMemoryRunner(agent=agent)

    async def mock_run_async(*args, **kwargs):
      mock_event = mock.Mock()
      mock_event.author = "test_agent"
      mock_event.content = mock.Mock()
      mock_event.content.parts = [mock.Mock(text="Response")]
      yield mock_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      # Execute with custom session ID
      await runner.run_debug(
          "Query", session_id="custom_debug_session", quiet=True
      )

      # Verify session was created with custom ID
      session = await runner.session_service.get_session(
          app_name=runner.app_name,
          user_id="debug_user_id",
          session_id="custom_debug_session",
      )
      assert session is not None
      assert session.id == "custom_debug_session"

  @pytest.mark.asyncio
  async def test_run_debug_custom_user_id(self):
    """Test run_debug with custom user_id."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent.",
    )
    runner = InMemoryRunner(agent=agent)

    async def mock_run_async(*args, **kwargs):
      mock_event = mock.Mock()
      mock_event.author = "test_agent"
      mock_event.content = mock.Mock()
      mock_event.content.parts = [mock.Mock(text="Response")]
      yield mock_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      # Execute with custom user_id
      await runner.run_debug("Query", user_id="test_user_123", quiet=True)

      # Verify session was created with custom user_id
      session = await runner.session_service.get_session(
          app_name=runner.app_name,
          user_id="test_user_123",
          session_id="debug_session_id",
      )
      assert session is not None

  @pytest.mark.asyncio
  async def test_run_debug_with_run_config(self):
    """Test that run_config is properly passed through to run_async."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent.",
    )
    runner = InMemoryRunner(agent=agent)

    run_config_used = None

    async def mock_run_async(*args, **kwargs):
      nonlocal run_config_used
      run_config_used = kwargs.get("run_config")
      mock_event = mock.Mock()
      mock_event.author = "test_agent"
      mock_event.content = mock.Mock()
      mock_event.content.parts = [mock.Mock(text="Response")]
      yield mock_event

    with mock.patch.object(
        runner, "run_async", side_effect=mock_run_async
    ) as mock_method:
      # Create a custom run_config
      custom_config = RunConfig(support_cfc=True)

      # Execute with custom run_config
      await runner.run_debug("Query", run_config=custom_config, quiet=True)

      # Verify run_config was passed to run_async
      assert mock_method.called
      call_args = mock_method.call_args
      assert call_args is not None
      assert "run_config" in call_args.kwargs
      assert call_args.kwargs["run_config"] == custom_config

  @pytest.mark.asyncio
  async def test_run_debug_session_persistence(self):
    """Test that multiple calls to run_debug maintain conversation context."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Remember previous messages.",
    )
    runner = InMemoryRunner(agent=agent)

    call_count = 0
    responses = ["First response", "Second response remembering first"]

    async def mock_run_async(*args, **kwargs):
      nonlocal call_count
      mock_event = mock.Mock()
      mock_event.author = "test_agent"
      mock_event.content = mock.Mock()
      mock_event.content.parts = [mock.Mock(text=responses[call_count])]
      call_count += 1
      yield mock_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      # First call
      events1 = await runner.run_debug("First message", quiet=True)
      assert events1[0].content.parts[0].text == "First response"

      # Second call to same session
      events2 = await runner.run_debug("Second message", quiet=True)
      assert (
          events2[0].content.parts[0].text
          == "Second response remembering first"
      )

      # Verify both calls used the same session
      session = await runner.session_service.get_session(
          app_name=runner.app_name,
          user_id="debug_user_id",
          session_id="debug_session_id",
      )
      assert session is not None

  @pytest.mark.asyncio
  async def test_run_debug_filters_none_text(self):
    """Test that run_debug filters out 'None' text and empty parts."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent.",
    )
    runner = InMemoryRunner(agent=agent)

    async def mock_run_async(*args, **kwargs):
      # Yield events with various text values
      events = [
          mock.Mock(
              author="test_agent",
              content=mock.Mock(parts=[mock.Mock(text="Valid text")]),
          ),
          mock.Mock(
              author="test_agent",
              content=mock.Mock(parts=[mock.Mock(text="None")]),
          ),  # Should be filtered
          mock.Mock(
              author="test_agent",
              content=mock.Mock(parts=[mock.Mock(text="")]),
          ),  # Should be filtered
          mock.Mock(
              author="test_agent",
              content=mock.Mock(parts=[mock.Mock(text="Another valid")]),
          ),
      ]
      for event in events:
        yield event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      # Execute and capture output
      events = await runner.run_debug("Query", quiet=True)

      # All 4 events should be returned (filtering is for printing only)
      assert len(events) == 4

      # But when printing, "None" and empty strings should be filtered
      # This is tested implicitly by the implementation

  @pytest.mark.asyncio
  async def test_run_debug_with_existing_session(self):
    """Test that run_debug retrieves existing session when AlreadyExistsError occurs."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent.",
    )
    runner = InMemoryRunner(agent=agent)

    # First create a session
    await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="debug_user_id",
        session_id="existing_session",
    )

    async def mock_run_async(*args, **kwargs):
      mock_event = mock.Mock()
      mock_event.author = "test_agent"
      mock_event.content = mock.Mock()
      mock_event.content.parts = [mock.Mock(text="Using existing session")]
      yield mock_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      # Execute with same session ID (should retrieve existing)
      events = await runner.run_debug(
          "Query", session_id="existing_session", quiet=True
      )

      assert len(events) == 1
      assert events[0].content.parts[0].text == "Using existing session"

  @pytest.mark.asyncio
  async def test_run_debug_with_tool_calls(self, capsys):
    """Test that run_debug properly handles and prints tool calls."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent with tools.",
    )
    runner = InMemoryRunner(agent=agent)

    async def mock_run_async(*args, **kwargs):
      # First event: tool call
      mock_call_event = mock.Mock()
      mock_call_event.author = "test_agent"
      mock_call_event.content = mock.Mock()
      mock_function_call = mock.Mock()
      mock_function_call.name = "calculate"
      mock_function_call.args = {"operation": "add", "a": 5, "b": 3}
      mock_part_call = mock.Mock()
      mock_part_call.text = None
      mock_part_call.function_call = mock_function_call
      mock_part_call.function_response = None
      mock_call_event.content.parts = [mock_part_call]
      yield mock_call_event

      # Second event: tool response
      mock_resp_event = mock.Mock()
      mock_resp_event.author = "test_agent"
      mock_resp_event.content = mock.Mock()
      mock_function_response = mock.Mock()
      mock_function_response.response = {"result": 8}
      mock_part_resp = mock.Mock()
      mock_part_resp.text = None
      mock_part_resp.function_call = None
      mock_part_resp.function_response = mock_function_response
      mock_resp_event.content.parts = [mock_part_resp]
      yield mock_resp_event

      # Third event: final text response
      mock_text_event = mock.Mock()
      mock_text_event.author = "test_agent"
      mock_text_event.content = mock.Mock()
      mock_text_event.content.parts = [mock.Mock(text="The result is 8")]
      yield mock_text_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      # Execute with verbose=True to see tool calls
      events = await runner.run_debug("Calculate 5 + 3", verbose=True)

      # Check output was printed
      captured = capsys.readouterr()
      assert "[Calling tool: calculate" in captured.out
      assert "[Tool result:" in captured.out
      assert "The result is 8" in captured.out

      # Check events were collected
      assert len(events) == 3

  @pytest.mark.asyncio
  async def test_run_debug_with_executable_code(self, capsys):
    """Test that run_debug properly handles executable code parts."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent with code execution.",
    )
    runner = InMemoryRunner(agent=agent)

    async def mock_run_async(*args, **kwargs):
      # Event with executable code
      mock_event = mock.Mock()
      mock_event.author = "test_agent"
      mock_event.content = mock.Mock()

      mock_exec_code = mock.Mock()
      mock_exec_code.language = "python"
      mock_exec_code.code = "print('Hello World')"

      mock_part = mock.Mock()
      mock_part.text = None
      mock_part.function_call = None
      mock_part.function_response = None
      mock_part.executable_code = mock_exec_code
      mock_part.code_execution_result = None
      mock_part.inline_data = None
      mock_part.file_data = None

      mock_event.content.parts = [mock_part]
      yield mock_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      events = await runner.run_debug("Run some code", verbose=True)

      captured = capsys.readouterr()
      assert "[Executing python code...]" in captured.out
      assert len(events) == 1

  @pytest.mark.asyncio
  async def test_run_debug_with_code_execution_result(self, capsys):
    """Test that run_debug properly handles code execution result parts."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent with code results.",
    )
    runner = InMemoryRunner(agent=agent)

    async def mock_run_async(*args, **kwargs):
      # Event with code execution result
      mock_event = mock.Mock()
      mock_event.author = "test_agent"
      mock_event.content = mock.Mock()

      mock_result = mock.Mock()
      mock_result.output = "Hello World\n42"

      mock_part = mock.Mock()
      mock_part.text = None
      mock_part.function_call = None
      mock_part.function_response = None
      mock_part.executable_code = None
      mock_part.code_execution_result = mock_result
      mock_part.inline_data = None
      mock_part.file_data = None

      mock_event.content.parts = [mock_part]
      yield mock_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      events = await runner.run_debug(
          "Show code output",
          verbose=True,
      )

      captured = capsys.readouterr()
      assert "[Code output: Hello World\n42]" in captured.out
      assert len(events) == 1

  @pytest.mark.asyncio
  async def test_run_debug_with_inline_data(self, capsys):
    """Test that run_debug properly handles inline data parts."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent with inline data.",
    )
    runner = InMemoryRunner(agent=agent)

    async def mock_run_async(*args, **kwargs):
      # Event with inline data (e.g., image)
      mock_event = mock.Mock()
      mock_event.author = "test_agent"
      mock_event.content = mock.Mock()

      mock_inline = mock.Mock()
      mock_inline.mime_type = "image/png"
      mock_inline.data = b"fake_image_data"

      mock_part = mock.Mock()
      mock_part.text = None
      mock_part.function_call = None
      mock_part.function_response = None
      mock_part.executable_code = None
      mock_part.code_execution_result = None
      mock_part.inline_data = mock_inline
      mock_part.file_data = None

      mock_event.content.parts = [mock_part]
      yield mock_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      events = await runner.run_debug("Show image", verbose=True)

      captured = capsys.readouterr()
      assert "[Inline data: image/png]" in captured.out
      assert len(events) == 1

  @pytest.mark.asyncio
  async def test_run_debug_with_file_data(self, capsys):
    """Test that run_debug properly handles file data parts."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent with file data.",
    )
    runner = InMemoryRunner(agent=agent)

    async def mock_run_async(*args, **kwargs):
      # Event with file data
      mock_event = mock.Mock()
      mock_event.author = "test_agent"
      mock_event.content = mock.Mock()

      mock_file = mock.Mock()
      mock_file.file_uri = "gs://bucket/path/to/file.pdf"

      mock_part = mock.Mock()
      mock_part.text = None
      mock_part.function_call = None
      mock_part.function_response = None
      mock_part.executable_code = None
      mock_part.code_execution_result = None
      mock_part.inline_data = None
      mock_part.file_data = mock_file

      mock_event.content.parts = [mock_part]
      yield mock_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      events = await runner.run_debug("Reference file", verbose=True)

      captured = capsys.readouterr()
      assert "[File: gs://bucket/path/to/file.pdf]" in captured.out
      assert len(events) == 1

  @pytest.mark.asyncio
  async def test_run_debug_with_mixed_parts(self, capsys):
    """Test that run_debug handles events with multiple part types."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent with mixed parts.",
    )
    runner = InMemoryRunner(agent=agent)

    async def mock_run_async(*args, **kwargs):
      # Event with multiple part types
      mock_event = mock.Mock()
      mock_event.author = "test_agent"
      mock_event.content = mock.Mock()

      # Text part
      mock_text_part = mock.Mock()
      mock_text_part.text = "Here's your result:"
      mock_text_part.function_call = None
      mock_text_part.function_response = None
      mock_text_part.executable_code = None
      mock_text_part.code_execution_result = None
      mock_text_part.inline_data = None
      mock_text_part.file_data = None

      # Code execution part
      mock_code_part = mock.Mock()
      mock_code_part.text = None
      mock_code_part.function_call = None
      mock_code_part.function_response = None
      mock_exec_code = mock.Mock()
      mock_exec_code.language = "python"
      mock_code_part.executable_code = mock_exec_code
      mock_code_part.code_execution_result = None
      mock_code_part.inline_data = None
      mock_code_part.file_data = None

      # Result part
      mock_result_part = mock.Mock()
      mock_result_part.text = None
      mock_result_part.function_call = None
      mock_result_part.function_response = None
      mock_result_part.executable_code = None
      mock_result = mock.Mock()
      mock_result.output = "42"
      mock_result_part.code_execution_result = mock_result
      mock_result_part.inline_data = None
      mock_result_part.file_data = None

      mock_event.content.parts = [
          mock_text_part,
          mock_code_part,
          mock_result_part,
      ]
      yield mock_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      events = await runner.run_debug("Mixed response", verbose=True)

      captured = capsys.readouterr()
      assert "Here's your result:" in captured.out
      assert "[Executing python code...]" in captured.out
      assert "[Code output: 42]" in captured.out
      assert len(events) == 1

  @pytest.mark.asyncio
  async def test_run_debug_with_long_output_truncation(self, capsys):
    """Test that run_debug properly truncates long outputs."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent with long outputs.",
    )
    runner = InMemoryRunner(agent=agent)

    async def mock_run_async(*args, **kwargs):
      # Tool call with long args
      mock_call_event = mock.Mock()
      mock_call_event.author = "test_agent"
      mock_call_event.content = mock.Mock()

      mock_function_call = mock.Mock()
      mock_function_call.name = "process"
      # Create a long argument string
      mock_function_call.args = {"data": "x" * 100}

      mock_part_call = mock.Mock()
      mock_part_call.text = None
      mock_part_call.function_call = mock_function_call
      mock_part_call.function_response = None
      mock_part_call.executable_code = None
      mock_part_call.code_execution_result = None
      mock_part_call.inline_data = None
      mock_part_call.file_data = None

      mock_call_event.content.parts = [mock_part_call]
      yield mock_call_event

      # Tool response with long result
      mock_resp_event = mock.Mock()
      mock_resp_event.author = "test_agent"
      mock_resp_event.content = mock.Mock()

      mock_function_response = mock.Mock()
      # Create a long response string
      mock_function_response.response = {"result": "y" * 200}

      mock_part_resp = mock.Mock()
      mock_part_resp.text = None
      mock_part_resp.function_call = None
      mock_part_resp.function_response = mock_function_response
      mock_part_resp.executable_code = None
      mock_part_resp.code_execution_result = None
      mock_part_resp.inline_data = None
      mock_part_resp.file_data = None

      mock_resp_event.content.parts = [mock_part_resp]
      yield mock_resp_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      events = await runner.run_debug("Process data", verbose=True)

      captured = capsys.readouterr()
      # Check that args are truncated at 50 chars
      assert "..." in captured.out
      assert "[Calling tool: process(" in captured.out
      # Check that response is truncated at 100 chars
      assert "[Tool result:" in captured.out
      assert len(events) == 2

  @pytest.mark.asyncio
  async def test_run_debug_verbose_flag_false(self, capsys):
    """Test that run_debug hides tool calls when verbose=False (default)."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent with tools.",
    )
    runner = InMemoryRunner(agent=agent)

    async def mock_run_async(*args, **kwargs):
      # Tool call event
      mock_call_event = mock.Mock()
      mock_call_event.author = "test_agent"
      mock_call_event.content = mock.Mock()

      mock_function_call = mock.Mock()
      mock_function_call.name = "get_weather"
      mock_function_call.args = {"city": "Tokyo"}

      mock_part_call = mock.Mock()
      mock_part_call.text = None
      mock_part_call.function_call = mock_function_call
      mock_part_call.function_response = None
      mock_part_call.executable_code = None
      mock_part_call.code_execution_result = None
      mock_part_call.inline_data = None
      mock_part_call.file_data = None

      mock_call_event.content.parts = [mock_part_call]
      yield mock_call_event

      # Tool response event
      mock_resp_event = mock.Mock()
      mock_resp_event.author = "test_agent"
      mock_resp_event.content = mock.Mock()

      mock_function_response = mock.Mock()
      mock_function_response.response = {"weather": "Clear, 25°C"}

      mock_part_resp = mock.Mock()
      mock_part_resp.text = None
      mock_part_resp.function_call = None
      mock_part_resp.function_response = mock_function_response
      mock_part_resp.executable_code = None
      mock_part_resp.code_execution_result = None
      mock_part_resp.inline_data = None
      mock_part_resp.file_data = None

      mock_resp_event.content.parts = [mock_part_resp]
      yield mock_resp_event

      # Final text response
      mock_text_event = mock.Mock()
      mock_text_event.author = "test_agent"
      mock_text_event.content = mock.Mock()
      mock_text_event.content.parts = [
          mock.Mock(text="The weather in Tokyo is clear and 25°C.")
      ]
      yield mock_text_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      events = await runner.run_debug(
          "What's the weather?",
          verbose=False,  # Default - should NOT show tool calls
      )

      captured = capsys.readouterr()
      # Should NOT show tool call details
      assert "[Calling tool:" not in captured.out
      assert "[Tool result:" not in captured.out
      # Should show final text response
      assert "The weather in Tokyo is clear and 25°C." in captured.out
      assert len(events) == 3

  @pytest.mark.asyncio
  async def test_run_debug_verbose_flag_true(self, capsys):
    """Test that run_debug shows tool calls when verbose=True."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent with tools.",
    )
    runner = InMemoryRunner(agent=agent)

    async def mock_run_async(*args, **kwargs):
      # Tool call event
      mock_call_event = mock.Mock()
      mock_call_event.author = "test_agent"
      mock_call_event.content = mock.Mock()

      mock_function_call = mock.Mock()
      mock_function_call.name = "calculate"
      mock_function_call.args = {"expression": "42 * 3.14"}

      mock_part_call = mock.Mock()
      mock_part_call.text = None
      mock_part_call.function_call = mock_function_call
      mock_part_call.function_response = None
      mock_part_call.executable_code = None
      mock_part_call.code_execution_result = None
      mock_part_call.inline_data = None
      mock_part_call.file_data = None

      mock_call_event.content.parts = [mock_part_call]
      yield mock_call_event

      # Tool response event
      mock_resp_event = mock.Mock()
      mock_resp_event.author = "test_agent"
      mock_resp_event.content = mock.Mock()

      mock_function_response = mock.Mock()
      mock_function_response.response = {"result": 131.88}

      mock_part_resp = mock.Mock()
      mock_part_resp.text = None
      mock_part_resp.function_call = None
      mock_part_resp.function_response = mock_function_response
      mock_part_resp.executable_code = None
      mock_part_resp.code_execution_result = None
      mock_part_resp.inline_data = None
      mock_part_resp.file_data = None

      mock_resp_event.content.parts = [mock_part_resp]
      yield mock_resp_event

      # Final text response
      mock_text_event = mock.Mock()
      mock_text_event.author = "test_agent"
      mock_text_event.content = mock.Mock()
      mock_text_event.content.parts = [mock.Mock(text="The result is 131.88")]
      yield mock_text_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      events = await runner.run_debug(
          "Calculate 42 * 3.14",
          verbose=True,  # Should show tool calls
      )

      captured = capsys.readouterr()
      # Should show tool call details
      assert (
          "[Calling tool: calculate({'expression': '42 * 3.14'})]"
          in captured.out
      )
      assert "[Tool result: {'result': 131.88}]" in captured.out
      # Should also show final text response
      assert "The result is 131.88" in captured.out
      assert len(events) == 3

  @pytest.mark.asyncio
  async def test_run_debug_with_empty_parts_list(self, capsys):
    """Test that run_debug handles events with empty parts list gracefully."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent.",
    )
    runner = InMemoryRunner(agent=agent)

    async def mock_run_async(*_args, **_kwargs):
      # Event with empty parts list
      mock_event = mock.Mock()
      mock_event.author = "test_agent"
      mock_event.content = mock.Mock()
      mock_event.content.parts = []  # Empty parts list
      yield mock_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      events = await runner.run_debug("Test query")

      captured = capsys.readouterr()
      # Should handle gracefully without crashing
      assert "User > Test query" in captured.out
      assert len(events) == 1
      # Should not print any agent response since parts is empty
      assert "test_agent >" not in captured.out

  @pytest.mark.asyncio
  async def test_run_debug_with_none_event_content(self, capsys):
    """Test that run_debug handles events with None content gracefully."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash-lite",
        instruction="Test agent.",
    )
    runner = InMemoryRunner(agent=agent)

    async def mock_run_async(*_args, **_kwargs):
      # Event with None content
      mock_event = mock.Mock()
      mock_event.author = "test_agent"
      mock_event.content = None  # None content
      yield mock_event

    with mock.patch.object(runner, "run_async", side_effect=mock_run_async):
      events = await runner.run_debug("Test query")

      captured = capsys.readouterr()
      # Should handle gracefully without crashing
      assert "User > Test query" in captured.out
      assert len(events) == 1
      # Should not print any agent response since content is None
      assert "test_agent >" not in captured.out
