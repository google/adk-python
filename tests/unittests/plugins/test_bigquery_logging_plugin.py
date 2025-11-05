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

from __future__ import annotations

import asyncio
import datetime
import json
import logging
from unittest import mock

from google.adk.agents import base_agent
from google.adk.agents import callback_context as callback_context_lib
from google.adk.agents import invocation_context as invocation_context_lib
from google.adk.events import event as event_lib
from google.adk.models import llm_request as llm_request_lib
from google.adk.models import llm_response as llm_response_lib
from google.adk.plugins import bigquery_logging_plugin
from google.adk.plugins import plugin_manager as plugin_manager_lib
from google.adk.sessions import base_session_service as base_session_service_lib
from google.adk.sessions import session as session_lib
from google.adk.tools import base_tool as base_tool_lib
from google.adk.tools import tool_context as tool_context_lib
import google.auth
from google.auth import exceptions as auth_exceptions
from google.cloud import bigquery
from google.genai import types
import pytest

BigQueryLoggerConfig = bigquery_logging_plugin.BigQueryLoggerConfig


class PluginTestBase:
  """Base class for plugin tests with common context setup."""

  def setup_method(self, method):
    self.mock_session = mock.create_autospec(session_lib.Session, instance=True)
    self.mock_session.id = "session-123"
    self.mock_session.user_id = "user-456"
    self.mock_session.app_name = "test_app"
    self.mock_session.state = {}
    self.mock_agent = mock.create_autospec(base_agent.BaseAgent, instance=True)
    self.mock_agent.name = "MyTestAgent"
    mock_session_service = mock.create_autospec(
        base_session_service_lib.BaseSessionService, instance=True
    )
    mock_plugin_manager = mock.create_autospec(
        plugin_manager_lib.PluginManager, instance=True
    )
    self.invocation_context = invocation_context_lib.InvocationContext(
        agent=self.mock_agent,
        session=self.mock_session,
        invocation_id="inv-789",
        session_service=mock_session_service,
        plugin_manager=mock_plugin_manager,
    )
    self.callback_context = callback_context_lib.CallbackContext(
        invocation_context=self.invocation_context
    )
    self.tool_context = tool_context_lib.ToolContext(
        invocation_context=self.invocation_context
    )

  def teardown_method(self, method):
    mock.patch.stopall()


class TestBigQueryAgentAnalyticsPlugin(PluginTestBase):
  """Tests for the BigQueryAgentAnalyticsPlugin."""

  def setup_method(self, method):
    super().setup_method(method)
    self.project_id = "test-gcp-project"
    self.dataset_id = "adk_logs"
    self.table_id = "agent_events"

    # Mock Google Auth default credentials
    self._auth_patch = mock.patch.object(google.auth, "default", autospec=True)
    self.mock_auth_default = self._auth_patch.start()
    self.mock_auth_default.return_value = (mock.Mock(), self.project_id)

    # Mock BigQuery Client class
    self._bq_client_patch = mock.patch.object(bigquery, "Client", autospec=True)
    self.mock_bq_client_cls = self._bq_client_patch.start()
    self.mock_bq_client = self.mock_bq_client_cls.return_value
    self.mock_bq_client.create_dataset.return_value = None
    self.mock_bq_client.create_table.return_value = None
    self.mock_bq_client.insert_rows_json.return_value = []  # No errors
    self.mock_table_ref = mock.Mock()
    self.mock_table_ref.dataset_id = self.dataset_id
    self.mock_table_ref.table_id = self.table_id
    self.mock_dataset_ref = mock.Mock()
    self.mock_dataset_ref.table.return_value = self.mock_table_ref
    self.mock_bq_client.dataset.return_value = self.mock_dataset_ref

    # Patch asyncio.to_thread to run the function synchronously
    self._asyncio_to_thread_patch = mock.patch(
        "asyncio.to_thread",
        side_effect=lambda func, *args, **kwargs: func(*args, **kwargs),
    )
    self._asyncio_to_thread_patch.start()

    self.plugin = asyncio.run(self._create_plugin())

  async def _create_plugin(self, config=None):
    plugin = bigquery_logging_plugin.BigQueryAgentAnalyticsPlugin(
        project_id=self.project_id,
        dataset_id=self.dataset_id,
        table_id=self.table_id,
        config=config,
    )
    if config is None or config.enabled:
      # Trigger lazy initialization by calling an async method once.
      await plugin._log_to_bigquery_async({"event_type": "INIT"})
      self.mock_bq_client.insert_rows_json.reset_mock()
    return plugin

  def _get_logged_entry(self):
    """Helper to get the single logged entry from the mocked client."""
    self.mock_bq_client.insert_rows_json.assert_called_once()
    args, _ = self.mock_bq_client.insert_rows_json.call_args
    rows = args[1]
    assert len(rows) == 1
    return rows[0]

  def _assert_common_fields(self, log_entry, event_type):
    assert log_entry["event_type"] == event_type
    assert log_entry["agent"] == "MyTestAgent"
    assert log_entry["session_id"] == "session-123"
    assert log_entry["invocation_id"] == "inv-789"
    assert log_entry["user_id"] == "user-456"
    assert log_entry["timestamp"] is not None

  @pytest.mark.asyncio
  async def test_plugin_disabled(self):
    self.mock_bq_client_cls.reset_mock()
    config = BigQueryLoggerConfig(enabled=False)
    plugin = await self._create_plugin(config)
    user_message = types.Content(parts=[types.Part(text="Test")])
    await plugin.on_user_message_callback(
        invocation_context=self.invocation_context, user_message=user_message
    )
    self.mock_bq_client_cls.assert_not_called()
    self.mock_bq_client.insert_rows_json.assert_not_called()

  @pytest.mark.asyncio
  async def test_event_allowlist(self):
    config = BigQueryLoggerConfig(event_allowlist=["LLM_REQUEST"])
    plugin = await self._create_plugin(config)

    # This should be logged
    llm_request = llm_request_lib.LlmRequest(
        model="gemini-pro",
        contents=[types.Content(parts=[types.Part(text="Prompt")])],
    )
    await plugin.before_model_callback(
        callback_context=self.callback_context, llm_request=llm_request
    )
    self.mock_bq_client.insert_rows_json.assert_called_once()
    self.mock_bq_client.insert_rows_json.reset_mock()

    # This should NOT be logged
    user_message = types.Content(parts=[types.Part(text="What is up?")])
    await plugin.on_user_message_callback(
        invocation_context=self.invocation_context, user_message=user_message
    )
    self.mock_bq_client.insert_rows_json.assert_not_called()

  @pytest.mark.asyncio
  async def test_event_denylist(self):
    config = BigQueryLoggerConfig(event_denylist=["USER_MESSAGE_RECEIVED"])
    plugin = await self._create_plugin(config)

    # This should NOT be logged
    user_message = types.Content(parts=[types.Part(text="What is up?")])
    await plugin.on_user_message_callback(
        invocation_context=self.invocation_context, user_message=user_message
    )
    self.mock_bq_client.insert_rows_json.assert_not_called()

    # This should be logged
    await plugin.before_run_callback(invocation_context=self.invocation_context)
    self.mock_bq_client.insert_rows_json.assert_called_once()

  @pytest.mark.asyncio
  async def test_content_formatter(self):
    def redact_content(content):
      return "[REDACTED]"

    config = BigQueryLoggerConfig(content_formatter=redact_content)
    plugin = await self._create_plugin(config)

    user_message = types.Content(parts=[types.Part(text="Secret message")])
    await plugin.on_user_message_callback(
        invocation_context=self.invocation_context, user_message=user_message
    )

    log_entry = self._get_logged_entry()
    self._assert_common_fields(log_entry, "USER_MESSAGE_RECEIVED")
    assert log_entry["content"] == "[REDACTED]"

  @pytest.mark.asyncio
  async def test_content_formatter_error(self):
    def error_formatter(content):
      raise ValueError("Formatter failed")

    config = BigQueryLoggerConfig(content_formatter=error_formatter)
    plugin = await self._create_plugin(config)

    user_message = types.Content(parts=[types.Part(text="Test")])
    with mock.patch.object(logging, "warning") as mock_log_warning:
      await plugin.on_user_message_callback(
          invocation_context=self.invocation_context, user_message=user_message
      )
      mock_log_warning.assert_called_once_with(
          "Error applying custom content formatter for event type %s: %s",
          "USER_MESSAGE_RECEIVED",
          mock.ANY,
      )

    log_entry = self._get_logged_entry()
    # Content should be a string, even if formatter failed
    assert isinstance(log_entry["content"], str)
    assert "User Content: text: 'Test'" in log_entry["content"]

  @pytest.mark.asyncio
  async def test_on_user_message_callback_logs_correctly(self):
    user_message = types.Content(parts=[types.Part(text="What is up?")])
    await self.plugin.on_user_message_callback(
        invocation_context=self.invocation_context, user_message=user_message
    )

    log_entry = self._get_logged_entry()
    self._assert_common_fields(log_entry, "USER_MESSAGE_RECEIVED")
    assert log_entry["content"] == "User Content: text: 'What is up?'"

  @pytest.mark.asyncio
  async def test_on_event_callback_tool_call(self):
    tool_fc = types.FunctionCall(name="get_weather", args={"location": "Paris"})
    event = event_lib.Event(
        author="MyTestAgent",
        content=types.Content(parts=[types.Part(function_call=tool_fc)]),
        timestamp=datetime.datetime(
            2025, 10, 22, 10, 0, 0, tzinfo=datetime.timezone.utc
        ).timestamp(),
    )
    await self.plugin.on_event_callback(
        invocation_context=self.invocation_context, event=event
    )

    log_entry = self._get_logged_entry()
    self._assert_common_fields(log_entry, "TOOL_CALL")
    logged_content = json.loads(log_entry["content"])
    assert logged_content[0]["function_call"]["args"] == {"location": "Paris"}
    assert logged_content[0]["function_call"]["name"] == "get_weather"
    assert log_entry["timestamp"] == "2025-10-22T10:00:00+00:00"

  @pytest.mark.asyncio
  async def test_on_event_callback_model_response(self):
    event = event_lib.Event(
        author="MyTestAgent",
        content=types.Content(parts=[types.Part(text="Hello there!")]),
        timestamp=datetime.datetime(
            2025, 10, 22, 11, 0, 0, tzinfo=datetime.timezone.utc
        ).timestamp(),
    )
    await self.plugin.on_event_callback(
        invocation_context=self.invocation_context, event=event
    )

    log_entry = self._get_logged_entry()
    self._assert_common_fields(log_entry, "MODEL_RESPONSE")
    logged_content = json.loads(log_entry["content"])
    assert logged_content[0]["text"] == "Hello there!"
    assert log_entry["timestamp"] == "2025-10-22T11:00:00+00:00"

  @pytest.mark.asyncio
  async def test_bigquery_client_initialization_failure(self):
    # Simulate auth failure
    self.mock_auth_default.side_effect = auth_exceptions.GoogleAuthError(
        "Auth failed"
    )
    self.mock_bq_client.insert_rows_json.reset_mock()

    # Re-instantiate the plugin so init is re-attempted
    plugin_with_fail = bigquery_logging_plugin.BigQueryAgentAnalyticsPlugin(
        project_id=self.project_id,
        dataset_id=self.dataset_id,
        table_id=self.table_id,
    )

    # Trigger a callback; initialization happens lazily
    with mock.patch.object(logging, "exception") as mock_log_exception:
      await plugin_with_fail.before_run_callback(
          invocation_context=self.invocation_context
      )
      mock_log_exception.assert_called_once()

    # Ensure insert_rows_json was never called because init failed
    self.mock_bq_client.insert_rows_json.assert_not_called()

  @pytest.mark.asyncio
  async def test_bigquery_insert_error_does_not_raise(self):
    # Simulate an insert error in the future result
    self.mock_bq_client.insert_rows_json.return_value = [{"errors": ["error"]}]

    with mock.patch.object(logging, "error") as mock_log_error:
      await self.plugin.on_user_message_callback(
          invocation_context=self.invocation_context,
          user_message=types.Content(parts=[types.Part(text="Test")]),
      )
      # The plugin should handle the error internally without raising
      mock_log_error.assert_called_with(
          "Errors occurred while inserting to BigQuery table %s.%s: %s",
          self.dataset_id,
          self.table_id,
          [{"errors": ["error"]}],
      )

    self.mock_bq_client.insert_rows_json.assert_called_once()

  @pytest.mark.asyncio
  async def test_before_run_callback_logs_correctly(self):
    await self.plugin.before_run_callback(
        invocation_context=self.invocation_context
    )
    log_entry = self._get_logged_entry()
    self._assert_common_fields(log_entry, "INVOCATION_STARTING")
    assert log_entry["content"] is None

  @pytest.mark.asyncio
  async def test_after_run_callback_logs_correctly(self):
    await self.plugin.after_run_callback(
        invocation_context=self.invocation_context
    )
    log_entry = self._get_logged_entry()
    self._assert_common_fields(log_entry, "INVOCATION_COMPLETED")
    assert log_entry["content"] is None

  @pytest.mark.asyncio
  async def test_before_agent_callback_logs_correctly(self):
    await self.plugin.before_agent_callback(
        agent=self.mock_agent, callback_context=self.callback_context
    )
    log_entry = self._get_logged_entry()
    self._assert_common_fields(log_entry, "AGENT_STARTING")
    assert log_entry["content"] == "Agent Name: MyTestAgent"

  @pytest.mark.asyncio
  async def test_after_agent_callback_logs_correctly(self):
    await self.plugin.after_agent_callback(
        agent=self.mock_agent, callback_context=self.callback_context
    )
    log_entry = self._get_logged_entry()
    self._assert_common_fields(log_entry, "AGENT_COMPLETED")
    assert log_entry["content"] == "Agent Name: MyTestAgent"

  @pytest.mark.asyncio
  async def test_before_model_callback_logs_correctly(self):
    llm_request = llm_request_lib.LlmRequest(
        model="gemini-pro",
        contents=[types.Content(parts=[types.Part(text="Prompt")])],
        config=types.GenerateContentConfig(
            temperature=0.5,
            top_p=0.9,
            max_output_tokens=100,
            system_instruction="Be helpful",
        ),
        tools_dict={
            "my_tool": mock.create_autospec(
                base_tool_lib.BaseTool, instance=True
            )
        },  # Fixed mock
    )

    await self.plugin.before_model_callback(
        callback_context=self.callback_context, llm_request=llm_request
    )
    log_entry = self._get_logged_entry()
    self._assert_common_fields(log_entry, "LLM_REQUEST")
    assert "Model: gemini-pro" in log_entry["content"]
    assert "System Prompt: Be helpful" in log_entry["content"]
    assert (
        "Params: {temperature=0.5, top_p=0.9, max_output_tokens=100}"
        in log_entry["content"]
    )
    assert "Available Tools: ['my_tool']" in log_entry["content"]

  @pytest.mark.asyncio
  async def test_after_model_callback_text_response(self):
    llm_response = llm_response_lib.LlmResponse(
        content=types.Content(parts=[types.Part(text="Model response")]),
        usage_metadata=types.UsageMetadata(
            prompt_token_count=10,
            total_token_count=15,
        ),
    )
    await self.plugin.after_model_callback(
        callback_context=self.callback_context, llm_response=llm_response
    )
    log_entry = self._get_logged_entry()
    self._assert_common_fields(log_entry, "LLM_RESPONSE")
    assert (
        "Tool Name: text_response, text: 'Model response'"
        in log_entry["content"]
    )
    # Adjusted assertion to expect None for candidates
    assert "Token Usage: {prompt: 10" in log_entry["content"]
    assert log_entry["error_message"] is None

  @pytest.mark.asyncio
  async def test_after_model_callback_tool_call(self):
    llm_response = llm_response_lib.LlmResponse(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(name="tool1", args={})
                )
            ]
        ),
    )
    await self.plugin.after_model_callback(
        callback_context=self.callback_context, llm_response=llm_response
    )
    log_entry = self._get_logged_entry()
    self._assert_common_fields(log_entry, "LLM_RESPONSE")
    assert "Tool Name: tool1" in log_entry["content"]

  @pytest.mark.asyncio
  async def test_before_tool_callback_logs_correctly(self):
    mock_tool = mock.create_autospec(base_tool_lib.BaseTool, instance=True)
    mock_tool.name = "MyTool"
    mock_tool.description = "Does something"
    tool_args = {"param": "value"}
    await self.plugin.before_tool_callback(
        tool=mock_tool, tool_args=tool_args, tool_context=self.tool_context
    )
    log_entry = self._get_logged_entry()
    self._assert_common_fields(log_entry, "TOOL_STARTING")
    assert "Tool Name: MyTool" in log_entry["content"]
    assert "Description: Does something" in log_entry["content"]
    assert "Arguments: {'param': 'value'}" in log_entry["content"]

  @pytest.mark.asyncio
  async def test_after_tool_callback_logs_correctly(self):
    mock_tool = mock.create_autospec(base_tool_lib.BaseTool, instance=True)
    mock_tool.name = "MyTool"
    tool_args = {"param": "value"}
    result = {"status": "success"}
    await self.plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=tool_args,
        tool_context=self.tool_context,
        result=result,
    )
    log_entry = self._get_logged_entry()
    self._assert_common_fields(log_entry, "TOOL_COMPLETED")
    assert "Tool Name: MyTool" in log_entry["content"]
    assert "Result: {'status': 'success'}" in log_entry["content"]

  @pytest.mark.asyncio
  async def test_on_model_error_callback_logs_correctly(self):
    llm_request = llm_request_lib.LlmRequest(
        model="gemini-pro",
        contents=[types.Content(parts=[types.Part(text="Prompt")])],
    )
    error = ValueError("LLM failed")
    await self.plugin.on_model_error_callback(
        callback_context=self.callback_context,
        llm_request=llm_request,
        error=error,
    )
    log_entry = self._get_logged_entry()
    self._assert_common_fields(log_entry, "LLM_ERROR")
    assert (
        log_entry["content"] is None
        or "Request Content: " in log_entry["content"]
    )
    assert log_entry["error_message"] == "LLM failed"

  @pytest.mark.asyncio
  async def test_on_tool_error_callback_logs_correctly(self):
    mock_tool = mock.create_autospec(base_tool_lib.BaseTool, instance=True)
    mock_tool.name = "MyTool"
    tool_args = {"param": "value"}
    error = TimeoutError("Tool timed out")
    await self.plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=tool_args,
        tool_context=self.tool_context,
        error=error,
    )
    log_entry = self._get_logged_entry()
    self._assert_common_fields(log_entry, "TOOL_ERROR")
    assert "Tool Name: MyTool" in log_entry["content"]
    assert "Arguments: {'param': 'value'}" in log_entry["content"]
    assert log_entry["error_message"] == "Tool timed out"
