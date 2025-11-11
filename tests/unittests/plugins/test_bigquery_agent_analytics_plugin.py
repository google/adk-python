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
from google.adk.plugins import bigquery_agent_analytics_plugin
from google.adk.plugins import plugin_manager as plugin_manager_lib
from google.adk.sessions import base_session_service as base_session_service_lib
from google.adk.sessions import session as session_lib
from google.adk.tools import base_tool as base_tool_lib
from google.adk.tools import tool_context as tool_context_lib
import google.auth
from google.auth import exceptions as auth_exceptions
import google.auth.credentials
from google.cloud import bigquery
from google.cloud.bigquery_storage_v1 import types as bq_storage_types
from google.genai import types
import pyarrow as pa
import pytest

BigQueryLoggerConfig = bigquery_agent_analytics_plugin.BigQueryLoggerConfig

PROJECT_ID = "test-gcp-project"
DATASET_ID = "adk_logs"
TABLE_ID = "agent_events"
DEFAULT_STREAM_NAME = (
    f"projects/{PROJECT_ID}/datasets/{DATASET_ID}/tables/{TABLE_ID}/_default"
)

# --- Pytest Fixtures ---


@pytest.fixture
def mock_session():
  mock_s = mock.create_autospec(
      session_lib.Session, instance=True, spec_set=True
  )
  type(mock_s).id = mock.PropertyMock(return_value="session-123")
  type(mock_s).user_id = mock.PropertyMock(return_value="user-456")
  type(mock_s).app_name = mock.PropertyMock(return_value="test_app")
  type(mock_s).state = mock.PropertyMock(return_value={})
  return mock_s


@pytest.fixture
def mock_agent():
  mock_a = mock.create_autospec(
      base_agent.BaseAgent, instance=True, spec_set=True
  )
  # Mock the 'name' property
  type(mock_a).name = mock.PropertyMock(return_value="MyTestAgent")
  return mock_a


@pytest.fixture
def invocation_context(mock_agent, mock_session):
  mock_session_service = mock.create_autospec(
      base_session_service_lib.BaseSessionService, instance=True, spec_set=True
  )
  mock_plugin_manager = mock.create_autospec(
      plugin_manager_lib.PluginManager, instance=True, spec_set=True
  )
  return invocation_context_lib.InvocationContext(
      agent=mock_agent,
      session=mock_session,
      invocation_id="inv-789",
      session_service=mock_session_service,
      plugin_manager=mock_plugin_manager,
  )


@pytest.fixture
def callback_context(invocation_context):
  return callback_context_lib.CallbackContext(
      invocation_context=invocation_context
  )


@pytest.fixture
def tool_context(invocation_context):
  return tool_context_lib.ToolContext(invocation_context=invocation_context)


@pytest.fixture
def mock_auth_default():
  mock_creds = mock.create_autospec(
      google.auth.credentials.Credentials, instance=True, spec_set=True
  )
  with mock.patch.object(
      google.auth,
      "default",
      autospec=True,
      return_value=(mock_creds, PROJECT_ID),
  ) as mock_auth:
    yield mock_auth


@pytest.fixture
def mock_bq_client():
  with mock.patch.object(bigquery, "Client", autospec=True) as mock_cls:
    yield mock_cls.return_value


@pytest.fixture
def mock_write_client():
  with mock.patch.object(
      bigquery_agent_analytics_plugin, "BigQueryWriteAsyncClient", autospec=True
  ) as mock_cls:
    mock_client = mock_cls.return_value
    mock_client.transport = mock.AsyncMock()

    async def fake_append_rows(requests, **kwargs):
      # This function is now async, so `await client.append_rows` works.
      mock_append_rows_response = mock.MagicMock()
      mock_append_rows_response.row_errors = []
      mock_append_rows_response.error = mock.MagicMock()
      mock_append_rows_response.error.code = 0  # OK status
      # This a gen is what's returned *after* the await.
      return _async_gen(mock_append_rows_response)

    mock_client.append_rows.side_effect = fake_append_rows
    yield mock_client


@pytest.fixture
def dummy_arrow_schema():
  return pa.schema([
      pa.field("timestamp", pa.timestamp("us", tz="UTC")),
      pa.field("event_type", pa.string()),
      pa.field("agent", pa.string()),
      pa.field("session_id", pa.string()),
      pa.field("invocation_id", pa.string()),
      pa.field("user_id", pa.string()),
      pa.field("content", pa.string()),
      pa.field("error_message", pa.string()),
  ])


@pytest.fixture
def mock_to_arrow_schema(dummy_arrow_schema):
  with mock.patch.object(
      bigquery_agent_analytics_plugin,
      "to_arrow_schema",
      autospec=True,
      return_value=dummy_arrow_schema,
  ) as mock_func:
    yield mock_func


@pytest.fixture
def mock_asyncio_to_thread():
  async def fake_to_thread(func, *args, **kwargs):
    return func(*args, **kwargs)

  with mock.patch(
      "asyncio.to_thread", side_effect=fake_to_thread
  ) as mock_async:
    yield mock_async


@pytest.fixture
async def bq_plugin_inst(
    mock_auth_default,
    mock_bq_client,
    mock_write_client,
    mock_to_arrow_schema,
    mock_asyncio_to_thread,
):
  plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
      project_id=PROJECT_ID,
      dataset_id=DATASET_ID,
      table_id=TABLE_ID,
  )
  await plugin._ensure_init()  # Ensure clients are initialized
  mock_write_client.append_rows.reset_mock()
  return plugin


# --- Helper Functions ---


async def _async_gen(val):
  yield val


def _get_captured_event_dict(mock_write_client, expected_schema):
  """Helper to get the event_dict passed to append_rows."""
  mock_write_client.append_rows.assert_called_once()
  call_args = mock_write_client.append_rows.call_args
  requests_iter = call_args.args[0]
  requests = list(requests_iter)
  assert len(requests) == 1
  request = requests[0]
  assert request.write_stream == DEFAULT_STREAM_NAME

  arrow_rows = request.arrow_rows
  message = pa.ipc.read_message(arrow_rows.rows.serialized_record_batch)
  batch = pa.ipc.read_record_batch(message, schema=expected_schema)
  table = pa.Table.from_batches([batch])
  assert table.schema.equals(
      expected_schema
  ), f"Schema mismatch: Expected {expected_schema}, got {table.schema}"
  pydict = table.to_pydict()
  return {k: v[0] for k, v in pydict.items()}


def _assert_common_fields(log_entry, event_type, agent="MyTestAgent"):
  assert log_entry["event_type"] == event_type
  assert log_entry["agent"] == agent
  assert log_entry["session_id"] == "session-123"
  assert log_entry["invocation_id"] == "inv-789"
  assert log_entry["user_id"] == "user-456"
  assert "timestamp" in log_entry
  assert isinstance(log_entry["timestamp"], datetime.datetime)


# --- Test Class ---


class TestBigQueryAgentAnalyticsPlugin:

  @pytest.mark.asyncio
  async def test_plugin_disabled(
      self,
      mock_auth_default,
      mock_bq_client,
      mock_write_client,
      invocation_context,
  ):
    config = BigQueryLoggerConfig(enabled=False)
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
        config=config,
    )
    # user_message = types.Content(parts=[types.Part(text="Test")])

    await plugin.on_user_message_callback(
        invocation_context=invocation_context,
        user_message=types.Content(parts=[types.Part(text="Test")]),
    )
    mock_auth_default.assert_not_called()
    mock_bq_client.assert_not_called()
    mock_write_client.append_rows.assert_not_called()

  @pytest.mark.asyncio
  async def test_event_allowlist(
      self,
      mock_write_client,
      callback_context,
      invocation_context,
      mock_auth_default,
      mock_bq_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    config = BigQueryLoggerConfig(event_allowlist=["LLM_REQUEST"])
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, TABLE_ID, config
    )
    await plugin._ensure_init()
    mock_write_client.append_rows.reset_mock()

    llm_request = llm_request_lib.LlmRequest(
        model="gemini-pro",
        contents=[types.Content(parts=[types.Part(text="Prompt")])],
    )
    await plugin.before_model_callback(
        callback_context=callback_context, llm_request=llm_request
    )
    await asyncio.sleep(0.01)  # Allow background task to run
    mock_write_client.append_rows.assert_called_once()
    mock_write_client.append_rows.reset_mock()

    user_message = types.Content(parts=[types.Part(text="What is up?")])
    await plugin.on_user_message_callback(
        invocation_context=invocation_context, user_message=user_message
    )
    await asyncio.sleep(0.01)  # Allow background task to run
    mock_write_client.append_rows.assert_not_called()

  @pytest.mark.asyncio
  async def test_event_denylist(
      self,
      mock_write_client,
      invocation_context,
      mock_auth_default,
      mock_bq_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    config = BigQueryLoggerConfig(event_denylist=["USER_MESSAGE_RECEIVED"])
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, TABLE_ID, config
    )
    await plugin._ensure_init()
    mock_write_client.append_rows.reset_mock()

    user_message = types.Content(parts=[types.Part(text="What is up?")])
    await plugin.on_user_message_callback(
        invocation_context=invocation_context, user_message=user_message
    )
    await asyncio.sleep(0.01)
    mock_write_client.append_rows.assert_not_called()

    await plugin.before_run_callback(invocation_context=invocation_context)
    await asyncio.sleep(0.01)
    mock_write_client.append_rows.assert_called_once()

  @pytest.mark.asyncio
  async def test_content_formatter(
      self,
      mock_write_client,
      invocation_context,
      mock_auth_default,
      mock_bq_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    def redact_content(content):
      return "[REDACTED]"

    config = BigQueryLoggerConfig(content_formatter=redact_content)
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, TABLE_ID, config
    )
    await plugin._ensure_init()
    mock_write_client.append_rows.reset_mock()

    user_message = types.Content(parts=[types.Part(text="Secret message")])
    await plugin.on_user_message_callback(
        invocation_context=invocation_context, user_message=user_message
    )
    await asyncio.sleep(0.01)
    mock_write_client.append_rows.assert_called_once()
    log_entry = _get_captured_event_dict(mock_write_client, dummy_arrow_schema)
    assert log_entry["content"] == "User Content: [REDACTED]"

  @pytest.mark.asyncio
  async def test_content_formatter_error(
      self,
      mock_write_client,
      invocation_context,
      mock_auth_default,
      mock_bq_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    def error_formatter(content):
      raise ValueError("Formatter failed")

    config = BigQueryLoggerConfig(content_formatter=error_formatter)
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, TABLE_ID, config
    )
    await plugin._ensure_init()
    mock_write_client.append_rows.reset_mock()

    user_message = types.Content(parts=[types.Part(text="Secret message")])
    await plugin.on_user_message_callback(
        invocation_context=invocation_context, user_message=user_message
    )
    await asyncio.sleep(0.01)
    mock_write_client.append_rows.assert_called_once()
    log_entry = _get_captured_event_dict(mock_write_client, dummy_arrow_schema)
    assert log_entry["content"] == "User Content: [FORMATTING FAILED]"

  @pytest.mark.asyncio
  async def test_on_user_message_callback_logs_correctly(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      dummy_arrow_schema,
  ):
    user_message = types.Content(parts=[types.Part(text="What is up?")])
    await bq_plugin_inst.on_user_message_callback(
        invocation_context=invocation_context, user_message=user_message
    )
    await asyncio.sleep(0.01)
    log_entry = _get_captured_event_dict(mock_write_client, dummy_arrow_schema)
    _assert_common_fields(log_entry, "USER_MESSAGE_RECEIVED")
    assert log_entry["content"] == "User Content: text: 'What is up?'"

  @pytest.mark.asyncio
  async def test_on_event_callback_tool_call(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      dummy_arrow_schema,
  ):
    tool_fc = types.FunctionCall(name="get_weather", args={"location": "Paris"})
    event = event_lib.Event(
        author="MyTestAgent",
        content=types.Content(parts=[types.Part(function_call=tool_fc)]),
        timestamp=datetime.datetime(
            2025, 10, 22, 10, 0, 0, tzinfo=datetime.timezone.utc
        ).timestamp(),
    )
    await bq_plugin_inst.on_event_callback(
        invocation_context=invocation_context, event=event
    )
    await asyncio.sleep(0.01)
    log_entry = _get_captured_event_dict(mock_write_client, dummy_arrow_schema)
    _assert_common_fields(log_entry, "TOOL_CALL", agent="MyTestAgent")
    assert '"name": "get_weather"' in log_entry["content"]
    assert log_entry["timestamp"] == datetime.datetime(
        2025, 10, 22, 10, 0, 0, tzinfo=datetime.timezone.utc
    )

  @pytest.mark.asyncio
  async def test_on_event_callback_model_response(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      dummy_arrow_schema,
  ):
    event = event_lib.Event(
        author="MyTestAgent",
        content=types.Content(parts=[types.Part(text="Hello there!")]),
        timestamp=datetime.datetime(
            2025, 10, 22, 11, 0, 0, tzinfo=datetime.timezone.utc
        ).timestamp(),
    )
    await bq_plugin_inst.on_event_callback(
        invocation_context=invocation_context, event=event
    )
    await asyncio.sleep(0.01)
    log_entry = _get_captured_event_dict(mock_write_client, dummy_arrow_schema)
    _assert_common_fields(log_entry, "MODEL_RESPONSE", agent="MyTestAgent")
    assert '"text": "Hello there!"' in log_entry["content"]
    assert log_entry["timestamp"] == datetime.datetime(
        2025, 10, 22, 11, 0, 0, tzinfo=datetime.timezone.utc
    )

  @pytest.mark.asyncio
  async def test_bigquery_client_initialization_failure(
      self,
      mock_auth_default,
      mock_write_client,
      invocation_context,
      mock_asyncio_to_thread,
  ):
    mock_auth_default.side_effect = auth_exceptions.GoogleAuthError(
        "Auth failed"
    )
    plugin_with_fail = (
        bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID,
            table_id=TABLE_ID,
        )
    )
    with mock.patch.object(logging, "error") as mock_log_error:
      await plugin_with_fail.on_user_message_callback(
          invocation_context=invocation_context,
          user_message=types.Content(parts=[types.Part(text="Test")]),
      )
      await asyncio.sleep(0.01)
      mock_log_error.assert_any_call("BQ Init Failed: Auth failed")
    mock_write_client.append_rows.assert_not_called()

  @pytest.mark.asyncio
  async def test_bigquery_insert_error_does_not_raise(
      self, bq_plugin_inst, mock_write_client, invocation_context
  ):

    async def fake_append_rows_with_error(requests, **kwargs):
      mock_append_rows_response = mock.MagicMock()
      mock_append_rows_response.row_errors = []  # No row errors
      mock_append_rows_response.error = mock.MagicMock()
      mock_append_rows_response.error.code = 3  # INVALID_ARGUMENT
      mock_append_rows_response.error.message = "Test BQ Error"
      return _async_gen(mock_append_rows_response)

    mock_write_client.append_rows.side_effect = fake_append_rows_with_error

    with mock.patch.object(logging, "error") as mock_log_error:
      await bq_plugin_inst.on_user_message_callback(
          invocation_context=invocation_context,
          user_message=types.Content(parts=[types.Part(text="Test")]),
      )
      await asyncio.sleep(0.01)
      mock_log_error.assert_called_with("BQ Write Error: Test BQ Error")
    mock_write_client.append_rows.assert_called_once()

  @pytest.mark.asyncio
  async def test_shutdown(
      self, bq_plugin_inst, mock_bq_client, mock_write_client
  ):
    await bq_plugin_inst.shutdown()
    mock_write_client.transport.close.assert_called_once()
    mock_bq_client.close.assert_called_once()

  # ... other tests remain the same ...
  @pytest.mark.asyncio
  async def test_before_run_callback_logs_correctly(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      dummy_arrow_schema,
  ):
    await bq_plugin_inst.before_run_callback(
        invocation_context=invocation_context
    )
    await asyncio.sleep(0.01)
    log_entry = _get_captured_event_dict(mock_write_client, dummy_arrow_schema)
    _assert_common_fields(log_entry, "INVOCATION_STARTING")
    assert log_entry["content"] is None

  @pytest.mark.asyncio
  async def test_after_run_callback_logs_correctly(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      dummy_arrow_schema,
  ):
    await bq_plugin_inst.after_run_callback(
        invocation_context=invocation_context
    )
    await asyncio.sleep(0.01)
    log_entry = _get_captured_event_dict(mock_write_client, dummy_arrow_schema)
    _assert_common_fields(log_entry, "INVOCATION_COMPLETED")
    assert log_entry["content"] is None

  @pytest.mark.asyncio
  async def test_before_agent_callback_logs_correctly(
      self,
      bq_plugin_inst,
      mock_write_client,
      mock_agent,
      callback_context,
      dummy_arrow_schema,
  ):
    await bq_plugin_inst.before_agent_callback(
        agent=mock_agent, callback_context=callback_context
    )
    await asyncio.sleep(0.01)
    log_entry = _get_captured_event_dict(mock_write_client, dummy_arrow_schema)
    _assert_common_fields(log_entry, "AGENT_STARTING")
    assert log_entry["content"] == "Agent Name: MyTestAgent"

  @pytest.mark.asyncio
  async def test_after_agent_callback_logs_correctly(
      self,
      bq_plugin_inst,
      mock_write_client,
      mock_agent,
      callback_context,
      dummy_arrow_schema,
  ):
    await bq_plugin_inst.after_agent_callback(
        agent=mock_agent, callback_context=callback_context
    )
    await asyncio.sleep(0.01)
    log_entry = _get_captured_event_dict(mock_write_client, dummy_arrow_schema)
    _assert_common_fields(log_entry, "AGENT_COMPLETED")
    assert log_entry["content"] == "Agent Name: MyTestAgent"

  @pytest.mark.asyncio
  async def test_before_model_callback_logs_correctly(
      self,
      bq_plugin_inst,
      mock_write_client,
      callback_context,
      dummy_arrow_schema,
  ):
    llm_request = llm_request_lib.LlmRequest(
        model="gemini-pro",
        contents=[types.Content(parts=[types.Part(text="Prompt")])],
    )
    await bq_plugin_inst.before_model_callback(
        callback_context=callback_context, llm_request=llm_request
    )
    await asyncio.sleep(0.01)
    log_entry = _get_captured_event_dict(mock_write_client, dummy_arrow_schema)
    _assert_common_fields(log_entry, "LLM_REQUEST")
    assert log_entry["content"] == "Model: gemini-pro | System Prompt: Empty"

  @pytest.mark.asyncio
  async def test_after_model_callback_text_response(
      self,
      bq_plugin_inst,
      mock_write_client,
      callback_context,
      dummy_arrow_schema,
  ):
    llm_response = llm_response_lib.LlmResponse(
        content=types.Content(parts=[types.Part(text="Model response")]),
        usage_metadata=types.UsageMetadata(
            prompt_token_count=10, total_token_count=15
        ),
    )
    await bq_plugin_inst.after_model_callback(
        callback_context=callback_context, llm_response=llm_response
    )
    await asyncio.sleep(0.01)
    log_entry = _get_captured_event_dict(mock_write_client, dummy_arrow_schema)
    _assert_common_fields(log_entry, "LLM_RESPONSE")
    assert (
        "Tool Name: text_response, text: 'Model response'"
        in log_entry["content"]
    )
    assert "Token Usage:" in log_entry["content"]
    assert "prompt: 10" in log_entry["content"]
    assert "total: 15" in log_entry["content"]
    assert log_entry["error_message"] is None

  @pytest.mark.asyncio
  async def test_after_model_callback_tool_call(
      self,
      bq_plugin_inst,
      mock_write_client,
      callback_context,
      dummy_arrow_schema,
  ):
    tool_fc = types.FunctionCall(name="get_weather", args={"location": "Paris"})
    llm_response = llm_response_lib.LlmResponse(
        content=types.Content(parts=[types.Part(function_call=tool_fc)]),
        usage_metadata=types.UsageMetadata(
            prompt_token_count=10, total_token_count=15
        ),
    )
    await bq_plugin_inst.after_model_callback(
        callback_context=callback_context, llm_response=llm_response
    )
    await asyncio.sleep(0.01)
    log_entry = _get_captured_event_dict(mock_write_client, dummy_arrow_schema)
    _assert_common_fields(log_entry, "LLM_RESPONSE")
    assert "Tool Name: get_weather" in log_entry["content"]
    assert "Token Usage:" in log_entry["content"]
    assert "prompt: 10" in log_entry["content"]
    assert "total: 15" in log_entry["content"]
    assert log_entry["error_message"] is None

  @pytest.mark.asyncio
  async def test_before_tool_callback_logs_correctly(
      self, bq_plugin_inst, mock_write_client, tool_context, dummy_arrow_schema
  ):
    mock_tool = mock.create_autospec(
        base_tool_lib.BaseTool, instance=True, spec_set=True
    )
    type(mock_tool).name = mock.PropertyMock(return_value="MyTool")
    type(mock_tool).description = mock.PropertyMock(return_value="Description")
    await bq_plugin_inst.before_tool_callback(
        tool=mock_tool, tool_args={"param": "value"}, tool_context=tool_context
    )
    await asyncio.sleep(0.01)
    log_entry = _get_captured_event_dict(mock_write_client, dummy_arrow_schema)
    _assert_common_fields(log_entry, "TOOL_STARTING")
    assert (
        log_entry["content"]
        == 'Tool Name: MyTool, Description: Description, Arguments: {"param":'
        ' "value"}'
    )

  @pytest.mark.asyncio
  async def test_after_tool_callback_logs_correctly(
      self, bq_plugin_inst, mock_write_client, tool_context, dummy_arrow_schema
  ):
    mock_tool = mock.create_autospec(
        base_tool_lib.BaseTool, instance=True, spec_set=True
    )
    type(mock_tool).name = mock.PropertyMock(return_value="MyTool")
    type(mock_tool).description = mock.PropertyMock(return_value="Description")
    await bq_plugin_inst.after_tool_callback(
        tool=mock_tool,
        tool_args={},
        tool_context=tool_context,
        result={"status": "success"},
    )
    await asyncio.sleep(0.01)
    log_entry = _get_captured_event_dict(mock_write_client, dummy_arrow_schema)
    _assert_common_fields(log_entry, "TOOL_COMPLETED")
    assert (
        log_entry["content"]
        == 'Tool Name: MyTool, Result: {"status": "success"}'
    )

  @pytest.mark.asyncio
  async def test_on_model_error_callback_logs_correctly(
      self,
      bq_plugin_inst,
      mock_write_client,
      callback_context,
      dummy_arrow_schema,
  ):
    llm_request = llm_request_lib.LlmRequest(
        model="gemini-pro",
        contents=[types.Content(parts=[types.Part(text="Prompt")])],
    )
    error = ValueError("LLM failed")
    await bq_plugin_inst.on_model_error_callback(
        callback_context=callback_context, llm_request=llm_request, error=error
    )
    await asyncio.sleep(0.01)
    log_entry = _get_captured_event_dict(mock_write_client, dummy_arrow_schema)
    _assert_common_fields(log_entry, "LLM_ERROR")
    assert log_entry["content"] is None
    assert log_entry["error_message"] == "LLM failed"

  @pytest.mark.asyncio
  async def test_on_tool_error_callback_logs_correctly(
      self, bq_plugin_inst, mock_write_client, tool_context, dummy_arrow_schema
  ):
    mock_tool = mock.create_autospec(
        base_tool_lib.BaseTool, instance=True, spec_set=True
    )
    type(mock_tool).name = mock.PropertyMock(return_value="MyTool")
    type(mock_tool).description = mock.PropertyMock(return_value="Description")
    error = TimeoutError("Tool timed out")
    await bq_plugin_inst.on_tool_error_callback(
        tool=mock_tool,
        tool_args={"param": "value"},
        tool_context=tool_context,
        error=error,
    )
    await asyncio.sleep(0.01)
    log_entry = _get_captured_event_dict(mock_write_client, dummy_arrow_schema)
    _assert_common_fields(log_entry, "TOOL_ERROR")
    assert (
        log_entry["content"]
        == 'Tool Name: MyTool, Arguments: {"param": "value"}'
    )
    assert log_entry["error_message"] == "Tool timed out"
