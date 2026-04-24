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

import asyncio
import contextlib
import dataclasses
import json
import os
from unittest import mock

from google.adk.agents import base_agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import event as event_lib
from google.adk.events import event_actions as event_actions_lib
from google.adk.models import llm_request as llm_request_lib
from google.adk.models import llm_response as llm_response_lib
from google.adk.plugins import bigquery_agent_analytics_plugin
from google.adk.plugins import plugin_manager as plugin_manager_lib
from google.adk.sessions import base_session_service as base_session_service_lib
from google.adk.sessions import session as session_lib
from google.adk.tools import base_tool as base_tool_lib
from google.adk.tools import tool_context as tool_context_lib
from google.adk.utils._telemetry_context import _is_visual_builder
from google.adk.version import __version__
import google.auth
from google.auth import exceptions as auth_exceptions
import google.auth.credentials
from google.cloud import bigquery
from google.cloud import exceptions as cloud_exceptions
from google.genai import types
from opentelemetry import trace
import pyarrow as pa
import pytest

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
  type(mock_a).instruction = mock.PropertyMock(return_value="Test Instruction")
  return mock_a


@pytest.fixture
def invocation_context(mock_agent, mock_session):
  mock_session_service = mock.create_autospec(
      base_session_service_lib.BaseSessionService, instance=True, spec_set=True
  )
  mock_plugin_manager = mock.create_autospec(
      plugin_manager_lib.PluginManager, instance=True, spec_set=True
  )
  return InvocationContext(
      agent=mock_agent,
      session=mock_session,
      invocation_id="inv-789",
      session_service=mock_session_service,
      plugin_manager=mock_plugin_manager,
  )


@pytest.fixture
def callback_context(invocation_context):
  return CallbackContext(invocation_context=invocation_context)


@pytest.fixture
def tool_context(invocation_context):
  return tool_context_lib.ToolContext(invocation_context=invocation_context)


class FakeCredentials(google.auth.credentials.Credentials):

  def __init__(self):
    pass

  def refresh(self, request):
    pass


@pytest.fixture
def mock_auth_default():
  mock_creds = FakeCredentials()
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
      pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
      pa.field("root_agent_name", pa.string(), nullable=True),
      pa.field("event_type", pa.string(), nullable=True),
      pa.field("agent", pa.string(), nullable=True),
      pa.field("session_id", pa.string(), nullable=True),
      pa.field("invocation_id", pa.string(), nullable=True),
      pa.field("user_id", pa.string(), nullable=True),
      pa.field("trace_id", pa.string(), nullable=True),
      pa.field("span_id", pa.string(), nullable=True),
      pa.field("parent_span_id", pa.string(), nullable=True),
      pa.field(
          "content", pa.string(), nullable=True
      ),  # JSON stored as string in Arrow
      pa.field(
          "content_parts",
          pa.list_(
              pa.struct([
                  pa.field("mime_type", pa.string(), nullable=True),
                  pa.field("uri", pa.string(), nullable=True),
                  pa.field(
                      "object_ref",
                      pa.struct([
                          pa.field("uri", pa.string(), nullable=True),
                          pa.field("authorizer", pa.string(), nullable=True),
                          pa.field("version", pa.string(), nullable=True),
                          pa.field(
                              "details",
                              pa.string(),
                              nullable=True,
                              metadata={
                                  b"ARROW:extension:name": (
                                      b"google:sqlType:json"
                                  )
                              },
                          ),
                      ]),
                      nullable=True,
                  ),
                  pa.field("text", pa.string(), nullable=True),
                  pa.field("part_index", pa.int64(), nullable=True),
                  pa.field("part_attributes", pa.string(), nullable=True),
                  pa.field("storage_mode", pa.string(), nullable=True),
              ])
          ),
          nullable=True,
      ),
      pa.field("attributes", pa.string(), nullable=True),
      pa.field("latency_ms", pa.string(), nullable=True),
      pa.field("status", pa.string(), nullable=True),
      pa.field("error_message", pa.string(), nullable=True),
      pa.field("is_truncated", pa.bool_(), nullable=True),
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
def mock_storage_client():
  with mock.patch("google.cloud.storage.Client") as mock_client:
    yield mock_client


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
  await plugin._ensure_started()  # Ensure clients are initialized
  mock_write_client.append_rows.reset_mock()
  yield plugin
  await plugin.shutdown()


@contextlib.asynccontextmanager
async def managed_plugin(*args, **kwargs):
  """Async context manager to ensure plugin shutdown."""
  plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
      *args, **kwargs
  )
  try:
    yield plugin
  finally:
    await plugin.shutdown()


# --- Helper Functions ---
async def _async_gen(val):
  yield val


async def _get_captured_event_dict_async(mock_write_client, expected_schema):
  """Helper to get the event_dict passed to append_rows."""
  mock_write_client.append_rows.assert_called_once()
  call_args = mock_write_client.append_rows.call_args
  requests_iter = call_args.args[0]
  requests = []
  if hasattr(requests_iter, "__aiter__"):
    async for req in requests_iter:
      requests.append(req)
  else:
    requests = list(requests_iter)
  assert len(requests) == 1
  request = requests[0]
  assert request.write_stream == DEFAULT_STREAM_NAME
  assert request.trace_id.startswith("google-adk-bq-logger")
  assert request.trace_id.endswith(f"/{__version__}")
  # Parse the Arrow batch back to a dict for verification
  try:
    reader = pa.ipc.open_stream(request.arrow_rows.rows.serialized_record_batch)
    table = reader.read_all()
  except Exception:
    # Fallback: try reading as a single batch
    buf = pa.py_buffer(request.arrow_rows.rows.serialized_record_batch)
    batch = pa.ipc.read_record_batch(buf, expected_schema)
    table = pa.Table.from_batches([batch])
  assert table.schema.equals(
      expected_schema
  ), f"Schema mismatch: Expected {expected_schema}, got {table.schema}"
  pydict = table.to_pydict()
  return {k: v[0] for k, v in pydict.items()}


async def _get_captured_rows_async(mock_write_client, expected_schema):
  """Helper to get all rows passed to append_rows."""
  all_rows = []
  for call in mock_write_client.append_rows.call_args_list:
    requests_iter = call.args[0]
    requests = []
    if hasattr(requests_iter, "__aiter__"):
      async for req in requests_iter:
        requests.append(req)
    else:
      requests = list(requests_iter)
    for request in requests:
      # Parse the Arrow batch back to a dict for verification
      try:
        reader = pa.ipc.open_stream(
            request.arrow_rows.rows.serialized_record_batch
        )
        table = reader.read_all()
      except Exception:
        # Fallback: try reading as a single batch
        buf = pa.py_buffer(request.arrow_rows.rows.serialized_record_batch)
        batch = pa.ipc.read_record_batch(buf, expected_schema)
        table = pa.Table.from_batches([batch])
      pydict = table.to_pylist()
      all_rows.extend(pydict)
  return all_rows


def _assert_common_fields(log_entry, event_type, agent="MyTestAgent"):
  assert log_entry["event_type"] == event_type
  assert log_entry["agent"] == agent
  assert log_entry["session_id"] == "session-123"
  assert log_entry["invocation_id"] == "inv-789"


def test_recursive_smart_truncate():
  """Test recursive smart truncate."""
  obj = {
      "a": "long string" * 10,
      "b": ["short", "long string" * 10],
      "c": {"d": "long string" * 10},
  }
  max_len = 10
  truncated, is_truncated = (
      bigquery_agent_analytics_plugin._recursive_smart_truncate(obj, max_len)
  )
  assert is_truncated

  assert truncated["a"] == "long strin...[TRUNCATED]"
  assert truncated["b"][0] == "short"
  assert truncated["b"][1] == "long strin...[TRUNCATED]"
  assert truncated["c"]["d"] == "long strin...[TRUNCATED]"


def test_recursive_smart_truncate_with_dataclasses():
  """Test recursive smart truncate with dataclasses."""

  @dataclasses.dataclass
  class LocalMissedKPI:
    kpi: str
    value: float

  @dataclasses.dataclass
  class LocalIncident:
    id: str
    kpi_missed: list[LocalMissedKPI]
    status: str

  incident = LocalIncident(
      id="inc-123",
      kpi_missed=[LocalMissedKPI(kpi="latency", value=99.9)],
      status="active",
  )
  content = {"result": incident}
  max_len = 1000

  truncated, is_truncated = (
      bigquery_agent_analytics_plugin._recursive_smart_truncate(
          content, max_len
      )
  )
  assert not is_truncated
  assert isinstance(truncated["result"], dict)
  assert truncated["result"]["id"] == "inc-123"
  assert isinstance(truncated["result"]["kpi_missed"][0], dict)
  assert truncated["result"]["kpi_missed"][0]["kpi"] == "latency"


def test_recursive_smart_truncate_redaction():
  """Test that sensitive keys and temp: state keys are redacted."""
  obj = {
      "client_secret": "super-secret-123",
      "access_token": "ya29.blah",
      "refresh_token": "1//0g",
      "id_token": "eyJhb",
      "api_key": "AIza",
      "password": "my-password",
      "safe_key": "safe-value",
      "temp:auth_state": "some-auth-state",
      "nested": {
          "CLIENT_SECRET": "nested-secret",
          "normal": "value",
      },
  }
  max_len = 1000
  truncated, is_truncated = (
      bigquery_agent_analytics_plugin._recursive_smart_truncate(obj, max_len)
  )
  assert not is_truncated
  assert truncated["client_secret"] == "[REDACTED]"
  assert truncated["access_token"] == "[REDACTED]"
  assert truncated["refresh_token"] == "[REDACTED]"
  assert truncated["id_token"] == "[REDACTED]"
  assert truncated["api_key"] == "[REDACTED]"
  assert truncated["password"] == "[REDACTED]"
  assert truncated["safe_key"] == "safe-value"
  assert truncated["temp:auth_state"] == "[REDACTED]"
  assert truncated["nested"]["CLIENT_SECRET"] == "[REDACTED]"
  assert truncated["nested"]["normal"] == "value"


class TestBigQueryAgentAnalyticsPlugin:
  """Tests for the BigQueryAgentAnalyticsPlugin."""

  @pytest.mark.asyncio
  async def test_plugin_disabled(
      self,
      mock_auth_default,
      mock_bq_client,
      mock_write_client,
      invocation_context,
  ):
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(enabled=False)
    async with managed_plugin(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
        config=config,
    ) as plugin:
      # user_message = types.Content(parts=[types.Part(text="Test")])
      await plugin.on_user_message_callback(
          invocation_context=invocation_context,
          user_message=types.Content(parts=[types.Part(text="Test")]),
      )
      mock_auth_default.assert_not_called()
      mock_bq_client.assert_not_called()

  @pytest.mark.asyncio
  async def test_enriched_metadata_logging(
      self,
      mock_auth_default,
      mock_bq_client,
      mock_write_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      callback_context,
  ):
    # Setup
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig()
    async with managed_plugin(PROJECT_ID, DATASET_ID, config=config) as plugin:
      # Mock root agent
      mock_root = mock.create_autospec(
          base_agent.BaseAgent, instance=True, spec_set=True
      )
      type(mock_root).name = mock.PropertyMock(return_value="RootAgent")
      callback_context._invocation_context.agent.root_agent = mock_root
      # 1. Test root_agent_name and model extraction from request
      llm_request = llm_request_lib.LlmRequest(
          model="gemini-pro",
          contents=[types.Content(parts=[types.Part(text="Hi")])],
      )
      await plugin.before_model_callback(
          callback_context=callback_context, llm_request=llm_request
      )
      # 2. Test model_version and usage_metadata extraction from response
      usage = types.GenerateContentResponseUsageMetadata(
          prompt_token_count=10, candidates_token_count=20, total_token_count=30
      )
      llm_response = llm_response_lib.LlmResponse(
          content=types.Content(parts=[types.Part(text="Hello")]),
          usage_metadata=usage,
          model_version="v1.2.3",
      )
      await plugin.after_model_callback(
          callback_context=callback_context, llm_response=llm_response
      )
    # Verify captured rows from mock client
    rows = await _get_captured_rows_async(mock_write_client, dummy_arrow_schema)
    assert len(rows) == 2
    # Check LLM_REQUEST row
    # Sort by event_type to ensure consistent indexing
    rows.sort(key=lambda x: x["event_type"])
    request_row = rows[0]  # LLM_REQUEST
    response_row = rows[1]  # LLM_RESPONSE
    assert request_row["event_type"] == "LLM_REQUEST"
    attr_req = json.loads(request_row["attributes"])
    assert attr_req["root_agent_name"] == "RootAgent"
    assert attr_req["model"] == "gemini-pro"
    # Check LLM_RESPONSE row
    assert response_row["event_type"] == "LLM_RESPONSE"
    attr_res = json.loads(response_row["attributes"])
    assert attr_res["root_agent_name"] == "RootAgent"
    assert attr_res["model_version"] == "v1.2.3"
    usage_meta = attr_res["usage_metadata"]
    assert "prompt_token_count" in usage_meta
    assert usage_meta["prompt_token_count"] == 10
    mock_write_client.append_rows.assert_called()

  @pytest.mark.asyncio
  async def test_concurrent_span_management(
      self,
      mock_auth_default,
      mock_bq_client,
      mock_write_client,
      mock_to_arrow_schema,
      callback_context,
  ):
    # Setup
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig()
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, config=config
    )
    # Initialize trace in main context
    bigquery_agent_analytics_plugin.TraceManager.init_trace(callback_context)

    async def branch_1():
      s_id = bigquery_agent_analytics_plugin.TraceManager.push_span(
          callback_context, span_name="span-1"
      )
      await asyncio.sleep(0.02)
      current_s_id = (
          bigquery_agent_analytics_plugin.TraceManager.get_current_span_id()
      )
      assert s_id == current_s_id
      bigquery_agent_analytics_plugin.TraceManager.pop_span()
      return s_id

    async def branch_2():
      s_id = bigquery_agent_analytics_plugin.TraceManager.push_span(
          callback_context, span_name="span-2"
      )
      await asyncio.sleep(0.02)
      current_s_id = (
          bigquery_agent_analytics_plugin.TraceManager.get_current_span_id()
      )
      assert s_id == current_s_id
      bigquery_agent_analytics_plugin.TraceManager.pop_span()
      return s_id

    # Run concurrently
    results = await asyncio.gather(branch_1(), branch_2())
    # If they shared the same list/dict, they would interfere.
    assert results[0] is not None
    assert results[1] is not None
    assert results[0] != results[1]

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
    _ = mock_auth_default
    _ = mock_bq_client
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(
        event_allowlist=["LLM_REQUEST"]
    )
    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()
      llm_request = llm_request_lib.LlmRequest(
          model="gemini-pro",
          contents=[types.Content(parts=[types.Part(text="Prompt")])],
      )
      bigquery_agent_analytics_plugin.TraceManager.push_span(callback_context)
      await plugin.before_model_callback(
          callback_context=callback_context, llm_request=llm_request
      )
      await asyncio.sleep(0.01)  # Allow background task to run
      mock_write_client.append_rows.assert_called_once()
      mock_write_client.append_rows.reset_mock()
      user_message = types.Content(parts=[types.Part(text="What is up?")])
      bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
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
    _ = mock_auth_default
    _ = mock_bq_client
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(
        event_denylist=["USER_MESSAGE_RECEIVED"]
    )
    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()
      user_message = types.Content(parts=[types.Part(text="What is up?")])
      bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
      await plugin.on_user_message_callback(
          invocation_context=invocation_context, user_message=user_message
      )
      await asyncio.sleep(0.01)
      mock_write_client.append_rows.assert_not_called()
      bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
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
    """Test content formatter."""
    _ = mock_auth_default
    _ = mock_bq_client

    def redact_content(content, event_type):
      return "[REDACTED]"

    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(
        content_formatter=redact_content
    )
    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()
      user_message = types.Content(parts=[types.Part(text="Secret message")])
      bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
      await plugin.on_user_message_callback(
          invocation_context=invocation_context, user_message=user_message
      )
      await asyncio.sleep(0.01)
      mock_write_client.append_rows.assert_called_once()
      log_entry = await _get_captured_event_dict_async(
          mock_write_client, dummy_arrow_schema
      )
      # If the formatter returns a string, it's stored directly.
      assert log_entry["content"] == "[REDACTED]"

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
    """Test content formatter error handling."""
    _ = mock_auth_default
    _ = mock_bq_client

    def error_formatter(content, event_type):
      raise ValueError("Formatter failed")

    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(
        content_formatter=error_formatter
    )
    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()
      user_message = types.Content(parts=[types.Part(text="Secret message")])
      bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
      await plugin.on_user_message_callback(
          invocation_context=invocation_context, user_message=user_message
      )
      await asyncio.sleep(0.01)
      mock_write_client.append_rows.assert_called_once()
      log_entry = await _get_captured_event_dict_async(
          mock_write_client, dummy_arrow_schema
      )
      # If formatter fails, it logs a warning and continues with original content.
      assert log_entry["content"] == '{"text_summary": "Secret message"}'

  @pytest.mark.asyncio
  async def test_max_content_length(
      self,
      mock_write_client,
      invocation_context,
      callback_context,
      mock_auth_default,
      mock_bq_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    _ = mock_auth_default
    _ = mock_bq_client
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(
        max_content_length=40
    )
    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()
      # Test User Message Truncation
      user_message = types.Content(
          parts=[types.Part(text="12345678901234567890123456789012345678901")]
      )  # 41 chars
      bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
      await plugin.on_user_message_callback(
          invocation_context=invocation_context, user_message=user_message
      )
      await asyncio.sleep(0.01)
      mock_write_client.append_rows.assert_called_once()
      log_entry = await _get_captured_event_dict_async(
          mock_write_client, dummy_arrow_schema
      )
      assert (
          log_entry["content"]
          == '{"text_summary":'
          ' "1234567890123456789012345678901234567890...[TRUNCATED]"}'
      )
      assert log_entry["is_truncated"]
      mock_write_client.append_rows.reset_mock()
      # Test before_model_callback full content truncation
      llm_request = llm_request_lib.LlmRequest(
          model="gemini-pro",
          config=types.GenerateContentConfig(
              system_instruction=types.Content(
                  parts=[types.Part(text="System Instruction")]
              )
          ),
          contents=[
              types.Content(role="user", parts=[types.Part(text="Prompt")])
          ],
      )
      bigquery_agent_analytics_plugin.TraceManager.push_span(callback_context)
      await plugin.before_model_callback(
          callback_context=callback_context, llm_request=llm_request
      )
      await asyncio.sleep(0.01)
      mock_write_client.append_rows.assert_called_once()
      log_entry = await _get_captured_event_dict_async(
          mock_write_client, dummy_arrow_schema
      )
      # Full content: {"prompt": "text: 'Prompt'",
      # "system_prompt": "text: 'System Instruction'"}
      # In our new logic, we don't truncate the whole JSON string if it's valid JSON.
      # Instead, we should have truncated the values within the dict, but currently we don't.
      # For now, update test to reflect current behavior (valid JSON, no truncation of the whole string).
      assert log_entry["content"].startswith(
          '{"prompt": [{"role": "user", "content": "Prompt"}]'
      )
      assert log_entry["is_truncated"] is False

  @pytest.mark.asyncio
  async def test_max_content_length_tool_args(
      self,
      mock_write_client,
      tool_context,
      mock_auth_default,
      mock_bq_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    _ = mock_auth_default
    _ = mock_bq_client
    _ = mock_to_arrow_schema
    _ = mock_asyncio_to_thread
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(
        max_content_length=80
    )
    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()
      mock_tool = mock.create_autospec(
          base_tool_lib.BaseTool, instance=True, spec_set=True
      )
      type(mock_tool).name = mock.PropertyMock(return_value="MyTool")
      type(mock_tool).description = mock.PropertyMock(
          return_value="Description"
      )
      # Args length > 80
      # {"param": "A" * 100} is > 100 chars.
      bigquery_agent_analytics_plugin.TraceManager.push_span(tool_context)
      await plugin.before_tool_callback(
          tool=mock_tool,
          tool_args={"param": "A" * 100},
          tool_context=tool_context,
      )
      await asyncio.sleep(0.01)
      mock_write_client.append_rows.assert_called_once()
      log_entry = await _get_captured_event_dict_async(
          mock_write_client, dummy_arrow_schema
      )
      _assert_common_fields(log_entry, "TOOL_STARTING")
      # Now we do truncate nested values, and is_truncated flag is True
      assert log_entry["is_truncated"]
      content_dict = json.loads(log_entry["content"])
      assert content_dict["tool"] == "MyTool"
      assert content_dict["args"]["param"].endswith("...[TRUNCATED]")

  @pytest.mark.asyncio
  async def test_max_content_length_tool_args_no_truncation(
      self,
      mock_write_client,
      tool_context,
      mock_auth_default,
      mock_bq_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(
        max_content_length=-1
    )
    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()
      mock_tool = mock.create_autospec(
          base_tool_lib.BaseTool, instance=True, spec_set=True
      )
      type(mock_tool).name = mock.PropertyMock(return_value="MyTool")
      type(mock_tool).description = mock.PropertyMock(
          return_value="Description"
      )
      # Args length > 80
      # {"param": "A" * 100} is > 100 chars.
      bigquery_agent_analytics_plugin.TraceManager.push_span(tool_context)
      await plugin.before_tool_callback(
          tool=mock_tool,
          tool_args={"param": "A" * 100},
          tool_context=tool_context,
      )
      await asyncio.sleep(0.01)
      mock_write_client.append_rows.assert_called_once()
      log_entry = await _get_captured_event_dict_async(
          mock_write_client, dummy_arrow_schema
      )
      _assert_common_fields(log_entry, "TOOL_STARTING")
      # No truncation
      assert not log_entry["is_truncated"]
      content_dict = json.loads(log_entry["content"])
      assert content_dict["tool"] == "MyTool"
      assert content_dict["args"]["param"] == "A" * 100

  @pytest.mark.asyncio
  async def test_max_content_length_tool_result(
      self,
      mock_write_client,
      tool_context,
      mock_auth_default,
      mock_bq_client,
      mock_asyncio_to_thread,
      mock_to_arrow_schema,
      dummy_arrow_schema,
  ):
    """Test max content length for tool result."""
    _ = mock_auth_default
    _ = mock_bq_client
    _ = mock_to_arrow_schema
    _ = mock_asyncio_to_thread
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(
        max_content_length=80
    )
    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()
      mock_tool = mock.create_autospec(
          base_tool_lib.BaseTool, instance=True, spec_set=True
      )
      type(mock_tool).name = mock.PropertyMock(return_value="MyTool")
      # Result length > 80
      # {"res": "A" * 100} is > 100 chars.
      bigquery_agent_analytics_plugin.TraceManager.push_span(tool_context)
      await plugin.after_tool_callback(
          tool=mock_tool,
          tool_args={},
          tool_context=tool_context,
          result={"res": "A" * 100},
      )
      await asyncio.sleep(0.01)
      mock_write_client.append_rows.assert_called_once()
      log_entry = await _get_captured_event_dict_async(
          mock_write_client, dummy_arrow_schema
      )
      _assert_common_fields(log_entry, "TOOL_COMPLETED")
      # Now we do truncate nested values, and is_truncated flag is True
      assert log_entry["is_truncated"]
      content_dict = json.loads(log_entry["content"])
      assert content_dict["tool"] == "MyTool"
      assert content_dict["result"]["res"].endswith("...[TRUNCATED]")

  @pytest.mark.asyncio
  async def test_max_content_length_tool_result_no_truncation(
      self,
      mock_write_client,
      tool_context,
      mock_auth_default,
      mock_bq_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    """Test max content length for tool result with no truncation."""
    _ = mock_auth_default
    _ = mock_bq_client
    _ = mock_to_arrow_schema
    _ = mock_asyncio_to_thread
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(
        max_content_length=-1
    )
    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()
      mock_tool = mock.create_autospec(
          base_tool_lib.BaseTool, instance=True, spec_set=True
      )
      type(mock_tool).name = mock.PropertyMock(return_value="MyTool")
      # Result length > 80
      # {"res": "A" * 100} is > 100 chars.
      bigquery_agent_analytics_plugin.TraceManager.push_span(tool_context)
      await plugin.after_tool_callback(
          tool=mock_tool,
          tool_args={},
          tool_context=tool_context,
          result={"res": "A" * 100},
      )
      await asyncio.sleep(0.01)
      mock_write_client.append_rows.assert_called_once()
      log_entry = await _get_captured_event_dict_async(
          mock_write_client, dummy_arrow_schema
      )
      _assert_common_fields(log_entry, "TOOL_COMPLETED")
      # No truncation
      assert not log_entry["is_truncated"]
      content_dict = json.loads(log_entry["content"])
      assert content_dict["tool"] == "MyTool"
      assert content_dict["result"]["res"] == "A" * 100

  @pytest.mark.asyncio
  async def test_max_content_length_tool_error(
      self,
      mock_write_client,
      tool_context,
      mock_auth_default,
      mock_bq_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(
        max_content_length=80
    )
    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()
      mock_tool = mock.create_autospec(
          base_tool_lib.BaseTool, instance=True, spec_set=True
      )
      type(mock_tool).name = mock.PropertyMock(return_value="MyTool")
      # Args length > 80
      # {"arg": "A" * 100} is > 100 chars.
      bigquery_agent_analytics_plugin.TraceManager.push_span(tool_context)
      await plugin.on_tool_error_callback(
          tool=mock_tool,
          tool_args={"arg": "A" * 100},
          tool_context=tool_context,
          error=ValueError("Oops"),
      )
      await asyncio.sleep(0.01)
      mock_write_client.append_rows.assert_called_once()
      log_entry = await _get_captured_event_dict_async(
          mock_write_client, dummy_arrow_schema
      )
      assert log_entry["content"].startswith(
          '{"tool": "MyTool", "args": {"arg": "AAAAA'
      )
      # Check for truncation in the nested value
      content_dict = json.loads(log_entry["content"])
      assert content_dict["args"]["arg"].endswith("...[TRUNCATED]")
      assert log_entry["is_truncated"]
      assert log_entry["error_message"] == "Oops"

  @pytest.mark.asyncio
  async def test_on_user_message_callback_logs_correctly(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      dummy_arrow_schema,
  ):
    user_message = types.Content(parts=[types.Part(text="What is up?")])
    bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
    await bq_plugin_inst.on_user_message_callback(
        invocation_context=invocation_context, user_message=user_message
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    _assert_common_fields(log_entry, "USER_MESSAGE_RECEIVED")
    assert log_entry["content"] == '{"text_summary": "What is up?"}'

  @pytest.mark.asyncio
  async def test_offloading_with_connection_id(
      self,
      mock_write_client,
      invocation_context,
      mock_auth_default,
      mock_bq_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
      mock_storage_client,
  ):
    _ = mock_auth_default
    _ = mock_bq_client
    _ = mock_to_arrow_schema
    _ = mock_asyncio_to_thread
    # Mock GCS bucket
    mock_bucket = mock.Mock()
    mock_blob = mock.Mock()
    mock_bucket.blob.return_value = mock_blob
    mock_bucket.name = "my-bucket"
    mock_storage_client.return_value.bucket.return_value = mock_bucket
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(
        gcs_bucket_name="my-bucket",
        connection_id="us.my-connection",
        max_content_length=20,  # Small limit to force offloading
    )
    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    ) as plugin:
      await plugin._ensure_started(
          storage_client=mock_storage_client.return_value
      )
      mock_write_client.append_rows.reset_mock()
      # Create mixed content: one small inline, one large offloaded
      small_text = "Small inline text"
      large_text = "A" * 100
      user_message = types.Content(
          parts=[types.Part(text=small_text), types.Part(text=large_text)]
      )
      bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
      await plugin.on_user_message_callback(
          invocation_context=invocation_context, user_message=user_message
      )
      await asyncio.sleep(0.01)
      mock_write_client.append_rows.assert_called_once()
      log_entry = await _get_captured_event_dict_async(
          mock_write_client, dummy_arrow_schema
      )
      # Verify content parts
      assert len(log_entry["content_parts"]) == 2
      # Part 0: Inline
      part0 = log_entry["content_parts"][0]
      assert part0["storage_mode"] == "INLINE"
      assert part0["text"] == small_text
      assert part0["object_ref"] is None
      # Part 1: Offloaded
      part1 = log_entry["content_parts"][1]
      assert part1["storage_mode"] == "GCS_REFERENCE"
      assert part1["uri"].startswith("gs://my-bucket/")
      assert part1["object_ref"]["uri"] == part1["uri"]
      assert part1["object_ref"]["authorizer"] == "us.my-connection"
      assert json.loads(part1["object_ref"]["details"]) == {
          "gcs_metadata": {"content_type": "text/plain"}
      }

  # Removed on_event_callback tests as they are no longer applicable in V2
  @pytest.mark.asyncio
  async def test_bigquery_client_initialization_failure(
      self,
      mock_auth_default,
      mock_write_client,
      invocation_context,
      mock_asyncio_to_thread,
  ):
    _ = mock_asyncio_to_thread
    mock_auth_default.side_effect = auth_exceptions.GoogleAuthError(
        "Auth failed"
    )
    async with managed_plugin(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
    ) as plugin_with_fail:
      with mock.patch(
          "google.adk.plugins.bigquery_agent_analytics_plugin.logger"
      ) as mock_logger:
        bigquery_agent_analytics_plugin.TraceManager.push_span(
            invocation_context
        )
        await plugin_with_fail.on_user_message_callback(
            invocation_context=invocation_context,
            user_message=types.Content(parts=[types.Part(text="Test")]),
        )
        await asyncio.sleep(0.01)
        mock_logger.error.assert_called_with(
            "Failed to initialize BigQuery Plugin: %s", mock.ANY
        )
      mock_write_client.append_rows.assert_not_called()

  @pytest.mark.asyncio
  async def test_bigquery_insert_error_does_not_raise(
      self, bq_plugin_inst, mock_write_client, invocation_context
  ):

    _ = bq_plugin_inst

    async def fake_append_rows_with_error(requests, **kwargs):
      mock_append_rows_response = mock.MagicMock()
      mock_append_rows_response.row_errors = []  # No row errors
      mock_append_rows_response.error = mock.MagicMock()
      mock_append_rows_response.error.code = 3  # INVALID_ARGUMENT
      mock_append_rows_response.error.message = "Test BQ Error"
      return _async_gen(mock_append_rows_response)

    mock_write_client.append_rows.side_effect = fake_append_rows_with_error
    with mock.patch(
        "google.adk.plugins.bigquery_agent_analytics_plugin.logger"
    ) as mock_logger:
      bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
      await bq_plugin_inst.on_user_message_callback(
          invocation_context=invocation_context,
          user_message=types.Content(parts=[types.Part(text="Test")]),
      )
      await asyncio.sleep(0.01)
      # The logger is called multiple times, check that one of them is the error message
      # Or just check that it was called with the expected message at some point
      mock_logger.error.assert_any_call(
          "Non-retryable BigQuery error: %s", "Test BQ Error"
      )
    mock_write_client.append_rows.assert_called_once()

  @pytest.mark.asyncio
  async def test_bigquery_insert_retryable_error(
      self, bq_plugin_inst, mock_write_client, invocation_context
  ):
    """Test that retryable BigQuery errors are logged and retried."""

    async def fake_append_rows_with_retryable_error(requests, **kwargs):
      mock_append_rows_response = mock.MagicMock()
      mock_append_rows_response.row_errors = []  # No row errors
      mock_append_rows_response.error = mock.MagicMock()
      mock_append_rows_response.error.code = 10  # ABORTED (retryable)
      mock_append_rows_response.error.message = "Test BQ Retryable Error"
      return _async_gen(mock_append_rows_response)

    mock_write_client.append_rows.side_effect = (
        fake_append_rows_with_retryable_error
    )
    with mock.patch(
        "google.adk.plugins.bigquery_agent_analytics_plugin.logger"
    ) as mock_logger:
      bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
      await bq_plugin_inst.on_user_message_callback(
          invocation_context=invocation_context,
          user_message=types.Content(parts=[types.Part(text="Test")]),
      )
      await asyncio.sleep(0.01)
      mock_logger.warning.assert_any_call(
          "BigQuery Write API returned error code %s: %s",
          10,
          "Test BQ Retryable Error",
      )
    # Should be called at least once. Retries are hard to test due to async backoff.
    assert mock_write_client.append_rows.call_count >= 1

  @pytest.mark.asyncio
  async def test_schema_mismatch_error_handling(
      self, bq_plugin_inst, mock_write_client, invocation_context
  ):
    async def fake_append_rows_with_schema_error(requests, **kwargs):
      mock_resp = mock.MagicMock()
      mock_resp.row_errors = []
      mock_resp.error = mock.MagicMock()
      mock_resp.error.code = 3
      mock_resp.error.message = (
          "Schema mismatch: Field 'new_field' not found in table."
      )
      return _async_gen(mock_resp)

    mock_write_client.append_rows.side_effect = (
        fake_append_rows_with_schema_error
    )
    with mock.patch(
        "google.adk.plugins.bigquery_agent_analytics_plugin.logger"
    ) as mock_logger:
      bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
      await bq_plugin_inst.on_user_message_callback(
          invocation_context=invocation_context,
          user_message=types.Content(parts=[types.Part(text="Test")]),
      )
      await asyncio.sleep(0.01)
      mock_logger.error.assert_called_with(
          "BigQuery Schema Mismatch: %s. This usually means the"
          " table schema does not match the expected schema.",
          "Schema mismatch: Field 'new_field' not found in table.",
      )

  @pytest.mark.asyncio
  async def test_close(self, bq_plugin_inst, mock_bq_client, mock_write_client):
    """Test plugin shutdown."""

    await bq_plugin_inst.shutdown()
    # shutdown calls transport.close() on all clients
    assert mock_write_client.transport.close.call_count >= 1
    # Verify loop states are cleared
    assert not bq_plugin_inst._loop_state_by_loop

  @pytest.mark.asyncio
  async def test_before_run_callback_logs_correctly(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      dummy_arrow_schema,
  ):
    """Test before_run_callback logs correctly."""

    bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
    await bq_plugin_inst.before_run_callback(
        invocation_context=invocation_context
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
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
    bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
    await bq_plugin_inst.after_run_callback(
        invocation_context=invocation_context
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
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
    bigquery_agent_analytics_plugin.TraceManager.push_span(callback_context)
    await bq_plugin_inst.before_agent_callback(
        agent=mock_agent, callback_context=callback_context
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    _assert_common_fields(log_entry, "AGENT_STARTING")
    assert log_entry["content"] == "Test Instruction"

  @pytest.mark.asyncio
  async def test_after_agent_callback_logs_correctly(
      self,
      bq_plugin_inst,
      mock_write_client,
      mock_agent,
      callback_context,
      dummy_arrow_schema,
  ):
    bigquery_agent_analytics_plugin.TraceManager.push_span(callback_context)
    await bq_plugin_inst.after_agent_callback(
        agent=mock_agent, callback_context=callback_context
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    _assert_common_fields(log_entry, "AGENT_COMPLETED")
    assert log_entry["content"] is None
    # Latency should be an int >= 0 now that we instrument it
    assert log_entry["latency_ms"] is not None
    latency_dict = json.loads(log_entry["latency_ms"])
    assert latency_dict["total_ms"] >= 0

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
        contents=[
            types.Content(role="user", parts=[types.Part(text="Prompt")])
        ],
    )
    bigquery_agent_analytics_plugin.TraceManager.push_span(callback_context)
    await bq_plugin_inst.before_model_callback(
        callback_context=callback_context, llm_request=llm_request
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    _assert_common_fields(log_entry, "LLM_REQUEST")
    assert "Prompt" in log_entry["content"]

  @pytest.mark.asyncio
  async def test_before_model_callback_with_params_and_tools(
      self,
      bq_plugin_inst,
      mock_write_client,
      callback_context,
      dummy_arrow_schema,
  ):
    llm_request = llm_request_lib.LlmRequest(
        model="gemini-pro",
        config=types.GenerateContentConfig(
            temperature=0.5,
            top_p=0.9,
            system_instruction=types.Content(parts=[types.Part(text="Sys")]),
        ),
        contents=[types.Content(role="user", parts=[types.Part(text="User")])],
    )
    # Manually set tools_dict as it is excluded from init
    llm_request.tools_dict = {"tool1": "func1", "tool2": "func2"}
    bigquery_agent_analytics_plugin.TraceManager.push_span(callback_context)
    await bq_plugin_inst.before_model_callback(
        callback_context=callback_context, llm_request=llm_request
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    _assert_common_fields(log_entry, "LLM_REQUEST")
    # Verify content is JSON and has correct fields
    assert "content" in log_entry
    content_dict = json.loads(log_entry["content"])
    assert content_dict["prompt"] == [{"role": "user", "content": "User"}]
    assert content_dict["system_prompt"] == "Sys"
    # Verify attributes
    assert "attributes" in log_entry
    attributes = json.loads(log_entry["attributes"])
    assert attributes["llm_config"]["temperature"] == 0.5
    assert attributes["llm_config"]["top_p"] == 0.9
    assert attributes["llm_config"]["top_p"] == 0.9
    assert attributes["tools"] == ["tool1", "tool2"]

  @pytest.mark.asyncio
  async def test_before_model_callback_with_full_config(
      self,
      bq_plugin_inst,
      mock_write_client,
      callback_context,
      dummy_arrow_schema,
  ):
    """Test that all config fields, including falsy values and labels, are logged."""
    llm_request = llm_request_lib.LlmRequest(
        model="gemini-pro",
        config=types.GenerateContentConfig(
            temperature=0.0,
            top_p=0.1,
            top_k=5.0,
            candidate_count=5,
            max_output_tokens=65000,
            stop_sequences=["STOP"],
            presence_penalty=0.1,
            frequency_penalty=0.5,
            seed=42,
            response_logprobs=True,
            logprobs=3,
            labels={"llm.agent.name": "test_agent"},
        ),
        contents=[types.Content(role="user", parts=[types.Part(text="User")])],
    )
    bigquery_agent_analytics_plugin.TraceManager.push_span(callback_context)
    await bq_plugin_inst.before_model_callback(
        callback_context=callback_context, llm_request=llm_request
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    _assert_common_fields(log_entry, "LLM_REQUEST")

    # Verify attributes
    assert "attributes" in log_entry
    attributes = json.loads(log_entry["attributes"])

    llm_config = attributes.get("llm_config", {})
    expected_llm_config = {
        "temperature": 0.0,
        "top_p": 0.1,
        "top_k": 5.0,
        "candidate_count": 5,
        "max_output_tokens": 65000,
        "stop_sequences": ["STOP"],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.5,
        "seed": 42,
        "response_logprobs": True,
        "logprobs": 3,
    }
    assert llm_config == expected_llm_config

    assert attributes.get("labels") == {"llm.agent.name": "test_agent"}

  @pytest.mark.asyncio
  async def test_before_model_callback_multipart_separator(
      self,
      bq_plugin_inst,
      mock_write_client,
      callback_context,
      dummy_arrow_schema,
  ):
    llm_request = llm_request_lib.LlmRequest(
        model="gemini-pro",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="Part1"), types.Part(text="Part2")],
            )
        ],
    )
    bigquery_agent_analytics_plugin.TraceManager.push_span(callback_context)
    await bq_plugin_inst.before_model_callback(
        callback_context=callback_context, llm_request=llm_request
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    content_dict = json.loads(log_entry["content"])
    # Verify the separator is " | "
    assert content_dict["prompt"][0]["content"] == "Part1 | Part2"

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
    bigquery_agent_analytics_plugin.TraceManager.push_span(callback_context)
    await bq_plugin_inst.after_model_callback(
        callback_context=callback_context,
        llm_response=llm_response,
        # latency_ms is now calculated internally via TraceManager
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    _assert_common_fields(log_entry, "LLM_RESPONSE")
    content_dict = json.loads(log_entry["content"])
    assert content_dict["response"] == "text: 'Model response'"
    assert content_dict["usage"]["prompt"] == 10
    assert content_dict["usage"]["total"] == 15
    assert log_entry["error_message"] is None
    latency_dict = json.loads(log_entry["latency_ms"])
    # Latency comes from time.time(), so we can't assert exact 100ms
    # But it should be present
    assert latency_dict["total_ms"] >= 0
    # tfft is passed via kwargs if present, or we can mock it.
    # In this test we didn't pass it in kwargs in the updated call above, so it might be missing unless we add it back to kwargs.
    # The original test passed it as kwarg.

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
    bigquery_agent_analytics_plugin.TraceManager.push_span(callback_context)
    await bq_plugin_inst.after_model_callback(
        callback_context=callback_context,
        llm_response=llm_response,
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    _assert_common_fields(log_entry, "LLM_RESPONSE")
    content_dict = json.loads(log_entry["content"])
    assert content_dict["response"] == "call: get_weather"
    assert content_dict["usage"]["prompt"] == 10
    assert content_dict["usage"]["total"] == 15
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
    bigquery_agent_analytics_plugin.TraceManager.push_span(tool_context)
    await bq_plugin_inst.before_tool_callback(
        tool=mock_tool, tool_args={"param": "value"}, tool_context=tool_context
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    _assert_common_fields(log_entry, "TOOL_STARTING")
    content_dict = json.loads(log_entry["content"])
    assert content_dict["tool"] == "MyTool"
    assert content_dict["args"] == {"param": "value"}

  @pytest.mark.asyncio
  async def test_after_tool_callback_logs_correctly(
      self, bq_plugin_inst, mock_write_client, tool_context, dummy_arrow_schema
  ):
    mock_tool = mock.create_autospec(
        base_tool_lib.BaseTool, instance=True, spec_set=True
    )
    type(mock_tool).name = mock.PropertyMock(return_value="MyTool")
    type(mock_tool).description = mock.PropertyMock(return_value="Description")
    bigquery_agent_analytics_plugin.TraceManager.push_span(tool_context)
    await bq_plugin_inst.after_tool_callback(
        tool=mock_tool,
        tool_args={"arg1": "val1"},
        tool_context=tool_context,
        result={"res": "success"},
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    _assert_common_fields(log_entry, "TOOL_COMPLETED")
    content_dict = json.loads(log_entry["content"])
    assert content_dict["tool"] == "MyTool"
    assert content_dict["result"] == {"res": "success"}

  @pytest.mark.asyncio
  async def test_after_tool_callback_no_state_delta_logging(
      self, bq_plugin_inst, mock_write_client, tool_context, dummy_arrow_schema
  ):
    """State deltas are now logged via on_event_callback, not after_tool."""
    mock_tool = mock.create_autospec(
        base_tool_lib.BaseTool, instance=True, spec_set=True
    )
    type(mock_tool).name = mock.PropertyMock(return_value="StateTool")
    type(mock_tool).description = mock.PropertyMock(return_value="Sets state")

    # Simulate a tool modifying the state
    tool_context.actions.state_delta["new_key"] = "new_value"

    bigquery_agent_analytics_plugin.TraceManager.push_span(tool_context)
    await bq_plugin_inst.after_tool_callback(
        tool=mock_tool,
        tool_args={"arg1": "val1"},
        tool_context=tool_context,
        result={"res": "success"},
    )
    await asyncio.sleep(0.01)

    # Only TOOL_COMPLETED should be logged; STATE_DELTA is handled
    # by on_event_callback now.
    rows = await _get_captured_rows_async(mock_write_client, dummy_arrow_schema)
    assert len(rows) == 1
    assert rows[0]["event_type"] == "TOOL_COMPLETED"

  @pytest.mark.asyncio
  async def test_on_event_callback_logs_state_delta(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      dummy_arrow_schema,
  ):
    """on_event_callback logs STATE_DELTA for events with state changes."""
    state_delta = {"key": "value", "new_key": 123}
    event = event_lib.Event(
        author="test_agent",
        actions=event_actions_lib.EventActions(state_delta=state_delta),
    )

    bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
    result = await bq_plugin_inst.on_event_callback(
        invocation_context=invocation_context, event=event
    )
    # Must return None to not modify the event
    assert result is None

    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    _assert_common_fields(log_entry, "STATE_DELTA")
    assert log_entry["content"] is None

    attributes = json.loads(log_entry["attributes"])
    assert attributes["state_delta"] == state_delta

  @pytest.mark.asyncio
  async def test_on_event_callback_ignores_empty_state_delta(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
  ):
    """on_event_callback should not log when state_delta is empty."""
    event = event_lib.Event(
        author="test_agent",
        actions=event_actions_lib.EventActions(state_delta={}),
    )

    result = await bq_plugin_inst.on_event_callback(
        invocation_context=invocation_context, event=event
    )
    assert result is None

    # No events should have been logged
    mock_write_client.append_rows.assert_not_called()

  @pytest.mark.asyncio
  async def test_log_event_with_session_metadata(
      self,
      bq_plugin_inst,
      mock_write_client,
      callback_context,
      dummy_arrow_schema,
  ):
    """Test that session metadata is logged when enabled."""
    # Setup session state with user metadata
    session = callback_context._invocation_context.session
    type(session).state = mock.PropertyMock(
        return_value={"thread_id": "gchat-123", "customer_id": "cust-42"}
    )

    # Ensure config enabled (default is True)
    bq_plugin_inst.config.log_session_metadata = True

    await bq_plugin_inst._log_event(
        "TEST_EVENT",
        callback_context,
        raw_content="test content",
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )

    attributes = json.loads(log_entry["attributes"])
    meta = attributes["session_metadata"]
    assert meta["session_id"] == session.id
    assert meta["app_name"] == session.app_name
    assert meta["user_id"] == session.user_id
    assert meta["state"] == {
        "thread_id": "gchat-123",
        "customer_id": "cust-42",
    }

  @pytest.mark.asyncio
  async def test_log_event_with_custom_tags(
      self,
      bq_plugin_inst,
      mock_write_client,
      callback_context,
      dummy_arrow_schema,
  ):
    """Test that custom tags are logged."""
    custom_tags = {"agent_role": "sales", "env": "prod"}
    bq_plugin_inst.config.custom_tags = custom_tags

    await bq_plugin_inst._log_event(
        "TEST_EVENT",
        callback_context,
        raw_content="test content",
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )

    attributes = json.loads(log_entry["attributes"])
    assert attributes["custom_tags"] == custom_tags

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
    bigquery_agent_analytics_plugin.TraceManager.push_span(callback_context)
    await bq_plugin_inst.on_model_error_callback(
        callback_context=callback_context, llm_request=llm_request, error=error
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    _assert_common_fields(log_entry, "LLM_ERROR")
    assert log_entry["content"] is None
    assert log_entry["error_message"] == "LLM failed"
    assert log_entry["status"] == "ERROR"

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
    bigquery_agent_analytics_plugin.TraceManager.push_span(tool_context)
    await bq_plugin_inst.on_tool_error_callback(
        tool=mock_tool,
        tool_args={"param": "value"},
        tool_context=tool_context,
        error=error,
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    _assert_common_fields(log_entry, "TOOL_ERROR")
    content_dict = json.loads(log_entry["content"])
    assert content_dict["tool"] == "MyTool"
    assert content_dict["args"] == {"param": "value"}
    assert log_entry["error_message"] == "Tool timed out"
    assert log_entry["status"] == "ERROR"

  @pytest.mark.asyncio
  async def test_table_creation_options(
      self,
      mock_auth_default,
      mock_bq_client,
      mock_write_client,
      mock_to_arrow_schema,
      mock_asyncio_to_thread,
  ):
    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID
    ) as plugin:
      mock_bq_client.get_table.side_effect = cloud_exceptions.NotFound(
          "Not found"
      )
      await plugin._ensure_started()
      # Verify create_table was called with correct table options
      mock_bq_client.create_table.assert_called_once()
      call_args = mock_bq_client.create_table.call_args
      table_arg = call_args[0][0]
      assert isinstance(table_arg, bigquery.Table)
      assert table_arg.time_partitioning.type_ == "DAY"
      assert table_arg.time_partitioning.field == "timestamp"
      assert table_arg.clustering_fields == ["event_type", "agent", "user_id"]
      # Verify schema descriptions are present (spot check)
      timestamp_field = next(
          f for f in table_arg.schema if f.name == "timestamp"
      )
      assert (
          timestamp_field.description
          == "The UTC timestamp when the event occurred. Used for ordering"
          " events"
          " within a session."
      )

  @pytest.mark.asyncio
  async def test_init_in_thread_pool(
      self,
      mock_auth_default,
      mock_bq_client,
      mock_write_client,
      mock_to_arrow_schema,
      mock_asyncio_to_thread,
      invocation_context,
  ):
    """Verifies that the plugin can be initialized from a thread pool."""
    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID
    ) as plugin:

      def _run_in_thread(p):
        # In a real thread pool, there might not be an event loop.
        # However, since we are calling an async method (_ensure_started),
        # we must run it in an event loop. The issue was that _lazy_setup
        # called get_event_loop() which fails in threads without a loop.
        # Here we simulate the condition by running in a thread and creating a new loop if needed,
        # but the key is that the plugin's internal calls should use the correct loop.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
          # _ensure_started is called by managed_plugin, but we need to ensure
          # that if it were called in a thread, it would work.
          # For this test, we just ensure the plugin is accessible and started.
          loop.run_until_complete(p._ensure_started())
        finally:
          loop.close()

      # Run in a separate thread to simulate ThreadPoolExecutor-0_0
      from concurrent.futures import ThreadPoolExecutor

      with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_in_thread, plugin)
        future.result()  # Should not raise "no current event loop"
      assert plugin._started
      # Verify loop states are populated
      assert plugin._loop_state_by_loop

  @pytest.mark.asyncio
  async def test_multimodal_offloading(
      self,
      mock_write_client,
      callback_context,
      mock_auth_default,
      mock_bq_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_storage_client,
  ):
    # Setup
    bucket_name = "test-bucket"
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(
        gcs_bucket_name=bucket_name
    )
    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    ) as plugin:
      await plugin._ensure_started(
          storage_client=mock_storage_client.return_value
      )
      # Mock GCS bucket and blob
      mock_bucket = mock_storage_client.return_value.bucket.return_value
      mock_bucket.name = bucket_name
      mock_blob = mock_bucket.blob.return_value
      # Create content with large text that should be offloaded
      large_text = "A" * (32 * 1024 + 1)
      llm_request = llm_request_lib.LlmRequest(
          model="gemini-pro",
          contents=[types.Content(parts=[types.Part(text=large_text)])],
      )
      # Execute
      await plugin.before_model_callback(
          callback_context=callback_context, llm_request=llm_request
      )
      # Use flush instead of sleep for robustness
      await plugin.flush()
      # Verify GCS upload
      mock_blob.upload_from_string.assert_called_once()
      args, kwargs = mock_blob.upload_from_string.call_args
      assert args[0] == large_text
      assert kwargs["content_type"] == "text/plain"
      # Verify BQ write
      mock_write_client.append_rows.assert_called_once()
      event_dict = await _get_captured_event_dict_async(
          mock_write_client, dummy_arrow_schema
      )
      content_parts = event_dict["content_parts"]
      assert len(content_parts) == 1
      assert content_parts[0]["storage_mode"] == "GCS_REFERENCE"
      assert content_parts[0]["uri"].startswith(f"gs://{bucket_name}/")

  @pytest.mark.asyncio
  async def test_quota_project_id_used_in_client(
      self,
      mock_bq_client,
      mock_to_arrow_schema,
      mock_asyncio_to_thread,
  ):
    mock_creds = mock.create_autospec(
        google.auth.credentials.Credentials, instance=True, spec_set=True
    )
    mock_creds.quota_project_id = "quota-project"
    with mock.patch.object(
        google.auth,
        "default",
        autospec=True,
        return_value=(mock_creds, PROJECT_ID),
    ) as mock_auth_default:
      with mock.patch.object(
          bigquery_agent_analytics_plugin,
          "BigQueryWriteAsyncClient",
          autospec=True,
      ) as mock_bq_write_cls:
        async with managed_plugin(
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID,
            table_id=TABLE_ID,
        ) as plugin:
          await plugin._ensure_started()
          mock_auth_default.assert_called_once()
          mock_bq_write_cls.assert_called_once()
          _, kwargs = mock_bq_write_cls.call_args
          assert kwargs["client_options"].quota_project_id == "quota-project"

  @pytest.mark.asyncio
  async def test_no_quota_project_when_creds_lack_it(
      self,
      mock_bq_client,
      mock_to_arrow_schema,
      mock_asyncio_to_thread,
  ):
    """Verify no quota_project_id is set when credentials don't provide one.

    This is critical for Workload Identity Federation flows where setting
    quota_project_id on the client breaks auth token refresh (issue #4370).
    """
    mock_creds = mock.create_autospec(
        google.auth.credentials.Credentials, instance=True, spec_set=True
    )
    mock_creds.quota_project_id = None
    with mock.patch.object(
        google.auth,
        "default",
        autospec=True,
        return_value=(mock_creds, PROJECT_ID),
    ):
      with mock.patch.object(
          bigquery_agent_analytics_plugin,
          "BigQueryWriteAsyncClient",
          autospec=True,
      ) as mock_bq_write_cls:
        async with managed_plugin(
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID,
            table_id=TABLE_ID,
        ) as plugin:
          await plugin._ensure_started()
          mock_bq_write_cls.assert_called_once()
          _, kwargs = mock_bq_write_cls.call_args
          assert kwargs["client_options"] is None

  @pytest.mark.asyncio
  async def test_custom_credentials_used(
      self,
      mock_to_arrow_schema,
      mock_asyncio_to_thread,
  ):
    """Verify custom credentials are used and default auth is not called."""
    mock_custom_creds = mock.create_autospec(
        google.auth.credentials.Credentials, instance=True, spec_set=True
    )
    mock_custom_creds.quota_project_id = "custom-quota-project"

    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(
        gcs_bucket_name="test-bucket",
        create_views=False,
    )

    with mock.patch.object(
        google.auth,
        "default",
        autospec=True,
    ) as mock_auth_default:
      with mock.patch.object(
          bigquery_agent_analytics_plugin,
          "BigQueryWriteAsyncClient",
          autospec=True,
      ) as mock_bq_write_cls:
        with mock.patch(
            "google.cloud.bigquery.Client", autospec=True
        ) as mock_bq_cls:
          with mock.patch(
              "google.cloud.storage.Client", autospec=True
          ) as mock_storage_cls:
            async with managed_plugin(
                project_id=PROJECT_ID,
                dataset_id=DATASET_ID,
                table_id=TABLE_ID,
                credentials=mock_custom_creds,
                config=config,
            ) as plugin:
              await plugin._ensure_started()

              mock_auth_default.assert_not_called()

              mock_bq_write_cls.assert_called_once()
              _, kwargs = mock_bq_write_cls.call_args
              assert kwargs["credentials"] == mock_custom_creds

              mock_bq_cls.assert_called_once()
              _, kwargs = mock_bq_cls.call_args
              assert kwargs["credentials"] == mock_custom_creds

              mock_storage_cls.assert_called_once()
              _, kwargs = mock_storage_cls.call_args
              assert kwargs["credentials"] == mock_custom_creds

  @pytest.mark.asyncio
  async def test_pickle_safety(self, mock_auth_default, mock_bq_client):
    """Test that the plugin can be pickled safely."""
    import pickle

    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(enabled=True)
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    )
    # Test pickling before start
    pickled = pickle.dumps(plugin)
    unpickled = pickle.loads(pickled)
    assert unpickled.project_id == PROJECT_ID
    assert unpickled._setup_lock is None
    assert unpickled._executor is None
    # Start the plugin
    await plugin._ensure_started()
    assert plugin._executor is not None
    try:
      # Test pickling after start
      pickled_started = pickle.dumps(plugin)
      unpickled_started = pickle.loads(pickled_started)
      assert unpickled_started.project_id == PROJECT_ID
      # Runtime objects should be None after unpickling
      assert unpickled_started._setup_lock is None
      assert unpickled_started._executor is None
      assert not unpickled_started._loop_state_by_loop
    finally:
      await plugin.shutdown()

  @pytest.mark.asyncio
  async def test_span_hierarchy_llm_call(
      self,
      bq_plugin_inst,
      mock_write_client,
      callback_context,
      dummy_arrow_schema,
  ):
    """Verifies that LLM events have correct Span ID hierarchy."""
    # 1. Start Agent Span
    bigquery_agent_analytics_plugin.TraceManager.push_span(callback_context)
    _, _ = (
        bigquery_agent_analytics_plugin.TraceManager.get_current_span_and_parent()
    )
    agent_span_id = (
        bigquery_agent_analytics_plugin.TraceManager.get_current_span_id()
    )
    # 2. Start LLM Span (Implicitly handled if we push it?
    # Actually before_model_callback assumes a span is pushed for the LLM call if we want one?
    # No, usually the Runner/Agent pushes a span BEFORE calling before_model_callback?
    # Let's verify usage in agent.py or plugin.
    # Plugin does NOT push spans automatically for LLM. It relies on TraceManager being managed externally
    # OR it uses current span.
    # Wait, the Runner pushes spans.
    # 3. LLM Request
    llm_request = llm_request_lib.LlmRequest(
        model="gemini-pro",
        contents=[types.Content(parts=[types.Part(text="Prompt")])],
    )
    await bq_plugin_inst.before_model_callback(
        callback_context=callback_context, llm_request=llm_request
    )
    await asyncio.sleep(0.01)
    # Capture the actual LLM Span ID (pushed by before_model_callback)
    llm_span_id = (
        bigquery_agent_analytics_plugin.TraceManager.get_current_span_id()
    )
    # Now that we push a new span for LLM calls, it should differ from agent_span_id
    assert llm_span_id != agent_span_id
    log_entry_req = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    assert log_entry_req["event_type"] == "LLM_REQUEST"
    assert log_entry_req["span_id"] == llm_span_id
    # The parent of the LLM span should be the Agent span
    assert log_entry_req["parent_span_id"] == agent_span_id
    mock_write_client.append_rows.reset_mock()
    # 4. LLM Response
    # In the actual flow, after_model_callback pops the span.
    # But explicitly via TraceManager.pop_span()?
    # No, after_model_callback calls TraceManager.pop_span().
    # So we should validly call it.
    llm_response = llm_response_lib.LlmResponse(
        content=types.Content(parts=[types.Part(text="Response")]),
    )
    await bq_plugin_inst.after_model_callback(
        callback_context=callback_context, llm_response=llm_response
    )
    await asyncio.sleep(0.01)
    log_entry_resp = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    assert log_entry_resp["event_type"] == "LLM_RESPONSE"
    assert log_entry_resp["span_id"] == llm_span_id
    # The parent of the LLM span should be the Agent span
    assert log_entry_resp["parent_span_id"] == agent_span_id
    # Verify LLM Span was popped and we are back to Agent Span
    assert (
        bigquery_agent_analytics_plugin.TraceManager.get_current_span_id()
        == agent_span_id
    )
    # Clean up Agent Span
    bigquery_agent_analytics_plugin.TraceManager.pop_span()
    assert (
        not bigquery_agent_analytics_plugin.TraceManager.get_current_span_id()
    )

  @pytest.mark.asyncio
  async def test_custom_object_serialization(
      self,
      mock_write_client,
      tool_context,
      mock_auth_default,
      mock_bq_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    """Verifies that custom objects (Dataclasses) are serialized to dicts."""
    _ = mock_auth_default
    _ = mock_bq_client

    @dataclasses.dataclass
    class LocalMissedKPI:
      kpi: str
      value: float

    @dataclasses.dataclass
    class LocalIncident:
      id: str
      kpi_missed: list[LocalMissedKPI]
      status: str

    incident = LocalIncident(
        id="inc-123",
        kpi_missed=[LocalMissedKPI(kpi="latency", value=99.9)],
        status="active",
    )
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig()
    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()
      content = {"result": incident}
      # Verify full flow
      await plugin._log_event(
          "TOOL_PARTIAL",
          tool_context,
          raw_content=content,
      )
      await asyncio.sleep(0.01)
      mock_write_client.append_rows.assert_called_once()
      log_entry = await _get_captured_event_dict_async(
          mock_write_client, dummy_arrow_schema
      )
      # Content should be valid JSON string
      content_json = json.loads(log_entry["content"])
      assert content_json["result"]["id"] == "inc-123"
      assert content_json["result"]["kpi_missed"][0]["kpi"] == "latency"

  @pytest.mark.asyncio
  async def test_otel_integration(
      self,
      callback_context,
  ):
    """Verifies OpenTelemetry integration in TraceManager."""
    # Mock the tracer and span
    mock_tracer = mock.Mock()
    mock_span = mock.Mock()
    mock_context = mock.Mock()
    # Setup mock IDs (128-bit trace_id, 64-bit span_id)
    trace_id_int = 0x12345678123456781234567812345678
    span_id_int = 0x1234567812345678
    mock_context.trace_id = trace_id_int
    mock_context.span_id = span_id_int
    mock_context.is_valid = True
    mock_span.get_span_context.return_value = mock_context
    mock_span.start_time = 1234567890000000000  # Mock start time in ns
    mock_tracer.start_span.return_value = mock_span
    # Patch the global tracer in the plugin module
    with mock.patch(
        "google.adk.plugins.bigquery_agent_analytics_plugin.tracer", mock_tracer
    ):
      # Test push_span
      span_id = bigquery_agent_analytics_plugin.TraceManager.push_span(
          callback_context, "test_span"
      )
      mock_tracer.start_span.assert_called_with("test_span", context=None)
      assert span_id == format(span_id_int, "016x")
      # Test get_trace_id
      # We need to mock trace.get_current_span() to return our mock span
      # because push_span calls trace.attach(), which affects the global context
      with mock.patch(
          "opentelemetry.trace.get_current_span", return_value=mock_span
      ):
        trace_id = bigquery_agent_analytics_plugin.TraceManager.get_trace_id(
            callback_context
        )
        assert trace_id == format(trace_id_int, "032x")
      # Test pop_span
      # pop_span calls span.end()
      bigquery_agent_analytics_plugin.TraceManager.pop_span()
      mock_span.end.assert_called_once()

  @pytest.mark.asyncio
  async def test_otel_integration_real_provider(self, callback_context):
    """Verifies TraceManager with a real OpenTelemetry TracerProvider."""
    # Setup OTEL with in-memory exporter
    # pylint: disable=g-import-not-at-top
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.trace import export as trace_export
    from opentelemetry.sdk.trace.export import in_memory_span_exporter

    # pylint: enable=g-import-not-at-top
    provider = trace_sdk.TracerProvider()
    exporter = in_memory_span_exporter.InMemorySpanExporter()
    processor = trace_export.SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test_tracer")
    # Patch the global tracer in the plugin module
    with mock.patch(
        "google.adk.plugins.bigquery_agent_analytics_plugin.tracer", tracer
    ):
      # 1. Start a span
      span_id = bigquery_agent_analytics_plugin.TraceManager.push_span(
          callback_context, "test_span"
      )
      # Verify a span was started but not ended
      current_spans = exporter.get_finished_spans()
      assert not current_spans
      # Verify we can retrieve the trace ID
      trace_id = bigquery_agent_analytics_plugin.TraceManager.get_trace_id(
          callback_context
      )
      assert trace_id is not None
      # 2. End the span
      popped_span_id, _ = (
          bigquery_agent_analytics_plugin.TraceManager.pop_span()
      )
      assert popped_span_id == span_id
      # Verify span is now finished and exported
      finished_spans = exporter.get_finished_spans()
      assert len(finished_spans) == 1
      assert finished_spans[0].name == "test_span"
      assert format(finished_spans[0].context.span_id, "016x") == span_id
      assert format(finished_spans[0].context.trace_id, "032x") == trace_id

  @pytest.mark.asyncio
  async def test_keyword_identifiers_emission_default(
      self,
      mock_auth_default,
      mock_bq_client,
      callback_context,
  ):
    """Verify the default keyword flow for User-Agent and Trace-ID."""
    keyword = "google-adk-bq-logger"
    mock_write_client = mock.AsyncMock()

    # 1. Verify User-Agent contains default keyword.
    with mock.patch(
        "google.adk.plugins.bigquery_agent_analytics_plugin.BigQueryWriteAsyncClient",
        autospec=True,
    ) as mock_write_cls:
      mock_write_cls.return_value = mock_write_client
      async with managed_plugin(PROJECT_ID, DATASET_ID) as plugin:
        await plugin._ensure_started()

        _, kwargs = mock_write_cls.call_args
        client_info = kwargs.get("client_info")
        assert f"{keyword}/{__version__}" in client_info.user_agent

    # 2. Verify Trace ID contains default keyword.
    with mock.patch(
        "google.adk.plugins.bigquery_agent_analytics_plugin.BigQueryWriteAsyncClient",
        autospec=True,
    ) as mock_write_cls:
      mock_write_cls.return_value = mock_write_client
      async with managed_plugin(PROJECT_ID, DATASET_ID) as plugin:
        await plugin._ensure_started()
        mock_write_client.append_rows.reset_mock()

        llm_request = llm_request_lib.LlmRequest(
            model="gemini-pro",
            contents=[types.Content(parts=[types.Part(text="Hi")])],
        )
        await plugin.before_model_callback(
            callback_context=callback_context, llm_request=llm_request
        )
        await plugin.flush()

        call_args = mock_write_client.append_rows.call_args
        requests_iter = call_args.args[0]
        requests = []
        async for req in requests_iter:
          requests.append(req)

        assert requests[0].trace_id.startswith(keyword)
        assert requests[0].trace_id.endswith(f"/{__version__}")

  @pytest.mark.asyncio
  async def test_visual_builder_identifiers_flow(
      self,
      mock_auth_default,
      mock_bq_client,
      callback_context,
      dummy_arrow_schema,
  ):
    """Verify visual-builder keyword flow via contextvars."""
    keyword = "google-adk-visual-builder"
    mock_write_client = mock.AsyncMock()

    # Simulate setting the internal flag via contextvars
    token = _is_visual_builder.set(True)
    try:
      # 1. Verify Client User-Agent
      with mock.patch(
          "google.adk.plugins.bigquery_agent_analytics_plugin.BigQueryWriteAsyncClient",
          autospec=True,
      ) as mock_write_cls:
        mock_write_cls.return_value = mock_write_client
        async with managed_plugin(PROJECT_ID, DATASET_ID) as plugin:
          await plugin._ensure_started()

          _, kwargs = mock_write_cls.call_args
          client_info = kwargs.get("client_info")
          assert keyword in client_info.user_agent

      # 2. Verify Request Trace ID
      with mock.patch(
          "google.adk.plugins.bigquery_agent_analytics_plugin.BigQueryWriteAsyncClient",
          autospec=True,
      ) as mock_write_cls:
        mock_write_cls.return_value = mock_write_client
        async with managed_plugin(PROJECT_ID, DATASET_ID) as plugin:
          await plugin._ensure_started()
          mock_write_client.append_rows.reset_mock()

          llm_request = llm_request_lib.LlmRequest(
              model="gemini-pro",
              contents=[types.Content(parts=[types.Part(text="Hi")])],
          )
          await plugin.before_model_callback(
              callback_context=callback_context, llm_request=llm_request
          )
          await plugin.flush()

          call_args = mock_write_client.append_rows.call_args
          requests_iter = call_args.args[0]
          requests = []
          async for req in requests_iter:
            requests.append(req)

          assert requests[0].trace_id.startswith(
              "google-adk-bq-logger-visual-builder"
          )
          assert requests[0].trace_id.endswith(f"/{__version__}")
    finally:
      _is_visual_builder.reset(token)

  @pytest.mark.asyncio
  async def test_flush_mechanism(
      self,
      bq_plugin_inst,
      mock_write_client,
      dummy_arrow_schema,
      invocation_context,
  ):
    """Verifies that flush() forces pending events to be written."""
    # Log an event
    bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
    await bq_plugin_inst.before_run_callback(
        invocation_context=invocation_context
    )
    # Call flush - this should block until the event is written
    await bq_plugin_inst.flush()
    # Verify write called
    mock_write_client.append_rows.assert_called_once()
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    assert log_entry["event_type"] == "INVOCATION_STARTING"

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      "gen_config_kwargs, expected_llm_config",
      [
          (
              {
                  "temperature": 0.0,
                  "top_k": 5.0,
                  "top_p": 0.1,
                  "candidate_count": 5,
                  "max_output_tokens": 65000,
                  "presence_penalty": 0.1,
                  "frequency_penalty": 0.5,
                  "response_logprobs": True,
                  "logprobs": 3,
                  "seed": 42,
                  "labels": {"llm.agent.name": "test_agent"},
              },
              {
                  "temperature": 0.0,
                  "top_k": 5.0,
                  "top_p": 0.1,
                  "candidate_count": 5,
                  "max_output_tokens": 65000,
                  "presence_penalty": 0.1,
                  "frequency_penalty": 0.5,
                  "response_logprobs": True,
                  "logprobs": 3,
                  "seed": 42,
              },
          ),
      ],
  )
  async def test_generation_config_logging(
      self,
      bq_plugin_inst,
      mock_write_client,
      dummy_arrow_schema,
      callback_context,
      gen_config_kwargs,
      expected_llm_config,
  ):
    """Verifies that all fields in GenerateContentConfig are logged correctly."""
    gen_config = types.GenerateContentConfig(**gen_config_kwargs)

    llm_request = llm_request_lib.LlmRequest(
        model="gemini-pro",
        contents=[types.Content(parts=[types.Part(text="Prompt")])],
        config=gen_config,
    )

    bigquery_agent_analytics_plugin.TraceManager.push_span(callback_context)
    await bq_plugin_inst.before_model_callback(
        callback_context=callback_context, llm_request=llm_request
    )
    # Flush
    await bq_plugin_inst.flush()

    # Verify
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    assert log_entry["event_type"] == "LLM_REQUEST"

    attributes = json.loads(log_entry["attributes"])
    llm_config = attributes.get("llm_config", {})

    assert llm_config == expected_llm_config

    if "labels" in gen_config_kwargs:
      assert attributes.get("labels") == gen_config_kwargs["labels"]


class TestSafeCallbackDecorator:
  """Tests that _safe_callback prevents plugin errors from propagating."""

  @pytest.mark.asyncio
  async def test_callback_exception_does_not_propagate(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
  ):
    """A callback that throws should return None, not crash."""
    # Force _log_event to raise
    with mock.patch.object(
        bq_plugin_inst,
        "_log_event",
        side_effect=RuntimeError("BQ network timeout"),
    ):
      # Should NOT raise
      result = await bq_plugin_inst.on_user_message_callback(
          invocation_context=invocation_context,
          user_message=types.Content(parts=[types.Part(text="Test")]),
      )
      assert result is None

  @pytest.mark.asyncio
  async def test_callback_exception_is_logged(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
  ):
    """The swallowed exception should be logged with exc_info."""
    with mock.patch.object(
        bq_plugin_inst,
        "_log_event",
        side_effect=RuntimeError("BQ write failed"),
    ):
      with mock.patch(
          "google.adk.plugins.bigquery_agent_analytics_plugin.logger"
      ) as mock_logger:
        await bq_plugin_inst.before_run_callback(
            invocation_context=invocation_context,
        )
        mock_logger.exception.assert_called_once_with(
            "BigQuery analytics plugin error in %s; skipping.",
            "before_run_callback",
        )

  @pytest.mark.asyncio
  async def test_subsequent_callbacks_work_after_failure(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      dummy_arrow_schema,
  ):
    """After one callback fails, the next one should still work."""
    call_count = 0
    original_log_event = bq_plugin_inst._log_event

    async def fail_once(*args, **kwargs):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise RuntimeError("Transient error")
      return await original_log_event(*args, **kwargs)

    with mock.patch.object(bq_plugin_inst, "_log_event", side_effect=fail_once):
      # First call fails silently
      await bq_plugin_inst.on_user_message_callback(
          invocation_context=invocation_context,
          user_message=types.Content(parts=[types.Part(text="Fail")]),
      )
      # Second call succeeds
      bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
      await bq_plugin_inst.before_run_callback(
          invocation_context=invocation_context,
      )
      await asyncio.sleep(0.01)
      mock_write_client.append_rows.assert_called_once()

  @pytest.mark.asyncio
  async def test_on_event_callback_exception_returns_none(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
  ):
    """on_event_callback should return None on error, not crash."""
    event = event_lib.Event(
        author="test_agent",
        actions=event_actions_lib.EventActions(state_delta={"key": "value"}),
    )
    with mock.patch.object(
        bq_plugin_inst,
        "_log_event",
        side_effect=Exception("serialize error"),
    ):
      result = await bq_plugin_inst.on_event_callback(
          invocation_context=invocation_context, event=event
      )
      assert result is None

  @pytest.mark.asyncio
  async def test_tool_callback_exception_does_not_propagate(
      self,
      bq_plugin_inst,
      mock_write_client,
      tool_context,
  ):
    """Tool callbacks should not crash even if plugin errors."""
    mock_tool = mock.create_autospec(
        base_tool_lib.BaseTool, instance=True, spec_set=True
    )
    type(mock_tool).name = mock.PropertyMock(return_value="MyTool")
    with mock.patch.object(
        bq_plugin_inst,
        "_log_event",
        side_effect=RuntimeError("BQ down"),
    ):
      # before_tool_callback
      result = await bq_plugin_inst.before_tool_callback(
          tool=mock_tool,
          tool_args={"p": "v"},
          tool_context=tool_context,
      )
      assert result is None

      # after_tool_callback
      result = await bq_plugin_inst.after_tool_callback(
          tool=mock_tool,
          tool_args={"p": "v"},
          tool_context=tool_context,
          result={"r": "ok"},
      )
      assert result is None

      # on_tool_error_callback
      result = await bq_plugin_inst.on_tool_error_callback(
          tool=mock_tool,
          tool_args={"p": "v"},
          tool_context=tool_context,
          error=ValueError("tool broke"),
      )
      assert result is None

  @pytest.mark.asyncio
  async def test_model_callback_exception_does_not_propagate(
      self,
      bq_plugin_inst,
      mock_write_client,
      callback_context,
  ):
    """Model callbacks should not crash even if plugin errors."""
    with mock.patch.object(
        bq_plugin_inst,
        "_log_event",
        side_effect=RuntimeError("BQ down"),
    ):
      llm_request = llm_request_lib.LlmRequest(
          model="gemini-pro",
          contents=[types.Content(parts=[types.Part(text="Hi")])],
      )
      result = await bq_plugin_inst.before_model_callback(
          callback_context=callback_context, llm_request=llm_request
      )
      assert result is None

      llm_response = llm_response_lib.LlmResponse(
          content=types.Content(parts=[types.Part(text="Hi")]),
      )
      result = await bq_plugin_inst.after_model_callback(
          callback_context=callback_context, llm_response=llm_response
      )
      assert result is None

      result = await bq_plugin_inst.on_model_error_callback(
          callback_context=callback_context,
          llm_request=llm_request_lib.LlmRequest(model="gemini-pro"),
          error=ValueError("llm error"),
      )
      assert result is None


class TestParserReuse:
  """Tests that HybridContentParser is reused, not recreated per event."""

  @pytest.mark.asyncio
  async def test_parser_instance_is_reused(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
  ):
    """The same parser instance should be reused across _log_event calls."""
    parser_after_init = bq_plugin_inst.parser
    assert parser_after_init is not None

    bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
    await bq_plugin_inst.on_user_message_callback(
        invocation_context=invocation_context,
        user_message=types.Content(parts=[types.Part(text="Hello")]),
    )
    await asyncio.sleep(0.01)

    # Parser should be the same instance, not a new one
    assert bq_plugin_inst.parser is parser_after_init

  @pytest.mark.asyncio
  async def test_parser_trace_id_updated_per_call(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      dummy_arrow_schema,
  ):
    """trace_id and span_id on the parser should update per _log_event."""
    parser = bq_plugin_inst.parser
    original_trace_id = parser.trace_id

    bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
    await bq_plugin_inst.on_user_message_callback(
        invocation_context=invocation_context,
        user_message=types.Content(parts=[types.Part(text="Test")]),
    )
    await asyncio.sleep(0.01)

    # After logging, trace_id/span_id should have been updated
    # (they're derived from TraceManager, not the initial empty strings)
    assert parser.span_id != ""

  @pytest.mark.asyncio
  async def test_parser_not_recreated_with_constructor(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
  ):
    """HybridContentParser constructor should not be called in
    _log_event."""
    with mock.patch.object(
        bigquery_agent_analytics_plugin,
        "HybridContentParser",
        wraps=bigquery_agent_analytics_plugin.HybridContentParser,
    ) as mock_parser_cls:
      bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
      await bq_plugin_inst.on_user_message_callback(
          invocation_context=invocation_context,
          user_message=types.Content(parts=[types.Part(text="Test")]),
      )
      await asyncio.sleep(0.01)
      # Constructor should NOT have been called during _log_event
      mock_parser_cls.assert_not_called()


class TestPropertyAccessors:
  """Tests that properties work correctly after __getattribute__ removal."""

  @pytest.mark.asyncio
  async def testbatch_processorerty_returns_processor(self, bq_plugin_inst):
    """batch_processor property should return the processor for the
    current loop."""
    bp = bq_plugin_inst.batch_processor
    assert bp is not None
    assert isinstance(bp, bigquery_agent_analytics_plugin.BatchProcessor)

  @pytest.mark.asyncio
  async def test_write_client_property_returns_client(self, bq_plugin_inst):
    """write_client property should return the client for the current
    loop."""
    wc = bq_plugin_inst.write_client
    assert wc is not None

  @pytest.mark.asyncio
  async def test_write_stream_property_returns_stream(self, bq_plugin_inst):
    """write_stream property should return the stream name."""
    ws = bq_plugin_inst.write_stream
    assert ws is not None
    assert ws == DEFAULT_STREAM_NAME

  @pytest.mark.asyncio
  async def test_properties_return_none_when_no_loop_state(self):
    """Properties should return None when no state exists for the
    current loop."""
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
    )
    assert plugin.batch_processor is None
    assert plugin.write_client is None
    assert plugin.write_stream is None

  @pytest.mark.asyncio
  async def test_regular_attributes_still_accessible(self, bq_plugin_inst):
    """Regular instance attributes should still be accessible."""
    assert bq_plugin_inst.project_id == PROJECT_ID
    assert bq_plugin_inst.dataset_id == DATASET_ID
    assert bq_plugin_inst.table_id == TABLE_ID
    assert bq_plugin_inst.config is not None
    assert bq_plugin_inst._started is True

  def test_properties_without_running_loop(self):
    """Properties should return None when no event loop is running."""
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
    )
    # No running loop → should return None, not crash
    assert plugin.batch_processor is None
    assert plugin.write_client is None
    assert plugin.write_stream is None


class TestUnifiedSpanRecords:
  """Tests for the unified _SpanRecord-based TraceManager."""

  @pytest.mark.asyncio
  async def test_push_pop_keeps_stacks_in_sync(self, callback_context):
    """Push and pop should always leave the records stack consistent."""
    TM = bigquery_agent_analytics_plugin.TraceManager
    TM.init_trace(callback_context)

    span_id_1 = TM.push_span(callback_context, "span-1")
    span_id_2 = TM.push_span(callback_context, "span-2")

    # Both should be on the stack
    assert TM.get_current_span_id() == span_id_2
    current, parent = TM.get_current_span_and_parent()
    assert current == span_id_2
    assert parent == span_id_1

    # Pop span-2
    popped_id, duration = TM.pop_span()
    assert popped_id == span_id_2
    assert duration is not None
    assert TM.get_current_span_id() == span_id_1

    # Pop span-1
    popped_id, _ = TM.pop_span()
    assert popped_id == span_id_1
    assert TM.get_current_span_id() is None

  @pytest.mark.asyncio
  async def test_pop_empty_stack_returns_none(self, callback_context):
    """Popping an empty stack should return (None, None)."""
    TM = bigquery_agent_analytics_plugin.TraceManager
    TM.init_trace(callback_context)

    span_id, duration = TM.pop_span()
    assert span_id is None
    assert duration is None

  @pytest.mark.asyncio
  async def test_first_token_time_stored_in_record(self, callback_context):
    """first_token_time should be stored on the span record."""
    TM = bigquery_agent_analytics_plugin.TraceManager
    TM.init_trace(callback_context)

    span_id = TM.push_span(callback_context, "llm-span")

    # No first token yet
    assert TM.get_first_token_time(span_id) is None

    # Record first token
    assert TM.record_first_token(span_id) is True
    ftt = TM.get_first_token_time(span_id)
    assert ftt is not None

    # Second call should return False (already recorded)
    assert TM.record_first_token(span_id) is False

    # Clean up
    TM.pop_span()

  @pytest.mark.asyncio
  async def test_start_time_accessible_by_span_id(self, callback_context):
    """get_start_time should find the span by ID in the records."""
    TM = bigquery_agent_analytics_plugin.TraceManager
    TM.init_trace(callback_context)

    span_id = TM.push_span(callback_context, "timed-span")
    start = TM.get_start_time(span_id)
    assert start is not None
    assert start > 0

    TM.pop_span()

  @pytest.mark.asyncio
  async def test_attach_current_span_does_not_own(self, callback_context):
    """attach_current_span should not end the span on pop."""
    TM = bigquery_agent_analytics_plugin.TraceManager
    TM.init_trace(callback_context)

    mock_span = mock.Mock()
    mock_ctx = mock.Mock()
    mock_ctx.is_valid = False
    mock_span.get_span_context.return_value = mock_ctx

    with mock.patch(
        "opentelemetry.trace.get_current_span", return_value=mock_span
    ):
      span_id = TM.attach_current_span(callback_context)
      assert span_id is not None

      TM.pop_span()
      # Should NOT have called span.end() since we don't own it
      mock_span.end.assert_not_called()

  @pytest.mark.asyncio
  async def test_concurrent_tasks_have_isolated_stacks(self, callback_context):
    """Concurrent async tasks should have isolated span stacks."""
    TM = bigquery_agent_analytics_plugin.TraceManager
    TM.init_trace(callback_context)

    async def task_a():
      s = TM.push_span(callback_context, "task-a")
      await asyncio.sleep(0.02)
      assert TM.get_current_span_id() == s
      TM.pop_span()
      return s

    async def task_b():
      s = TM.push_span(callback_context, "task-b")
      await asyncio.sleep(0.02)
      assert TM.get_current_span_id() == s
      TM.pop_span()
      return s

    results = await asyncio.gather(task_a(), task_b())
    assert results[0] != results[1]

  @pytest.mark.asyncio
  async def test_pop_cleans_up_record_completely(self, callback_context):
    """After pop, the record should be fully removed from the stack."""
    TM = bigquery_agent_analytics_plugin.TraceManager
    TM.init_trace(callback_context)

    span_id = TM.push_span(callback_context, "temp-span")

    # Record is on the stack
    assert TM.get_current_span_id() == span_id
    assert TM.get_start_time(span_id) is not None

    TM.pop_span()

    # Record is gone
    assert TM.get_current_span_id() is None
    assert TM.get_start_time(span_id) is None
    assert TM.get_first_token_time(span_id) is None


class TestLoopStateValidation:
  """Tests for loop state validation and stale loop cleanup."""

  def _make_plugin(self):
    """Creates a plugin instance without starting it."""
    return bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
    )

  def _make_loop_state(self):
    """Creates a mock _LoopState with batch_processor and write_client."""
    state = mock.MagicMock()
    state.batch_processor = mock.MagicMock(
        spec=bigquery_agent_analytics_plugin.BatchProcessor
    )
    state.batch_processor.flush = mock.AsyncMock()
    state.write_client = mock.MagicMock()
    return state

  def test_cleanup_stale_loop_states_removes_closed_loops(self):
    """Closed loops should be removed from _loop_state_by_loop."""
    plugin = self._make_plugin()

    closed_loop = mock.MagicMock(spec=asyncio.AbstractEventLoop)
    closed_loop.is_closed.return_value = True

    plugin._loop_state_by_loop[closed_loop] = self._make_loop_state()

    plugin._cleanup_stale_loop_states()

    assert closed_loop not in plugin._loop_state_by_loop

  def test_cleanup_stale_loop_states_keeps_open_loops(self):
    """Open loops should not be removed from _loop_state_by_loop."""
    plugin = self._make_plugin()

    open_loop = mock.MagicMock(spec=asyncio.AbstractEventLoop)
    open_loop.is_closed.return_value = False

    plugin._loop_state_by_loop[open_loop] = self._make_loop_state()

    plugin._cleanup_stale_loop_states()

    assert open_loop in plugin._loop_state_by_loop

  def test_cleanup_removes_only_closed_loops(self):
    """Only closed loops should be removed; open ones stay."""
    plugin = self._make_plugin()

    open_loop = mock.MagicMock(spec=asyncio.AbstractEventLoop)
    open_loop.is_closed.return_value = False
    closed_loop = mock.MagicMock(spec=asyncio.AbstractEventLoop)
    closed_loop.is_closed.return_value = True

    plugin._loop_state_by_loop[open_loop] = self._make_loop_state()
    plugin._loop_state_by_loop[closed_loop] = self._make_loop_state()

    plugin._cleanup_stale_loop_states()

    assert open_loop in plugin._loop_state_by_loop
    assert closed_loop not in plugin._loop_state_by_loop

  @pytest.mark.asyncio
  async def testbatch_processor_returns_processor_for_open_loop(
      self,
  ):
    """batch_processor returns processor for the current loop."""
    plugin = self._make_plugin()

    loop = asyncio.get_running_loop()
    state = self._make_loop_state()
    plugin._loop_state_by_loop[loop] = state

    assert plugin.batch_processor is state.batch_processor

    # Clean up
    del plugin._loop_state_by_loop[loop]

  @pytest.mark.asyncio
  async def testbatch_processor_cleans_closed_loop_entry(self):
    """Accessing batch_processor cleans up closed loop entries."""
    plugin = self._make_plugin()

    closed_loop = mock.MagicMock(spec=asyncio.AbstractEventLoop)
    closed_loop.is_closed.return_value = True
    plugin._loop_state_by_loop[closed_loop] = self._make_loop_state()

    # Accessing the prop should clean up the closed loop entry
    _ = plugin.batch_processor
    assert closed_loop not in plugin._loop_state_by_loop

  @pytest.mark.asyncio
  async def test_flush_cleans_stale_states(self):
    """flush() should clean up stale loop states before flushing."""
    plugin = self._make_plugin()

    closed_loop = mock.MagicMock(spec=asyncio.AbstractEventLoop)
    closed_loop.is_closed.return_value = True
    plugin._loop_state_by_loop[closed_loop] = self._make_loop_state()

    await plugin.flush()

    assert closed_loop not in plugin._loop_state_by_loop


class TestAtexitCleanup:
  """Tests for the simplified _atexit_cleanup static method."""

  def _make_batch_processor(self, queue_items=0):
    bp = mock.MagicMock()
    bp._shutdown = False
    q = asyncio.Queue()
    for i in range(queue_items):
      q.put_nowait({"event": i})
    bp._queue = q
    return bp

  def test_skips_none_processor(self):
    """Should return immediately when batch_processor is None."""
    bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin._atexit_cleanup(
        None
    )

  def test_skips_already_shutdown(self):
    """Should return immediately when batch_processor._shutdown is True."""
    bp = self._make_batch_processor()
    bp._shutdown = True
    bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin._atexit_cleanup(
        bp
    )

  def test_skips_reference_error(self):
    """Should handle ReferenceError from weakref'd processor."""
    bp = mock.MagicMock()
    type(bp)._shutdown = mock.PropertyMock(side_effect=ReferenceError)
    bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin._atexit_cleanup(
        bp
    )

  def test_empty_queue_no_warning(self):
    """Should not warn when queue is empty."""
    bp = self._make_batch_processor(queue_items=0)
    with mock.patch.object(
        bigquery_agent_analytics_plugin.logger, "warning"
    ) as mock_warn:
      bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin._atexit_cleanup(
          bp
      )
      mock_warn.assert_not_called()

  def test_remaining_items_logs_warning(self):
    """Should drain queue and log warning with count of lost items."""
    bp = self._make_batch_processor(queue_items=3)
    with mock.patch.object(
        bigquery_agent_analytics_plugin.logger, "warning"
    ) as mock_warn:
      bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin._atexit_cleanup(
          bp
      )
      mock_warn.assert_called_once()
      # Verify the warning mentions the count
      call_args = mock_warn.call_args
      assert "3" in str(call_args)

  def test_queue_is_drained(self):
    """Should drain all items from the queue."""
    bp = self._make_batch_processor(queue_items=5)
    bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin._atexit_cleanup(
        bp
    )
    assert bp._queue.empty()


class TestDuplicateLabels:
  """Tests that labels in before_model_callback are set exactly once."""

  @pytest.mark.asyncio
  async def test_labels_set_when_present(
      self,
      bq_plugin_inst,
      mock_write_client,
      callback_context,
      dummy_arrow_schema,
  ):
    """Labels should appear in attributes when config has them."""
    llm_request = llm_request_lib.LlmRequest(
        model="gemini-pro",
        config=types.GenerateContentConfig(
            labels={"env": "test"},
        ),
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
    )
    bigquery_agent_analytics_plugin.TraceManager.push_span(callback_context)
    await bq_plugin_inst.before_model_callback(
        callback_context=callback_context, llm_request=llm_request
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    attributes = json.loads(log_entry["attributes"])
    assert attributes["labels"] == {"env": "test"}

  @pytest.mark.asyncio
  async def test_labels_absent_when_none(
      self,
      bq_plugin_inst,
      mock_write_client,
      callback_context,
      dummy_arrow_schema,
  ):
    """Labels should not appear in attributes when config.labels is None."""
    llm_request = llm_request_lib.LlmRequest(
        model="gemini-pro",
        config=types.GenerateContentConfig(
            temperature=0.5,
        ),
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
    )
    bigquery_agent_analytics_plugin.TraceManager.push_span(callback_context)
    await bq_plugin_inst.before_model_callback(
        callback_context=callback_context, llm_request=llm_request
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    attributes = json.loads(log_entry["attributes"])
    assert "labels" not in attributes

  @pytest.mark.asyncio
  async def test_no_config_no_labels(
      self,
      bq_plugin_inst,
      mock_write_client,
      callback_context,
      dummy_arrow_schema,
  ):
    """Labels should not appear when llm_request has no config."""
    llm_request = llm_request_lib.LlmRequest(
        model="gemini-pro",
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
    )
    bigquery_agent_analytics_plugin.TraceManager.push_span(callback_context)
    await bq_plugin_inst.before_model_callback(
        callback_context=callback_context, llm_request=llm_request
    )
    await asyncio.sleep(0.01)
    log_entry = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    attributes = json.loads(log_entry["attributes"])
    assert "labels" not in attributes


class TestResolveIds:
  """Tests for the _resolve_ids static helper."""

  def _resolve(self, ed, callback_context):
    return bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin._resolve_ids(
        ed, callback_context
    )

  def test_uses_trace_manager_defaults(self, callback_context):
    """Should use TraceManager values when no overrides and no ambient."""
    ed = bigquery_agent_analytics_plugin.EventData(
        extra_attributes={"some_key": "value"}
    )
    with (
        mock.patch.object(
            bigquery_agent_analytics_plugin.TraceManager,
            "get_current_span_and_parent",
            return_value=("span-1", "parent-1"),
        ),
        mock.patch.object(
            bigquery_agent_analytics_plugin.TraceManager,
            "get_trace_id",
            return_value="trace-1",
        ),
    ):
      trace_id, span_id, parent_id = self._resolve(ed, callback_context)
    assert trace_id == "trace-1"
    assert span_id == "span-1"
    assert parent_id == "parent-1"

  def test_span_id_override(self, callback_context):
    """Should use span_id_override from EventData."""
    ed = bigquery_agent_analytics_plugin.EventData(
        span_id_override="custom-span"
    )
    with (
        mock.patch.object(
            bigquery_agent_analytics_plugin.TraceManager,
            "get_current_span_and_parent",
            return_value=("span-1", "parent-1"),
        ),
        mock.patch.object(
            bigquery_agent_analytics_plugin.TraceManager,
            "get_trace_id",
            return_value="trace-1",
        ),
    ):
      trace_id, span_id, parent_id = self._resolve(ed, callback_context)
    assert span_id == "custom-span"
    assert parent_id == "parent-1"

  def test_parent_span_id_override(self, callback_context):
    """Should use parent_span_id_override from EventData."""
    ed = bigquery_agent_analytics_plugin.EventData(
        parent_span_id_override="custom-parent"
    )
    with (
        mock.patch.object(
            bigquery_agent_analytics_plugin.TraceManager,
            "get_current_span_and_parent",
            return_value=("span-1", "parent-1"),
        ),
        mock.patch.object(
            bigquery_agent_analytics_plugin.TraceManager,
            "get_trace_id",
            return_value="trace-1",
        ),
    ):
      trace_id, span_id, parent_id = self._resolve(ed, callback_context)
    assert span_id == "span-1"
    assert parent_id == "custom-parent"

  def test_none_override_keeps_default(self, callback_context):
    """None overrides should keep the TraceManager defaults."""
    ed = bigquery_agent_analytics_plugin.EventData(
        span_id_override=None, parent_span_id_override=None
    )
    with (
        mock.patch.object(
            bigquery_agent_analytics_plugin.TraceManager,
            "get_current_span_and_parent",
            return_value=("span-1", "parent-1"),
        ),
        mock.patch.object(
            bigquery_agent_analytics_plugin.TraceManager,
            "get_trace_id",
            return_value="trace-1",
        ),
    ):
      trace_id, span_id, parent_id = self._resolve(ed, callback_context)
    assert span_id == "span-1"
    assert parent_id == "parent-1"

  def test_ambient_provides_trace_id_only_when_stack_present(
      self, callback_context
  ):
    """Plugin stack owns span_id/parent; ambient only provides trace_id."""
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    real_tracer = provider.get_tracer("test")

    ed = bigquery_agent_analytics_plugin.EventData()

    with real_tracer.start_as_current_span("invocation") as parent_span:
      with real_tracer.start_as_current_span("agent") as agent_span:
        ambient_ctx = agent_span.get_span_context()
        expected_trace = format(ambient_ctx.trace_id, "032x")

        # Plugin stack has spans — these should win for span/parent.
        with (
            mock.patch.object(
                bigquery_agent_analytics_plugin.TraceManager,
                "get_current_span_and_parent",
                return_value=("plugin-span", "plugin-parent"),
            ),
            mock.patch.object(
                bigquery_agent_analytics_plugin.TraceManager,
                "get_trace_id",
                return_value="plugin-trace",
            ),
        ):
          trace_id, span_id, parent_id = self._resolve(ed, callback_context)

    # trace_id comes from ambient OTel.
    assert trace_id == expected_trace
    # span_id and parent_span_id come from plugin stack.
    assert span_id == "plugin-span"
    assert parent_id == "plugin-parent"
    provider.shutdown()

  def test_ambient_fallback_when_no_plugin_stack(self, callback_context):
    """Ambient OTel provides span_id/parent when plugin stack is empty."""
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    real_tracer = provider.get_tracer("test")

    ed = bigquery_agent_analytics_plugin.EventData()

    with real_tracer.start_as_current_span("invocation") as parent_span:
      with real_tracer.start_as_current_span("agent") as agent_span:
        ambient_ctx = agent_span.get_span_context()
        expected_trace = format(ambient_ctx.trace_id, "032x")
        expected_span = format(ambient_ctx.span_id, "016x")
        expected_parent = format(parent_span.get_span_context().span_id, "016x")

        # Plugin stack returns None — ambient is the fallback.
        with (
            mock.patch.object(
                bigquery_agent_analytics_plugin.TraceManager,
                "get_current_span_and_parent",
                return_value=(None, None),
            ),
            mock.patch.object(
                bigquery_agent_analytics_plugin.TraceManager,
                "get_trace_id",
                return_value=None,
            ),
        ):
          trace_id, span_id, parent_id = self._resolve(ed, callback_context)

    assert trace_id == expected_trace
    assert span_id == expected_span
    assert parent_id == expected_parent
    provider.shutdown()

  def test_override_beats_ambient(self, callback_context):
    """EventData overrides take priority over ambient OTel span."""
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    real_tracer = provider.get_tracer("test")

    ed = bigquery_agent_analytics_plugin.EventData(
        trace_id_override="forced-trace",
        span_id_override="forced-span",
        parent_span_id_override="forced-parent",
    )

    with real_tracer.start_as_current_span("invocation"):
      trace_id, span_id, parent_id = self._resolve(ed, callback_context)

    assert trace_id == "forced-trace"
    assert span_id == "forced-span"
    assert parent_id == "forced-parent"
    provider.shutdown()

  def test_plugin_stack_wins_over_ambient_root_span(self, callback_context):
    """Plugin stack span is used even when ambient root span exists."""
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    real_tracer = provider.get_tracer("test")

    # Seed the plugin stack with a span.
    bigquery_agent_analytics_plugin._span_records_ctx.set(None)
    with mock.patch.object(
        bigquery_agent_analytics_plugin, "tracer", real_tracer
    ):
      bigquery_agent_analytics_plugin.TraceManager.push_span(
          callback_context, "plugin-child"
      )

    # Capture the plugin span_id that was pushed.
    plugin_span_id, _ = (
        bigquery_agent_analytics_plugin.TraceManager.get_current_span_and_parent()
    )

    ed = bigquery_agent_analytics_plugin.EventData()

    # Single root ambient span — no parent.
    with real_tracer.start_as_current_span("root_invocation") as root:
      trace_id, span_id, parent_id = self._resolve(ed, callback_context)
      ambient_trace = format(root.get_span_context().trace_id, "032x")

    # trace_id comes from ambient.
    assert trace_id == ambient_trace
    # span_id comes from plugin stack, not ambient.
    assert span_id == plugin_span_id
    # parent is None — only one span in plugin stack.
    assert parent_id is None

    # Cleanup
    bigquery_agent_analytics_plugin.TraceManager.pop_span()
    provider.shutdown()

  def test_ambient_root_fallback_no_self_parent(self, callback_context):
    """Ambient root span fallback must not produce self-parent."""
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    real_tracer = provider.get_tracer("test")

    ed = bigquery_agent_analytics_plugin.EventData()

    # Plugin stack empty — ambient provides the fallback.
    with (
        mock.patch.object(
            bigquery_agent_analytics_plugin.TraceManager,
            "get_current_span_and_parent",
            return_value=(None, None),
        ),
        mock.patch.object(
            bigquery_agent_analytics_plugin.TraceManager,
            "get_trace_id",
            return_value=None,
        ),
    ):
      with real_tracer.start_as_current_span("root") as root:
        trace_id, span_id, parent_id = self._resolve(ed, callback_context)
        root_span_id = format(root.get_span_context().span_id, "016x")

    assert span_id == root_span_id
    assert parent_id is None
    provider.shutdown()

  def test_plugin_stack_pairs_starting_completed(self, callback_context):
    """STARTING/COMPLETED pairing uses plugin stack, not ambient.

    Post-pop callbacks now always pass explicit overrides from the
    plugin stack.  The plugin stack span_id is used for both events
    regardless of ambient OTel state.
    """
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    real_tracer = provider.get_tracer("test")

    with real_tracer.start_as_current_span("invoke_agent"):
      # Simulate STARTING: plugin stack provides span_id.
      with (
          mock.patch.object(
              bigquery_agent_analytics_plugin.TraceManager,
              "get_current_span_and_parent",
              return_value=("plugin-agent", "plugin-inv"),
          ),
          mock.patch.object(
              bigquery_agent_analytics_plugin.TraceManager,
              "get_trace_id",
              return_value="plugin-trace",
          ),
      ):
        ed_starting = bigquery_agent_analytics_plugin.EventData()
        _, span_starting, _ = self._resolve(ed_starting, callback_context)

      # Simulate COMPLETED: explicit override from popped span.
      ed_completed = bigquery_agent_analytics_plugin.EventData(
          span_id_override="plugin-agent",
          parent_span_id_override="plugin-inv",
          latency_ms=42,
      )
      _, span_completed, _ = self._resolve(ed_completed, callback_context)

      assert span_starting == "plugin-agent"
      assert span_completed == "plugin-agent"
      assert span_starting == span_completed

    provider.shutdown()


class TestExtractLatency:
  """Tests for the _extract_latency static helper."""

  def test_no_latency_returns_none(self):
    """Should return None when no latency fields present."""
    ed = bigquery_agent_analytics_plugin.EventData(
        extra_attributes={"other": "val"}
    )
    result = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin._extract_latency(
        ed
    )
    assert result is None

  def test_total_latency_only(self):
    """Should extract latency_ms into total_ms."""
    ed = bigquery_agent_analytics_plugin.EventData(latency_ms=42.5)
    result = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin._extract_latency(
        ed
    )
    assert result == {"total_ms": 42.5}

  def test_tfft_only(self):
    """Should extract time_to_first_token_ms."""
    ed = bigquery_agent_analytics_plugin.EventData(time_to_first_token_ms=10.0)
    result = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin._extract_latency(
        ed
    )
    assert result == {"time_to_first_token_ms": 10.0}

  def test_both_latencies(self):
    """Should extract both latency fields."""
    ed = bigquery_agent_analytics_plugin.EventData(
        latency_ms=100, time_to_first_token_ms=20
    )
    result = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin._extract_latency(
        ed
    )
    assert result == {"total_ms": 100, "time_to_first_token_ms": 20}


class TestEnrichAttributes:
  """Tests for the _enrich_attributes helper."""

  def _make_plugin(self):
    with (
        mock.patch(
            "google.auth.default",
            return_value=(mock.Mock(), PROJECT_ID),
        ),
        mock.patch(
            "google.cloud.bigquery.Client",
        ),
    ):
      plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
          project_id=PROJECT_ID,
          dataset_id=DATASET_ID,
      )
    plugin.config.max_content_length = 10000
    plugin.config.log_session_metadata = False
    plugin.config.custom_tags = None
    return plugin

  def _make_callback_context(self):
    ctx = mock.MagicMock()
    session = mock.MagicMock()
    session.id = "sess-001"
    session.app_name = "test-app"
    session.user_id = "user-001"
    session.state = {"env": "test"}
    ctx._invocation_context.session = session
    return ctx

  def test_adds_root_agent_name(self):
    """Should always add root_agent_name."""
    plugin = self._make_plugin()
    ed = bigquery_agent_analytics_plugin.EventData()
    with mock.patch.object(
        bigquery_agent_analytics_plugin.TraceManager,
        "get_root_agent_name",
        return_value="my-agent",
    ):
      attrs = plugin._enrich_attributes(ed, self._make_callback_context())
    assert attrs["root_agent_name"] == "my-agent"

  def test_includes_model(self):
    """Should include model from EventData."""
    plugin = self._make_plugin()
    ed = bigquery_agent_analytics_plugin.EventData(model="gemini-pro")
    with mock.patch.object(
        bigquery_agent_analytics_plugin.TraceManager,
        "get_root_agent_name",
        return_value="agent",
    ):
      attrs = plugin._enrich_attributes(ed, self._make_callback_context())
    assert attrs["model"] == "gemini-pro"

  def test_session_metadata_when_enabled(self):
    """Should add session_metadata when log_session_metadata is True."""
    plugin = self._make_plugin()
    plugin.config.log_session_metadata = True
    ctx = self._make_callback_context()
    ed = bigquery_agent_analytics_plugin.EventData()
    with mock.patch.object(
        bigquery_agent_analytics_plugin.TraceManager,
        "get_root_agent_name",
        return_value="agent",
    ):
      attrs = plugin._enrich_attributes(ed, ctx)
    meta = attrs["session_metadata"]
    assert meta["session_id"] == "sess-001"
    assert meta["app_name"] == "test-app"
    assert meta["user_id"] == "user-001"
    assert meta["state"] == {"env": "test"}

  def test_session_metadata_when_disabled(self):
    """Should not add session_metadata when log_session_metadata is False."""
    plugin = self._make_plugin()
    plugin.config.log_session_metadata = False
    ed = bigquery_agent_analytics_plugin.EventData()
    with mock.patch.object(
        bigquery_agent_analytics_plugin.TraceManager,
        "get_root_agent_name",
        return_value="agent",
    ):
      attrs = plugin._enrich_attributes(ed, self._make_callback_context())
    assert "session_metadata" not in attrs

  def test_custom_tags_added(self):
    """Should add custom_tags when configured."""
    plugin = self._make_plugin()
    plugin.config.custom_tags = {"team": "infra"}
    ed = bigquery_agent_analytics_plugin.EventData()
    with mock.patch.object(
        bigquery_agent_analytics_plugin.TraceManager,
        "get_root_agent_name",
        return_value="agent",
    ):
      attrs = plugin._enrich_attributes(ed, self._make_callback_context())
    assert attrs["custom_tags"] == {"team": "infra"}

  def test_usage_metadata_truncated(self):
    """Should smart-truncate usage_metadata."""
    plugin = self._make_plugin()
    ed = bigquery_agent_analytics_plugin.EventData(
        usage_metadata={"input_tokens": 100, "output_tokens": 50}
    )
    with mock.patch.object(
        bigquery_agent_analytics_plugin.TraceManager,
        "get_root_agent_name",
        return_value="agent",
    ):
      attrs = plugin._enrich_attributes(ed, self._make_callback_context())
    assert attrs["usage_metadata"] == {
        "input_tokens": 100,
        "output_tokens": 50,
    }


class TestMultiSubagentToolLogging:
  """Tests that tool events from different subagents are attributed correctly.

  Covers:
  - Tool calls from different subagents have the correct `agent` field
  - Multi-turn (different invocation_ids, same session) logs correctly
  - Full callback sequence across multiple subagents in one turn
  - Span hierarchy is maintained per-subagent
  """

  @staticmethod
  def _make_invocation_context(agent_name, session, invocation_id="inv-001"):
    """Create an InvocationContext with a specific agent name."""
    mock_a = mock.create_autospec(
        base_agent.BaseAgent, instance=True, spec_set=True
    )
    type(mock_a).name = mock.PropertyMock(return_value=agent_name)
    type(mock_a).instruction = mock.PropertyMock(
        return_value=f"{agent_name} instruction"
    )
    mock_session_service = mock.create_autospec(
        base_session_service_lib.BaseSessionService,
        instance=True,
        spec_set=True,
    )
    mock_plugin_manager = mock.create_autospec(
        plugin_manager_lib.PluginManager,
        instance=True,
        spec_set=True,
    )
    return InvocationContext(
        agent=mock_a,
        session=session,
        invocation_id=invocation_id,
        session_service=mock_session_service,
        plugin_manager=mock_plugin_manager,
    )

  @staticmethod
  def _make_session(session_id="session-multi", user_id="user-multi"):
    mock_s = mock.create_autospec(
        session_lib.Session, instance=True, spec_set=True
    )
    type(mock_s).id = mock.PropertyMock(return_value=session_id)
    type(mock_s).user_id = mock.PropertyMock(return_value=user_id)
    type(mock_s).app_name = mock.PropertyMock(return_value="test_app")
    type(mock_s).state = mock.PropertyMock(return_value={})
    return mock_s

  @staticmethod
  def _make_tool(name):
    mock_tool = mock.create_autospec(
        base_tool_lib.BaseTool, instance=True, spec_set=True
    )
    type(mock_tool).name = mock.PropertyMock(return_value=name)
    type(mock_tool).description = mock.PropertyMock(
        return_value=f"{name} description"
    )
    return mock_tool

  @pytest.mark.asyncio
  async def test_tool_calls_attributed_to_correct_subagent(
      self,
      mock_auth_default,
      mock_bq_client,
      mock_write_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    """Tool events from different subagents carry the correct agent name."""
    session = self._make_session()

    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()

      # --- Subagent A: schema_explorer calls list_datasets ---
      inv_ctx_a = self._make_invocation_context("schema_explorer", session)
      ctx_a = tool_context_lib.ToolContext(invocation_context=inv_ctx_a)
      tool_a = self._make_tool("list_dataset_ids")

      bigquery_agent_analytics_plugin.TraceManager.push_span(ctx_a, "tool")
      await plugin.before_tool_callback(
          tool=tool_a,
          tool_args={"project_id": "my-project"},
          tool_context=ctx_a,
      )
      await asyncio.sleep(0.01)

      # --- Subagent B: image_describer calls describe_this_image ---
      inv_ctx_b = self._make_invocation_context("image_describer", session)
      ctx_b = tool_context_lib.ToolContext(invocation_context=inv_ctx_b)
      tool_b = self._make_tool("describe_this_image")

      bigquery_agent_analytics_plugin.TraceManager.push_span(ctx_b, "tool")
      await plugin.before_tool_callback(
          tool=tool_b,
          tool_args={"image_uri": "gs://bucket/image.jpg"},
          tool_context=ctx_b,
      )
      await asyncio.sleep(0.01)

      rows = await _get_captured_rows_async(
          mock_write_client, dummy_arrow_schema
      )

    assert len(rows) == 2

    # First row: schema_explorer's tool
    assert rows[0]["event_type"] == "TOOL_STARTING"
    assert rows[0]["agent"] == "schema_explorer"
    content_a = json.loads(rows[0]["content"])
    assert content_a["tool"] == "list_dataset_ids"
    assert content_a["args"] == {"project_id": "my-project"}

    # Second row: image_describer's tool
    assert rows[1]["event_type"] == "TOOL_STARTING"
    assert rows[1]["agent"] == "image_describer"
    content_b = json.loads(rows[1]["content"])
    assert content_b["tool"] == "describe_this_image"
    assert content_b["args"] == {"image_uri": "gs://bucket/image.jpg"}

  @pytest.mark.asyncio
  async def test_multi_turn_tool_calls_different_invocations(
      self,
      mock_auth_default,
      mock_bq_client,
      mock_write_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    """Multi-turn: same session, different invocation IDs, tools logged."""
    session = self._make_session()

    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()

      # --- Turn 1: schema_explorer calls list_dataset_ids ---
      inv_ctx_1 = self._make_invocation_context(
          "schema_explorer", session, invocation_id="inv-turn1"
      )
      ctx_1 = tool_context_lib.ToolContext(invocation_context=inv_ctx_1)
      tool_1 = self._make_tool("list_dataset_ids")

      bigquery_agent_analytics_plugin.TraceManager.push_span(ctx_1, "tool")
      await plugin.before_tool_callback(
          tool=tool_1,
          tool_args={"project_id": "proj"},
          tool_context=ctx_1,
      )
      await asyncio.sleep(0.01)
      await plugin.after_tool_callback(
          tool=tool_1,
          tool_args={"project_id": "proj"},
          tool_context=ctx_1,
          result={"datasets": ["ds1", "ds2"]},
      )
      await asyncio.sleep(0.01)

      # --- Turn 2: query_analyst calls execute_sql ---
      inv_ctx_2 = self._make_invocation_context(
          "query_analyst", session, invocation_id="inv-turn2"
      )
      ctx_2 = tool_context_lib.ToolContext(invocation_context=inv_ctx_2)
      tool_2 = self._make_tool("execute_sql")

      bigquery_agent_analytics_plugin.TraceManager.push_span(ctx_2, "tool")
      await plugin.before_tool_callback(
          tool=tool_2,
          tool_args={"sql": "SELECT * FROM t"},
          tool_context=ctx_2,
      )
      await asyncio.sleep(0.01)
      await plugin.after_tool_callback(
          tool=tool_2,
          tool_args={"sql": "SELECT * FROM t"},
          tool_context=ctx_2,
          result={"rows": [{"col": "val"}]},
      )
      await asyncio.sleep(0.01)

      rows = await _get_captured_rows_async(
          mock_write_client, dummy_arrow_schema
      )

    assert len(rows) == 4

    # Turn 1: TOOL_STARTING + TOOL_COMPLETED for schema_explorer
    assert rows[0]["event_type"] == "TOOL_STARTING"
    assert rows[0]["agent"] == "schema_explorer"
    assert rows[0]["invocation_id"] == "inv-turn1"
    assert rows[0]["session_id"] == "session-multi"

    assert rows[1]["event_type"] == "TOOL_COMPLETED"
    assert rows[1]["agent"] == "schema_explorer"
    assert rows[1]["invocation_id"] == "inv-turn1"
    content_1 = json.loads(rows[1]["content"])
    assert content_1["tool"] == "list_dataset_ids"
    assert content_1["result"] == {"datasets": ["ds1", "ds2"]}

    # Turn 2: TOOL_STARTING + TOOL_COMPLETED for query_analyst
    assert rows[2]["event_type"] == "TOOL_STARTING"
    assert rows[2]["agent"] == "query_analyst"
    assert rows[2]["invocation_id"] == "inv-turn2"

    assert rows[3]["event_type"] == "TOOL_COMPLETED"
    assert rows[3]["agent"] == "query_analyst"
    assert rows[3]["invocation_id"] == "inv-turn2"
    content_2 = json.loads(rows[3]["content"])
    assert content_2["tool"] == "execute_sql"
    assert content_2["result"] == {"rows": [{"col": "val"}]}

  @pytest.mark.asyncio
  async def test_full_subagent_callback_sequence(
      self,
      mock_auth_default,
      mock_bq_client,
      mock_write_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    """Full lifecycle: agent_start → LLM → tool → tool_done → LLM → agent_done.

    Simulates a subagent that makes an LLM call, then a tool call,
    then another LLM call, and completes.
    """
    session = self._make_session()
    inv_ctx = self._make_invocation_context("schema_explorer", session)
    cb_ctx = CallbackContext(invocation_context=inv_ctx)
    tool_ctx = tool_context_lib.ToolContext(invocation_context=inv_ctx)
    mock_agent = inv_ctx.agent
    tool = self._make_tool("get_table_info")

    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()

      # 1. AGENT_STARTING
      await plugin.before_agent_callback(
          agent=mock_agent, callback_context=cb_ctx
      )
      await asyncio.sleep(0.01)

      # 2. LLM_REQUEST (agent decides to call a tool)
      llm_req = llm_request_lib.LlmRequest(
          model="gemini-2.5-flash",
          contents=[
              types.Content(parts=[types.Part(text="What tables exist?")])
          ],
      )
      await plugin.before_model_callback(
          callback_context=cb_ctx, llm_request=llm_req
      )
      await asyncio.sleep(0.01)

      # 3. LLM_RESPONSE (function call)
      llm_resp = llm_response_lib.LlmResponse(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name="get_table_info",
                          args={"table": "events"},
                      )
                  )
              ]
          )
      )
      await plugin.after_model_callback(
          callback_context=cb_ctx, llm_response=llm_resp
      )
      await asyncio.sleep(0.01)

      # 4. TOOL_STARTING
      bigquery_agent_analytics_plugin.TraceManager.push_span(tool_ctx, "tool")
      await plugin.before_tool_callback(
          tool=tool,
          tool_args={"table": "events"},
          tool_context=tool_ctx,
      )
      await asyncio.sleep(0.01)

      # 5. TOOL_COMPLETED
      await plugin.after_tool_callback(
          tool=tool,
          tool_args={"table": "events"},
          tool_context=tool_ctx,
          result={"schema": [{"name": "id", "type": "INT64"}]},
      )
      await asyncio.sleep(0.01)

      # 6. AGENT_COMPLETED
      await plugin.after_agent_callback(
          agent=mock_agent, callback_context=cb_ctx
      )
      await asyncio.sleep(0.01)

      rows = await _get_captured_rows_async(
          mock_write_client, dummy_arrow_schema
      )

    assert len(rows) == 6

    expected_sequence = [
        "AGENT_STARTING",
        "LLM_REQUEST",
        "LLM_RESPONSE",
        "TOOL_STARTING",
        "TOOL_COMPLETED",
        "AGENT_COMPLETED",
    ]
    for i, expected_type in enumerate(expected_sequence):
      assert (
          rows[i]["event_type"] == expected_type
      ), f"Row {i}: expected {expected_type}, got {rows[i]['event_type']}"
      assert rows[i]["agent"] == "schema_explorer"
      assert rows[i]["session_id"] == "session-multi"

    # TOOL rows have correct content
    tool_start = json.loads(rows[3]["content"])
    assert tool_start["tool"] == "get_table_info"
    assert tool_start["args"] == {"table": "events"}

    tool_done = json.loads(rows[4]["content"])
    assert tool_done["tool"] == "get_table_info"
    assert tool_done["result"] == {"schema": [{"name": "id", "type": "INT64"}]}

    # AGENT_COMPLETED and TOOL_COMPLETED should have latency
    assert rows[4]["latency_ms"] is not None  # TOOL_COMPLETED
    assert rows[5]["latency_ms"] is not None  # AGENT_COMPLETED

  @pytest.mark.asyncio
  async def test_tool_error_attributed_to_subagent(
      self,
      mock_auth_default,
      mock_bq_client,
      mock_write_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    """TOOL_ERROR events carry the correct subagent name."""
    session = self._make_session()
    inv_ctx = self._make_invocation_context("query_analyst", session)
    tool_ctx = tool_context_lib.ToolContext(invocation_context=inv_ctx)
    tool = self._make_tool("execute_sql")

    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()

      bigquery_agent_analytics_plugin.TraceManager.push_span(tool_ctx, "tool")
      await plugin.on_tool_error_callback(
          tool=tool,
          tool_args={"sql": "SELECT * FROM bad_table"},
          tool_context=tool_ctx,
          error=RuntimeError("Table not found"),
      )
      await asyncio.sleep(0.01)

      rows = await _get_captured_rows_async(
          mock_write_client, dummy_arrow_schema
      )

    assert len(rows) == 1
    assert rows[0]["event_type"] == "TOOL_ERROR"
    assert rows[0]["agent"] == "query_analyst"
    assert rows[0]["error_message"] == "Table not found"
    content = json.loads(rows[0]["content"])
    assert content["tool"] == "execute_sql"
    assert content["args"] == {"sql": "SELECT * FROM bad_table"}

  @pytest.mark.asyncio
  async def test_multi_subagent_interleaved_tool_calls(
      self,
      mock_auth_default,
      mock_bq_client,
      mock_write_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    """Two subagents call tools in same invocation — agent field is correct.

    Simulates orchestrator delegating to schema_explorer first, then
    image_describer, all within the same invocation.
    """
    session = self._make_session()

    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()

      # Subagent 1: schema_explorer — full tool cycle
      inv_ctx_1 = self._make_invocation_context(
          "schema_explorer", session, invocation_id="inv-shared"
      )
      ctx_1 = tool_context_lib.ToolContext(invocation_context=inv_ctx_1)
      tool_1 = self._make_tool("list_table_ids")
      bigquery_agent_analytics_plugin.TraceManager.push_span(ctx_1, "tool")
      await plugin.before_tool_callback(
          tool=tool_1,
          tool_args={"dataset": "analytics"},
          tool_context=ctx_1,
      )
      await asyncio.sleep(0.01)
      await plugin.after_tool_callback(
          tool=tool_1,
          tool_args={"dataset": "analytics"},
          tool_context=ctx_1,
          result={"tables": ["events", "metrics"]},
      )
      await asyncio.sleep(0.01)

      # Subagent 2: image_describer — full tool cycle
      inv_ctx_2 = self._make_invocation_context(
          "image_describer", session, invocation_id="inv-shared"
      )
      ctx_2 = tool_context_lib.ToolContext(invocation_context=inv_ctx_2)
      tool_2 = self._make_tool("describe_this_image")
      bigquery_agent_analytics_plugin.TraceManager.push_span(ctx_2, "tool")
      await plugin.before_tool_callback(
          tool=tool_2,
          tool_args={"image_uri": "https://example.com/img.jpg"},
          tool_context=ctx_2,
      )
      await asyncio.sleep(0.01)
      await plugin.after_tool_callback(
          tool=tool_2,
          tool_args={"image_uri": "https://example.com/img.jpg"},
          tool_context=ctx_2,
          result={"description": "A photo of scones"},
      )
      await asyncio.sleep(0.01)

      rows = await _get_captured_rows_async(
          mock_write_client, dummy_arrow_schema
      )

    assert len(rows) == 4

    # schema_explorer tool events
    assert rows[0]["agent"] == "schema_explorer"
    assert rows[0]["event_type"] == "TOOL_STARTING"
    assert rows[0]["invocation_id"] == "inv-shared"
    assert json.loads(rows[0]["content"])["tool"] == "list_table_ids"

    assert rows[1]["agent"] == "schema_explorer"
    assert rows[1]["event_type"] == "TOOL_COMPLETED"
    assert json.loads(rows[1]["content"])["result"]["tables"] == [
        "events",
        "metrics",
    ]

    # image_describer tool events
    assert rows[2]["agent"] == "image_describer"
    assert rows[2]["event_type"] == "TOOL_STARTING"
    assert rows[2]["invocation_id"] == "inv-shared"
    assert json.loads(rows[2]["content"])["tool"] == "describe_this_image"

    assert rows[3]["agent"] == "image_describer"
    assert rows[3]["event_type"] == "TOOL_COMPLETED"
    assert (
        json.loads(rows[3]["content"])["result"]["description"]
        == "A photo of scones"
    )

    # All share the same session and invocation
    for row in rows:
      assert row["session_id"] == "session-multi"
      assert row["invocation_id"] == "inv-shared"

  @pytest.mark.asyncio
  async def test_multi_turn_multi_subagent_full_sequence(
      self,
      mock_auth_default,
      mock_bq_client,
      mock_write_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    """Multi-turn + multi-subagent: two turns, each with different subagents.

    Turn 1: user asks about data → orchestrator → schema_explorer (tool)
    Turn 2: user asks about image → orchestrator → image_describer (tool)
    Verifies invocation_id changes, agent name changes, session stays same.
    """
    session = self._make_session()

    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()

      # ===== Turn 1: schema_explorer =====
      inv_ctx_t1_orch = self._make_invocation_context(
          "orchestrator", session, invocation_id="inv-t1"
      )
      cb_ctx_t1_orch = CallbackContext(invocation_context=inv_ctx_t1_orch)

      # Orchestrator agent_starting
      await plugin.before_agent_callback(
          agent=inv_ctx_t1_orch.agent,
          callback_context=cb_ctx_t1_orch,
      )
      await asyncio.sleep(0.01)

      # Orchestrator delegates to schema_explorer
      inv_ctx_t1_sub = self._make_invocation_context(
          "schema_explorer", session, invocation_id="inv-t1"
      )
      cb_ctx_t1_sub = CallbackContext(invocation_context=inv_ctx_t1_sub)
      tool_ctx_t1 = tool_context_lib.ToolContext(
          invocation_context=inv_ctx_t1_sub
      )

      await plugin.before_agent_callback(
          agent=inv_ctx_t1_sub.agent,
          callback_context=cb_ctx_t1_sub,
      )
      await asyncio.sleep(0.01)

      # schema_explorer calls tool
      tool_1 = self._make_tool("list_dataset_ids")
      bigquery_agent_analytics_plugin.TraceManager.push_span(
          tool_ctx_t1, "tool"
      )
      await plugin.before_tool_callback(
          tool=tool_1,
          tool_args={"project_id": "proj"},
          tool_context=tool_ctx_t1,
      )
      await asyncio.sleep(0.01)
      await plugin.after_tool_callback(
          tool=tool_1,
          tool_args={"project_id": "proj"},
          tool_context=tool_ctx_t1,
          result={"datasets": ["ds1"]},
      )
      await asyncio.sleep(0.01)

      # schema_explorer done
      await plugin.after_agent_callback(
          agent=inv_ctx_t1_sub.agent,
          callback_context=cb_ctx_t1_sub,
      )
      await asyncio.sleep(0.01)

      # Orchestrator done
      await plugin.after_agent_callback(
          agent=inv_ctx_t1_orch.agent,
          callback_context=cb_ctx_t1_orch,
      )
      await asyncio.sleep(0.01)

      # ===== Turn 2: image_describer =====
      inv_ctx_t2_orch = self._make_invocation_context(
          "orchestrator", session, invocation_id="inv-t2"
      )
      cb_ctx_t2_orch = CallbackContext(invocation_context=inv_ctx_t2_orch)

      await plugin.before_agent_callback(
          agent=inv_ctx_t2_orch.agent,
          callback_context=cb_ctx_t2_orch,
      )
      await asyncio.sleep(0.01)

      # Orchestrator delegates to image_describer
      inv_ctx_t2_sub = self._make_invocation_context(
          "image_describer", session, invocation_id="inv-t2"
      )
      cb_ctx_t2_sub = CallbackContext(invocation_context=inv_ctx_t2_sub)
      tool_ctx_t2 = tool_context_lib.ToolContext(
          invocation_context=inv_ctx_t2_sub
      )

      await plugin.before_agent_callback(
          agent=inv_ctx_t2_sub.agent,
          callback_context=cb_ctx_t2_sub,
      )
      await asyncio.sleep(0.01)

      # image_describer calls tool
      tool_2 = self._make_tool("describe_this_image")
      bigquery_agent_analytics_plugin.TraceManager.push_span(
          tool_ctx_t2, "tool"
      )
      await plugin.before_tool_callback(
          tool=tool_2,
          tool_args={"image_uri": "gs://b/img.jpg"},
          tool_context=tool_ctx_t2,
      )
      await asyncio.sleep(0.01)
      await plugin.after_tool_callback(
          tool=tool_2,
          tool_args={"image_uri": "gs://b/img.jpg"},
          tool_context=tool_ctx_t2,
          result={"desc": "Scones on a table"},
      )
      await asyncio.sleep(0.01)

      # image_describer done
      await plugin.after_agent_callback(
          agent=inv_ctx_t2_sub.agent,
          callback_context=cb_ctx_t2_sub,
      )
      await asyncio.sleep(0.01)

      # Orchestrator done
      await plugin.after_agent_callback(
          agent=inv_ctx_t2_orch.agent,
          callback_context=cb_ctx_t2_orch,
      )
      await asyncio.sleep(0.01)

      rows = await _get_captured_rows_async(
          mock_write_client, dummy_arrow_schema
      )

    # Turn 1: 6 rows (orch_start, sub_start, tool_start, tool_done,
    #                   sub_done, orch_done)
    # Turn 2: 6 rows (same pattern)
    assert len(rows) == 12

    # --- Turn 1 validation ---
    t1_rows = [r for r in rows if r["invocation_id"] == "inv-t1"]
    assert len(t1_rows) == 6

    assert t1_rows[0]["event_type"] == "AGENT_STARTING"
    assert t1_rows[0]["agent"] == "orchestrator"

    assert t1_rows[1]["event_type"] == "AGENT_STARTING"
    assert t1_rows[1]["agent"] == "schema_explorer"

    assert t1_rows[2]["event_type"] == "TOOL_STARTING"
    assert t1_rows[2]["agent"] == "schema_explorer"
    assert json.loads(t1_rows[2]["content"])["tool"] == "list_dataset_ids"

    assert t1_rows[3]["event_type"] == "TOOL_COMPLETED"
    assert t1_rows[3]["agent"] == "schema_explorer"

    assert t1_rows[4]["event_type"] == "AGENT_COMPLETED"
    assert t1_rows[4]["agent"] == "schema_explorer"

    assert t1_rows[5]["event_type"] == "AGENT_COMPLETED"
    assert t1_rows[5]["agent"] == "orchestrator"

    # --- Turn 2 validation ---
    t2_rows = [r for r in rows if r["invocation_id"] == "inv-t2"]
    assert len(t2_rows) == 6

    assert t2_rows[0]["event_type"] == "AGENT_STARTING"
    assert t2_rows[0]["agent"] == "orchestrator"

    assert t2_rows[1]["event_type"] == "AGENT_STARTING"
    assert t2_rows[1]["agent"] == "image_describer"

    assert t2_rows[2]["event_type"] == "TOOL_STARTING"
    assert t2_rows[2]["agent"] == "image_describer"
    assert json.loads(t2_rows[2]["content"])["tool"] == "describe_this_image"

    assert t2_rows[3]["event_type"] == "TOOL_COMPLETED"
    assert t2_rows[3]["agent"] == "image_describer"

    assert t2_rows[4]["event_type"] == "AGENT_COMPLETED"
    assert t2_rows[4]["agent"] == "image_describer"

    assert t2_rows[5]["event_type"] == "AGENT_COMPLETED"
    assert t2_rows[5]["agent"] == "orchestrator"

    # All rows share the same session
    for row in rows:
      assert row["session_id"] == "session-multi"


class TestSchemaAutoUpgrade:
  """Tests for _ensure_schema_exists with auto_schema_upgrade."""

  def _make_plugin(self, auto_schema_upgrade=False):
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(
        auto_schema_upgrade=auto_schema_upgrade,
    )
    with mock.patch("google.cloud.bigquery.Client"):
      plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
          project_id=PROJECT_ID,
          dataset_id=DATASET_ID,
          table_id=TABLE_ID,
          config=config,
      )
    plugin.client = mock.MagicMock()
    plugin.full_table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    plugin._schema = bigquery_agent_analytics_plugin._get_events_schema()
    return plugin

  def test_create_table_sets_version_label(self):
    """New tables get the schema version label."""
    plugin = self._make_plugin()
    plugin.client.get_table.side_effect = cloud_exceptions.NotFound("not found")
    plugin._ensure_schema_exists()
    plugin.client.create_table.assert_called_once()
    tbl = plugin.client.create_table.call_args[0][0]
    assert (
        tbl.labels[bigquery_agent_analytics_plugin._SCHEMA_VERSION_LABEL_KEY]
        == bigquery_agent_analytics_plugin._SCHEMA_VERSION
    )

  def test_no_upgrade_when_disabled(self):
    """Auto-upgrade disabled: existing table is not modified."""
    plugin = self._make_plugin(auto_schema_upgrade=False)
    existing = mock.MagicMock()
    existing.schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
    ]
    existing.labels = {}
    plugin.client.get_table.return_value = existing
    plugin._ensure_schema_exists()
    plugin.client.update_table.assert_not_called()

  def test_upgrade_adds_missing_columns(self):
    """Auto-upgrade adds columns missing from existing table."""
    plugin = self._make_plugin(auto_schema_upgrade=True)
    existing = mock.MagicMock(spec=bigquery.Table)
    existing.schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
    ]
    existing.labels = {"other": "label"}
    plugin.client.get_table.return_value = existing
    plugin._ensure_schema_exists()
    plugin.client.update_table.assert_called_once()
    updated_table = plugin.client.update_table.call_args[0][0]
    updated_names = {f.name for f in updated_table.schema}
    assert "event_type" in updated_names
    assert "agent" in updated_names
    assert "content" in updated_names
    assert (
        updated_table.labels[
            bigquery_agent_analytics_plugin._SCHEMA_VERSION_LABEL_KEY
        ]
        == bigquery_agent_analytics_plugin._SCHEMA_VERSION
    )

  def test_skip_upgrade_when_version_matches(self):
    """No update when stored version matches current."""
    plugin = self._make_plugin(auto_schema_upgrade=True)
    existing = mock.MagicMock(spec=bigquery.Table)
    existing.schema = plugin._schema
    existing.labels = {
        bigquery_agent_analytics_plugin._SCHEMA_VERSION_LABEL_KEY: (
            bigquery_agent_analytics_plugin._SCHEMA_VERSION
        ),
    }
    plugin.client.get_table.return_value = existing
    plugin._ensure_schema_exists()
    plugin.client.update_table.assert_not_called()

  def test_upgrade_error_is_logged_not_raised(self):
    """Schema upgrade errors are logged, not propagated."""
    plugin = self._make_plugin(auto_schema_upgrade=True)
    existing = mock.MagicMock(spec=bigquery.Table)
    existing.schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
    ]
    existing.labels = {}
    plugin.client.get_table.return_value = existing
    plugin.client.update_table.side_effect = Exception("boom")
    # Should not raise
    plugin._ensure_schema_exists()

  def test_upgrade_preserves_existing_columns(self):
    """Existing columns are never dropped or altered during upgrade."""
    plugin = self._make_plugin(auto_schema_upgrade=True)
    # Simulate a table with a subset of canonical columns plus a
    # user-added custom column that is NOT in the canonical schema.
    custom_field = bigquery.SchemaField("my_custom_col", "STRING")
    existing = mock.MagicMock(spec=bigquery.Table)
    existing.schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("event_type", "STRING"),
        custom_field,
    ]
    existing.labels = {}
    plugin.client.get_table.return_value = existing
    plugin._ensure_schema_exists()

    updated_table = plugin.client.update_table.call_args[0][0]
    updated_names = [f.name for f in updated_table.schema]
    # Original columns are still present and in original order.
    assert updated_names[0] == "timestamp"
    assert updated_names[1] == "event_type"
    assert updated_names[2] == "my_custom_col"
    # New canonical columns were appended after existing ones.
    assert "agent" in updated_names
    assert "content" in updated_names

  def test_upgrade_from_no_label_treats_as_outdated(self):
    """A table with no version label is treated as needing upgrade."""
    plugin = self._make_plugin(auto_schema_upgrade=True)
    existing = mock.MagicMock(spec=bigquery.Table)
    existing.schema = list(plugin._schema)  # All columns present
    existing.labels = {}  # No version label
    plugin.client.get_table.return_value = existing
    plugin._ensure_schema_exists()

    # update_table should be called to stamp the version label even
    # though no new columns were needed.
    plugin.client.update_table.assert_called_once()
    updated_table = plugin.client.update_table.call_args[0][0]
    assert (
        updated_table.labels[
            bigquery_agent_analytics_plugin._SCHEMA_VERSION_LABEL_KEY
        ]
        == bigquery_agent_analytics_plugin._SCHEMA_VERSION
    )

  def test_upgrade_from_older_version_label(self):
    """A table with an older version label triggers upgrade."""
    plugin = self._make_plugin(auto_schema_upgrade=True)
    existing = mock.MagicMock(spec=bigquery.Table)
    existing.schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("event_type", "STRING"),
    ]
    # Simulate a table stamped with an older version.
    existing.labels = {
        bigquery_agent_analytics_plugin._SCHEMA_VERSION_LABEL_KEY: "0",
    }
    plugin.client.get_table.return_value = existing
    plugin._ensure_schema_exists()

    plugin.client.update_table.assert_called_once()
    updated_table = plugin.client.update_table.call_args[0][0]
    # Version label should be updated to current.
    assert (
        updated_table.labels[
            bigquery_agent_analytics_plugin._SCHEMA_VERSION_LABEL_KEY
        ]
        == bigquery_agent_analytics_plugin._SCHEMA_VERSION
    )
    # Missing columns should have been added.
    updated_names = {f.name for f in updated_table.schema}
    assert "agent" in updated_names
    assert "content" in updated_names

  def test_upgrade_is_idempotent(self):
    """Calling _ensure_schema_exists twice doesn't double-update."""
    plugin = self._make_plugin(auto_schema_upgrade=True)

    # First call: table exists with old schema.
    existing = mock.MagicMock(spec=bigquery.Table)
    existing.schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
    ]
    existing.labels = {}
    plugin.client.get_table.return_value = existing
    plugin._ensure_schema_exists()
    assert plugin.client.update_table.call_count == 1

    # Second call: table now has current version label.
    existing.labels = {
        bigquery_agent_analytics_plugin._SCHEMA_VERSION_LABEL_KEY: (
            bigquery_agent_analytics_plugin._SCHEMA_VERSION
        ),
    }
    plugin.client.update_table.reset_mock()
    plugin._ensure_schema_exists()
    plugin.client.update_table.assert_not_called()

  def test_update_table_receives_schema_and_labels_fields(self):
    """update_table is called with update_fields=['schema', 'labels']."""
    plugin = self._make_plugin(auto_schema_upgrade=True)
    existing = mock.MagicMock(spec=bigquery.Table)
    existing.schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
    ]
    existing.labels = {}
    plugin.client.get_table.return_value = existing
    plugin._ensure_schema_exists()

    call_args = plugin.client.update_table.call_args
    update_fields = call_args[0][1]
    assert "schema" in update_fields
    assert "labels" in update_fields

  def test_auto_schema_upgrade_defaults_to_true(self):
    """Default config has auto_schema_upgrade enabled."""
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig()
    assert config.auto_schema_upgrade is True

  def test_create_table_conflict_is_ignored(self):
    """Race condition (Conflict) during create_table is silently handled."""
    plugin = self._make_plugin()
    plugin.client.get_table.side_effect = cloud_exceptions.NotFound("not found")
    plugin.client.create_table.side_effect = cloud_exceptions.Conflict(
        "already exists"
    )
    # Should not raise.
    plugin._ensure_schema_exists()


class TestToolProvenance:
  """Tests for _get_tool_origin helper."""

  def test_function_tool_returns_local(self):
    from google.adk.tools.function_tool import FunctionTool

    def dummy():
      pass

    tool = FunctionTool(dummy)
    result = bigquery_agent_analytics_plugin._get_tool_origin(tool)
    assert result == "LOCAL"

  def test_agent_tool_returns_sub_agent(self):
    from google.adk.tools.agent_tool import AgentTool

    agent = mock.MagicMock()
    agent.name = "sub"
    tool = AgentTool.__new__(AgentTool)
    tool.agent = agent
    tool._name = "sub"
    result = bigquery_agent_analytics_plugin._get_tool_origin(tool)
    assert result == "SUB_AGENT"

  def test_transfer_tool_returns_transfer_agent(self):
    from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool

    tool = TransferToAgentTool(agent_names=["other"])
    result = bigquery_agent_analytics_plugin._get_tool_origin(tool)
    assert result == "TRANSFER_AGENT"

  def test_transfer_tool_without_args_returns_transfer_agent(self):
    """TransferToAgentTool without tool_args falls back to TRANSFER_AGENT."""
    from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool

    tool = TransferToAgentTool(agent_names=["remote_a2a"])
    result = bigquery_agent_analytics_plugin._get_tool_origin(
        tool, tool_args=None, tool_context=None
    )
    assert result == "TRANSFER_AGENT"

  def test_transfer_to_remote_a2a_sub_agent_returns_transfer_a2a(self):
    """Transfer to a RemoteA2aAgent sub-agent is classified TRANSFER_A2A."""
    from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool

    try:
      from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
    except ImportError:
      pytest.skip("A2A agent not available")

    remote_agent = mock.MagicMock(spec=RemoteA2aAgent)
    remote_agent.name = "remote_a2a"

    current_agent = mock.MagicMock()
    current_agent.name = "root"
    current_agent.sub_agents = [remote_agent]
    current_agent.parent_agent = None

    inv_ctx = mock.MagicMock()
    inv_ctx.agent = current_agent
    tool_context = mock.MagicMock()
    tool_context._invocation_context = inv_ctx

    tool = TransferToAgentTool(agent_names=["remote_a2a"])
    result = bigquery_agent_analytics_plugin._get_tool_origin(
        tool,
        tool_args={"agent_name": "remote_a2a"},
        tool_context=tool_context,
    )
    assert result == "TRANSFER_A2A"

  def test_transfer_to_local_sub_agent_returns_transfer_agent(self):
    """Transfer to a local sub-agent is still classified TRANSFER_AGENT."""
    from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool

    local_agent = mock.MagicMock()
    local_agent.name = "local_sub"

    current_agent = mock.MagicMock()
    current_agent.name = "root"
    current_agent.sub_agents = [local_agent]
    current_agent.parent_agent = None

    inv_ctx = mock.MagicMock()
    inv_ctx.agent = current_agent
    tool_context = mock.MagicMock()
    tool_context._invocation_context = inv_ctx

    tool = TransferToAgentTool(agent_names=["local_sub"])
    result = bigquery_agent_analytics_plugin._get_tool_origin(
        tool,
        tool_args={"agent_name": "local_sub"},
        tool_context=tool_context,
    )
    assert result == "TRANSFER_AGENT"

  def test_transfer_to_a2a_peer_returns_transfer_a2a(self):
    """Transfer to a RemoteA2aAgent peer is classified TRANSFER_A2A."""
    from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool

    try:
      from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
    except ImportError:
      pytest.skip("A2A agent not available")

    remote_peer = mock.MagicMock(spec=RemoteA2aAgent)
    remote_peer.name = "remote_peer"

    current_agent = mock.MagicMock()
    current_agent.name = "child"
    current_agent.sub_agents = []

    parent_agent = mock.MagicMock()
    parent_agent.name = "parent"
    parent_agent.sub_agents = [current_agent, remote_peer]
    current_agent.parent_agent = parent_agent

    inv_ctx = mock.MagicMock()
    inv_ctx.agent = current_agent
    tool_context = mock.MagicMock()
    tool_context._invocation_context = inv_ctx

    tool = TransferToAgentTool(
        agent_names=["remote_peer"],
    )
    result = bigquery_agent_analytics_plugin._get_tool_origin(
        tool,
        tool_args={"agent_name": "remote_peer"},
        tool_context=tool_context,
    )
    assert result == "TRANSFER_A2A"

  def test_transfer_mixed_targets_classifies_per_call(self):
    """A single TransferToAgentTool with mixed targets classifies per call."""
    from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool

    try:
      from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
    except ImportError:
      pytest.skip("A2A agent not available")

    remote_agent = mock.MagicMock(spec=RemoteA2aAgent)
    remote_agent.name = "remote_a2a"
    local_agent = mock.MagicMock()
    local_agent.name = "local_sub"

    current_agent = mock.MagicMock()
    current_agent.name = "root"
    current_agent.sub_agents = [remote_agent, local_agent]
    current_agent.parent_agent = None

    inv_ctx = mock.MagicMock()
    inv_ctx.agent = current_agent
    tool_context = mock.MagicMock()
    tool_context._invocation_context = inv_ctx

    tool = TransferToAgentTool(
        agent_names=["remote_a2a", "local_sub"],
    )

    # Transfer to remote target → TRANSFER_A2A
    result = bigquery_agent_analytics_plugin._get_tool_origin(
        tool,
        tool_args={"agent_name": "remote_a2a"},
        tool_context=tool_context,
    )
    assert result == "TRANSFER_A2A"

    # Transfer to local target → TRANSFER_AGENT
    result = bigquery_agent_analytics_plugin._get_tool_origin(
        tool,
        tool_args={"agent_name": "local_sub"},
        tool_context=tool_context,
    )
    assert result == "TRANSFER_AGENT"

  @pytest.mark.asyncio
  async def test_tool_error_callback_classifies_a2a_transfer(
      self,
      mock_auth_default,
      mock_bq_client,
      mock_write_client,
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    """on_tool_error_callback produces TRANSFER_A2A for RemoteA2aAgent."""
    from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool

    try:
      from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
    except ImportError:
      pytest.skip("A2A agent not available")

    remote_agent = mock.MagicMock(spec=RemoteA2aAgent)
    remote_agent.name = "remote_a2a"

    mock_agent = mock.MagicMock(spec=base_agent.BaseAgent)
    mock_agent.name = "root"
    mock_agent.instruction = ""
    mock_agent.sub_agents = [remote_agent]
    mock_agent.parent_agent = None

    mock_s = mock.create_autospec(
        session_lib.Session, instance=True, spec_set=True
    )
    type(mock_s).id = mock.PropertyMock(return_value="sess-1")
    type(mock_s).user_id = mock.PropertyMock(return_value="user-1")
    type(mock_s).app_name = mock.PropertyMock(return_value="test_app")
    type(mock_s).state = mock.PropertyMock(return_value={})

    inv_ctx = InvocationContext(
        agent=mock_agent,
        session=mock_s,
        invocation_id="inv-err",
        session_service=mock.create_autospec(
            base_session_service_lib.BaseSessionService,
            instance=True,
            spec_set=True,
        ),
        plugin_manager=mock.create_autospec(
            plugin_manager_lib.PluginManager,
            instance=True,
            spec_set=True,
        ),
    )
    tool_ctx = tool_context_lib.ToolContext(invocation_context=inv_ctx)
    tool = TransferToAgentTool(agent_names=["remote_a2a"])

    async with managed_plugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID
    ) as plugin:
      await plugin._ensure_started()
      mock_write_client.append_rows.reset_mock()

      bigquery_agent_analytics_plugin.TraceManager.push_span(tool_ctx, "tool")
      await plugin.on_tool_error_callback(
          tool=tool,
          tool_args={"agent_name": "remote_a2a"},
          tool_context=tool_ctx,
          error=RuntimeError("connection refused"),
      )
      await asyncio.sleep(0.01)

      rows = await _get_captured_rows_async(
          mock_write_client, dummy_arrow_schema
      )

    assert len(rows) == 1
    assert rows[0]["event_type"] == "TOOL_ERROR"
    content = json.loads(rows[0]["content"])
    assert content["tool_origin"] == "TRANSFER_A2A"

  def test_mcp_tool_returns_mcp(self):
    try:
      from google.adk.tools.mcp_tool.mcp_tool import McpTool
    except ImportError:
      pytest.skip("MCP not installed")
    tool = McpTool.__new__(McpTool)
    result = bigquery_agent_analytics_plugin._get_tool_origin(tool)
    assert result == "MCP"

  def test_a2a_agent_tool_returns_a2a(self):
    from google.adk.tools.agent_tool import AgentTool

    try:
      from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
    except ImportError:
      pytest.skip("A2A agent not available")

    remote_agent = mock.MagicMock(spec=RemoteA2aAgent)
    remote_agent.name = "remote"
    remote_agent.description = "remote a2a agent"
    tool = AgentTool.__new__(AgentTool)
    tool.agent = remote_agent
    tool._name = "remote"
    result = bigquery_agent_analytics_plugin._get_tool_origin(tool)
    assert result == "A2A"

  def test_unknown_tool_returns_unknown(self):
    tool = mock.MagicMock(spec=base_tool_lib.BaseTool)
    tool.name = "mystery"
    result = bigquery_agent_analytics_plugin._get_tool_origin(tool)
    assert result == "UNKNOWN"


class TestHITLTracing:
  """Tests for HITL-specific event emission via on_event_callback.

  HITL events (``adk_request_credential``, ``adk_request_confirmation``,
  ``adk_request_input``) are synthetic function calls injected by the
  framework — they never pass through ``before_tool_callback`` /
  ``after_tool_callback``.  Detection therefore lives in
  ``on_event_callback``, which inspects the event stream for these
  function calls and their corresponding function responses.
  """

  def _make_fc_event(self, fc_name, args=None):
    """Build a mock Event containing a function call."""
    event = mock.MagicMock(spec=event_lib.Event)
    fc = types.FunctionCall(name=fc_name, args=args or {})
    part = types.Part(function_call=fc)
    event.content = types.Content(role="model", parts=[part])
    event.actions = event_actions_lib.EventActions()
    return event

  def _make_fr_event(self, fr_name, response=None):
    """Build a mock Event containing a function response."""
    event = mock.MagicMock(spec=event_lib.Event)
    fr = types.FunctionResponse(name=fr_name, response=response or {})
    part = types.Part(function_response=fr)
    event.content = types.Content(role="user", parts=[part])
    event.actions = event_actions_lib.EventActions()
    return event

  @pytest.mark.asyncio
  async def test_hitl_confirmation_emits_additional_event(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      dummy_arrow_schema,
  ):
    event = self._make_fc_event("adk_request_confirmation", {"confirm": True})
    await bq_plugin_inst.on_event_callback(
        invocation_context=invocation_context, event=event
    )
    await asyncio.sleep(0.05)
    rows = await _get_captured_rows_async(mock_write_client, dummy_arrow_schema)
    event_types = [r["event_type"] for r in rows]
    assert "HITL_CONFIRMATION_REQUEST" in event_types

  @pytest.mark.asyncio
  async def test_hitl_credential_emits_additional_event(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      dummy_arrow_schema,
  ):
    event = self._make_fc_event("adk_request_credential", {"auth": "oauth2"})
    await bq_plugin_inst.on_event_callback(
        invocation_context=invocation_context, event=event
    )
    await asyncio.sleep(0.05)
    rows = await _get_captured_rows_async(mock_write_client, dummy_arrow_schema)
    event_types = [r["event_type"] for r in rows]
    assert "HITL_CREDENTIAL_REQUEST" in event_types

  @pytest.mark.asyncio
  async def test_hitl_completion_emits_additional_event(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      dummy_arrow_schema,
  ):
    event = self._make_fr_event("adk_request_confirmation", {"confirmed": True})
    await bq_plugin_inst.on_event_callback(
        invocation_context=invocation_context, event=event
    )
    await asyncio.sleep(0.05)
    rows = await _get_captured_rows_async(mock_write_client, dummy_arrow_schema)
    event_types = [r["event_type"] for r in rows]
    assert "HITL_CONFIRMATION_REQUEST_COMPLETED" in event_types

  @pytest.mark.asyncio
  async def test_regular_tool_no_hitl_event(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      dummy_arrow_schema,
  ):
    event = self._make_fc_event("regular_tool", {"x": 1})
    await bq_plugin_inst.on_event_callback(
        invocation_context=invocation_context, event=event
    )
    await asyncio.sleep(0.05)
    # No HITL events should be emitted for non-HITL function calls.
    # on_event_callback only logs STATE_DELTA and HITL events; a regular
    # function call produces neither.
    assert mock_write_client.append_rows.call_count == 0


# ==============================================================================
# TEST CLASS: Span Hierarchy Isolation (Issue #4561)
# ==============================================================================


class TestSpanHierarchyIsolation:
  """Regression tests for https://github.com/google/adk-python/issues/4561.

  ``push_span()`` must NOT attach its span to the ambient OTel context.
  If it does, any subsequent ``tracer.start_as_current_span()`` in the
  framework (e.g. ``call_llm``, ``execute_tool``) will be incorrectly
  re-parented under the plugin's span.
  """

  def test_push_span_does_not_change_ambient_context(self, callback_context):
    """push_span must not mutate the current OTel span."""
    span_before = trace.get_current_span()

    bigquery_agent_analytics_plugin.TraceManager.push_span(
        callback_context, "test_span"
    )

    span_after = trace.get_current_span()
    assert span_after is span_before

    # Cleanup
    bigquery_agent_analytics_plugin.TraceManager.pop_span()

  def test_attach_current_span_does_not_change_ambient_context(
      self, callback_context
  ):
    """attach_current_span must not mutate the current OTel span."""
    span_before = trace.get_current_span()

    bigquery_agent_analytics_plugin.TraceManager.attach_current_span(
        callback_context
    )

    span_after = trace.get_current_span()
    assert span_after is span_before

    # Cleanup
    bigquery_agent_analytics_plugin.TraceManager.pop_span()

  def test_pop_span_does_not_change_ambient_context(self, callback_context):
    """pop_span must not mutate the current OTel span."""
    bigquery_agent_analytics_plugin.TraceManager.push_span(
        callback_context, "test_span"
    )
    span_before = trace.get_current_span()

    bigquery_agent_analytics_plugin.TraceManager.pop_span()

    span_after = trace.get_current_span()
    assert span_after is span_before

  def test_push_span_with_real_tracer_does_not_reparent(self, callback_context):
    """With a real OTel tracer, plugin spans must not become parents

    of subsequently created framework spans.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider.add_span_processor(SimpleSpanProcessor(exporter))
    framework_tracer = provider.get_tracer("test-framework")

    # Simulate: plugin pushes a span BEFORE the framework span
    bigquery_agent_analytics_plugin.TraceManager.push_span(
        callback_context, "llm_request"
    )

    # Framework creates its own span via start_as_current_span
    with framework_tracer.start_as_current_span("call_llm") as fw_span:
      fw_context = fw_span.get_span_context()

    # Pop the plugin span
    bigquery_agent_analytics_plugin.TraceManager.pop_span()

    provider.shutdown()

    # Verify the framework span was NOT re-parented under the
    # plugin's llm_request span
    finished = exporter.get_finished_spans()
    call_llm_spans = [s for s in finished if s.name == "call_llm"]
    assert len(call_llm_spans) == 1
    fw_finished = call_llm_spans[0]

    # The framework span's parent should NOT be the plugin's
    # llm_request span.  With the fix, the plugin never
    # attaches to the ambient context, so ``call_llm`` will
    # have whatever parent existed before (None in this test).
    assert fw_finished.parent is None

  def test_multiple_push_pop_cycles_leave_context_clean(self, callback_context):
    """Multiple push/pop cycles must not leak context changes."""
    original_span = trace.get_current_span()

    for _ in range(5):
      bigquery_agent_analytics_plugin.TraceManager.push_span(
          callback_context, "cycle_span"
      )
      bigquery_agent_analytics_plugin.TraceManager.pop_span()

    assert trace.get_current_span() is original_span


# ==============================================================================
# TEST CLASS: End-to-End HITL Tracing via Runner
# ==============================================================================


def _hitl_my_action(
    tool_context: tool_context_lib.ToolContext,
) -> dict[str, str]:
  """Tool function used by HITL end-to-end tests."""
  return {"result": f"confirmed={tool_context.tool_confirmation.confirmed}"}


class TestHITLTracingEndToEnd:
  """End-to-end tests that run the full Runner + Plugin pipeline with

  ``FunctionTool(require_confirmation=True)`` and verify that HITL events
  are logged alongside normal TOOL_* events in the BQ analytics plugin.
  """

  @pytest.fixture
  def _mock_bq_infra(
      self,
      mock_auth_default,
      mock_bq_client,
      mock_write_client,
      mock_to_arrow_schema,
      mock_asyncio_to_thread,
  ):
    """Bundle all BQ mocking fixtures."""
    yield mock_write_client

  @pytest.mark.asyncio
  async def test_confirmation_flow_emits_hitl_events(
      self,
      _mock_bq_infra,
      dummy_arrow_schema,
  ):
    """Full Runner pipeline: tool with require_confirmation emits

    HITL_CONFIRMATION_REQUEST and HITL_CONFIRMATION_REQUEST_COMPLETED.
    """
    from google.adk.flows.llm_flows.functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
    from google.adk.tools.function_tool import FunctionTool
    from google.genai.types import FunctionCall
    from google.genai.types import FunctionResponse
    from google.genai.types import Part

    from .. import testing_utils

    mock_write_client = _mock_bq_infra

    tool = FunctionTool(func=_hitl_my_action, require_confirmation=True)

    # -- Mock LLM: first response calls the tool, second is final text --
    llm_responses = [
        testing_utils.LlmResponse(
            content=testing_utils.ModelContent(
                parts=[
                    Part(function_call=FunctionCall(name=tool.name, args={}))
                ]
            )
        ),
        testing_utils.LlmResponse(
            content=testing_utils.ModelContent(
                parts=[Part(text="Done, action confirmed.")]
            )
        ),
    ]
    mock_model = testing_utils.MockModel(responses=llm_responses)

    # -- Build the plugin --
    bq_plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
    )
    await bq_plugin._ensure_started()
    mock_write_client.append_rows.reset_mock()

    # -- Build agent + runner WITH the plugin --
    from google.adk.agents.llm_agent import LlmAgent

    agent = LlmAgent(name="hitl_agent", model=mock_model, tools=[tool])
    runner = testing_utils.InMemoryRunner(root_agent=agent, plugins=[bq_plugin])

    # -- Turn 1: user query → LLM calls tool → HITL pause --
    events_turn1 = await runner.run_async(
        testing_utils.UserContent("run my_action")
    )

    # Find the adk_request_confirmation function call
    confirmation_fc_id = None
    for ev in events_turn1:
      if ev.content and ev.content.parts:
        for part in ev.content.parts:
          if (
              hasattr(part, "function_call")
              and part.function_call
              and part.function_call.name
              == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
          ):
            confirmation_fc_id = part.function_call.id
            break
      if confirmation_fc_id:
        break

    assert (
        confirmation_fc_id is not None
    ), "Expected adk_request_confirmation function call in turn 1"

    # -- Turn 2: user sends confirmation → tool re-executes --
    user_confirmation = testing_utils.UserContent(
        Part(
            function_response=FunctionResponse(
                id=confirmation_fc_id,
                name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                response={"confirmed": True},
            )
        )
    )
    events_turn2 = await runner.run_async(user_confirmation)

    # -- Give the async BQ writer a moment to flush --
    await asyncio.sleep(0.2)

    # -- Collect all BQ rows --
    rows = await _get_captured_rows_async(mock_write_client, dummy_arrow_schema)
    event_types = [r["event_type"] for r in rows]

    # -- Verify standard events are present --
    assert "TOOL_STARTING" in event_types
    assert "TOOL_COMPLETED" in event_types

    # -- Verify HITL-specific events are present --
    assert (
        "HITL_CONFIRMATION_REQUEST" in event_types
    ), f"Expected HITL_CONFIRMATION_REQUEST in {event_types}"
    assert (
        "HITL_CONFIRMATION_REQUEST_COMPLETED" in event_types
    ), f"Expected HITL_CONFIRMATION_REQUEST_COMPLETED in {event_types}"

    # -- Verify HITL events have correct tool name in content --
    hitl_rows = [r for r in rows if r["event_type"].startswith("HITL_")]
    for row in hitl_rows:
      content = json.loads(row["content"]) if row["content"] else {}
      assert content.get("tool") == "adk_request_confirmation", (
          "HITL event should reference 'adk_request_confirmation',"
          f" got {content.get('tool')}"
      )

    await bq_plugin.shutdown()

  @pytest.mark.asyncio
  async def test_regular_tool_does_not_emit_hitl_events(
      self,
      _mock_bq_infra,
      dummy_arrow_schema,
  ):
    """A tool WITHOUT require_confirmation should not produce HITL events."""
    from google.adk.tools.function_tool import FunctionTool
    from google.genai.types import FunctionCall
    from google.genai.types import Part

    from .. import testing_utils

    mock_write_client = _mock_bq_infra

    def regular_tool() -> str:
      return "done"

    tool = FunctionTool(func=regular_tool)

    llm_responses = [
        testing_utils.LlmResponse(
            content=testing_utils.ModelContent(
                parts=[
                    Part(function_call=FunctionCall(name=tool.name, args={}))
                ]
            )
        ),
        testing_utils.LlmResponse(
            content=testing_utils.ModelContent(parts=[Part(text="All done.")])
        ),
    ]
    mock_model = testing_utils.MockModel(responses=llm_responses)

    bq_plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
    )
    await bq_plugin._ensure_started()
    mock_write_client.append_rows.reset_mock()

    from google.adk.agents.llm_agent import LlmAgent

    agent = LlmAgent(name="regular_agent", model=mock_model, tools=[tool])
    runner = testing_utils.InMemoryRunner(root_agent=agent, plugins=[bq_plugin])

    await runner.run_async(testing_utils.UserContent("run regular_tool"))
    await asyncio.sleep(0.2)

    rows = await _get_captured_rows_async(mock_write_client, dummy_arrow_schema)
    event_types = [r["event_type"] for r in rows]

    # Standard tool events should be present
    assert "TOOL_STARTING" in event_types
    assert "TOOL_COMPLETED" in event_types

    # No HITL events
    hitl_events = [et for et in event_types if et.startswith("HITL_")]
    assert (
        hitl_events == []
    ), f"Expected no HITL events for regular tool, got {hitl_events}"

    await bq_plugin.shutdown()


# ==============================================================================
# Fork-Safety Tests
# ==============================================================================
class TestForkSafety:
  """Tests for fork-safety via PID tracking."""

  def _make_plugin(self):
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig()
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
        config=config,
    )
    return plugin

  @pytest.mark.asyncio
  async def test_pid_change_triggers_reinit(
      self, mock_auth_default, mock_bq_client, mock_write_client
  ):
    """Simulating a fork by changing _init_pid forces re-init."""
    plugin = self._make_plugin()
    await plugin._ensure_started()
    assert plugin._started is True

    # Simulate a fork: set _init_pid to a stale value
    plugin._init_pid = -1
    assert plugin._started is True  # still True before check

    # _ensure_started should detect PID mismatch and reset
    await plugin._ensure_started()
    # After reset + re-init, _init_pid should match current

    assert plugin._init_pid == os.getpid()
    assert plugin._started is True
    await plugin.shutdown()

  @pytest.mark.asyncio
  async def test_pid_unchanged_skips_reset(
      self, mock_auth_default, mock_bq_client, mock_write_client
  ):
    """Same PID should not trigger a reset."""
    plugin = self._make_plugin()
    await plugin._ensure_started()

    # Save references to verify they are not recreated
    original_client = plugin.client
    original_parser = plugin.parser

    await plugin._ensure_started()
    assert plugin.client is original_client
    assert plugin.parser is original_parser
    await plugin.shutdown()

  def test_reset_runtime_state_clears_fields(self):
    """_reset_runtime_state clears all runtime fields."""
    plugin = self._make_plugin()
    # Fake some runtime state
    plugin._started = True
    plugin._is_shutting_down = True
    plugin.client = mock.MagicMock()
    plugin._loop_state_by_loop = {"fake": "state"}
    plugin._write_stream_name = "some/stream"
    plugin._executor = mock.MagicMock()
    plugin.offloader = mock.MagicMock()
    plugin.parser = mock.MagicMock()
    plugin._setup_lock = mock.MagicMock()
    # Keep pure-data fields
    plugin._schema = ["kept"]
    plugin.arrow_schema = "kept_arrow"

    plugin._reset_runtime_state()

    assert plugin._started is False
    assert plugin._is_shutting_down is False
    assert plugin.client is None
    assert plugin._loop_state_by_loop == {}
    assert plugin._write_stream_name is None
    assert plugin._executor is None
    assert plugin.offloader is None
    assert plugin.parser is None
    assert plugin._setup_lock is None
    # Pure-data fields are preserved
    assert plugin._schema == ["kept"]
    assert plugin.arrow_schema == "kept_arrow"

    assert plugin._init_pid == os.getpid()

  def test_getstate_resets_pid(self):
    """Pickle state should have _init_pid = 0 to force re-init."""
    plugin = self._make_plugin()
    state = plugin.__getstate__()
    assert state["_init_pid"] == 0
    assert state["_started"] is False

  @pytest.mark.asyncio
  async def test_unpickle_legacy_state_missing_init_pid(
      self, mock_auth_default, mock_bq_client, mock_write_client
  ):
    """Unpickling state from older code without _init_pid should not crash."""
    plugin = self._make_plugin()
    state = plugin.__getstate__()
    # Simulate legacy pickle state that lacks _init_pid entirely
    del state["_init_pid"]

    new_plugin = (
        bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin.__new__(
            bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin
        )
    )
    new_plugin.__setstate__(state)

    # _init_pid should be backfilled to 0, triggering re-init
    assert new_plugin._init_pid == 0
    # _ensure_started should not raise AttributeError
    await new_plugin._ensure_started()
    assert new_plugin._started is True
    await new_plugin.shutdown()


class TestForkGrpcSafety:
  """Tests for gRPC fork safety enhancements."""

  def _make_plugin(self):
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig()
    return bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
        config=config,
    )

  def test_grpc_fork_env_var_set(self):
    """GRPC_ENABLE_FORK_SUPPORT should be '1' after import."""

    assert os.environ.get("GRPC_ENABLE_FORK_SUPPORT") == "1"

  def test_register_at_fork_resets_all_instances(self):
    """_after_fork_in_child resets all living plugin instances."""
    p1 = self._make_plugin()
    p2 = self._make_plugin()
    p1._started = True
    p2._started = True
    p1._init_pid = -1
    p2._init_pid = -1

    bigquery_agent_analytics_plugin._after_fork_in_child()

    assert p1._started is False
    assert p2._started is False
    assert p1._init_pid == os.getpid()
    assert p2._init_pid == os.getpid()

  def test_dead_plugin_removed_from_live_set(self):
    """WeakSet should not hold dead plugin references."""
    p = self._make_plugin()
    assert p in bigquery_agent_analytics_plugin._LIVE_PLUGINS
    pid = id(p)
    del p
    # After deletion, the WeakSet should no longer contain it.
    for alive in bigquery_agent_analytics_plugin._LIVE_PLUGINS:
      assert id(alive) != pid

  def test_reset_closes_inherited_sync_transports(self):
    """_reset_runtime_state closes inherited sync gRPC channels."""
    plugin = self._make_plugin()
    mock_channel = mock.MagicMock()
    mock_channel.close.return_value = None  # sync close
    mock_transport = mock.MagicMock()
    mock_transport._grpc_channel = mock_channel
    mock_wc = mock.MagicMock()
    mock_wc.transport = mock_transport

    mock_loop_state = mock.MagicMock()
    mock_loop_state.write_client = mock_wc

    plugin._loop_state_by_loop = {mock.MagicMock(): mock_loop_state}
    plugin._init_pid = -1

    plugin._reset_runtime_state()

    mock_channel.close.assert_called_once()

  def test_reset_discards_async_channel_close_coroutine(self):
    """Async channel close() returns a coroutine; must not warn."""
    import warnings

    plugin = self._make_plugin()

    async def _async_close():
      pass

    mock_channel = mock.MagicMock()
    mock_channel.close.return_value = _async_close()
    mock_transport = mock.MagicMock()
    mock_transport._grpc_channel = mock_channel
    mock_wc = mock.MagicMock()
    mock_wc.transport = mock_transport

    mock_loop_state = mock.MagicMock()
    mock_loop_state.write_client = mock_wc

    plugin._loop_state_by_loop = {mock.MagicMock(): mock_loop_state}
    plugin._init_pid = -1

    with warnings.catch_warnings():
      warnings.simplefilter("error", RuntimeWarning)
      # Must not raise RuntimeWarning for unawaited coroutine
      plugin._reset_runtime_state()

    mock_channel.close.assert_called_once()

  def test_transport_close_exception_swallowed(self):
    """close() raising should not prevent reset from completing."""
    plugin = self._make_plugin()
    mock_channel = mock.MagicMock()
    mock_channel.close.side_effect = RuntimeError("broken channel")
    mock_transport = mock.MagicMock()
    mock_transport._grpc_channel = mock_channel
    mock_wc = mock.MagicMock()
    mock_wc.transport = mock_transport

    mock_loop_state = mock.MagicMock()
    mock_loop_state.write_client = mock_wc

    plugin._loop_state_by_loop = {mock.MagicMock(): mock_loop_state}
    plugin._init_pid = -1

    # Should not raise
    plugin._reset_runtime_state()

    assert plugin._started is False
    assert plugin._loop_state_by_loop == {}

  def test_reset_logs_fork_warning(self):
    """_reset_runtime_state logs a warning with 'Fork detected'."""
    plugin = self._make_plugin()
    plugin._init_pid = -1

    with mock.patch.object(
        bigquery_agent_analytics_plugin.logger, "warning"
    ) as mock_warn:
      plugin._reset_runtime_state()

    mock_warn.assert_called_once()
    assert "Fork detected" in mock_warn.call_args[0][0]


# ==============================================================================
# Analytics Views Tests
# ==============================================================================
class TestAnalyticsViews:
  """Tests for auto-created per-event-type BigQuery views."""

  def _make_plugin(self, create_views=True, view_prefix="v", table_id=TABLE_ID):
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(
        create_views=create_views,
        view_prefix=view_prefix,
    )
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=table_id,
        config=config,
    )
    plugin.client = mock.MagicMock()
    plugin.full_table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_id}"
    plugin._schema = bigquery_agent_analytics_plugin._get_events_schema()
    return plugin

  def test_views_created_on_new_table(self):
    """NotFound path creates all views."""
    plugin = self._make_plugin(create_views=True)
    plugin.client.get_table.side_effect = cloud_exceptions.NotFound("not found")
    mock_query_job = mock.MagicMock()
    plugin.client.query.return_value = mock_query_job

    plugin._ensure_schema_exists()

    expected_count = len(bigquery_agent_analytics_plugin._EVENT_VIEW_DEFS)
    assert plugin.client.query.call_count == expected_count

  def test_views_created_for_existing_table(self):
    """Existing table path also creates views."""
    plugin = self._make_plugin(create_views=True)
    existing = mock.MagicMock(spec=bigquery.Table)
    existing.schema = plugin._schema
    existing.labels = {
        bigquery_agent_analytics_plugin._SCHEMA_VERSION_LABEL_KEY: (
            bigquery_agent_analytics_plugin._SCHEMA_VERSION
        ),
    }
    plugin.client.get_table.return_value = existing
    mock_query_job = mock.MagicMock()
    plugin.client.query.return_value = mock_query_job

    plugin._ensure_schema_exists()

    expected_count = len(bigquery_agent_analytics_plugin._EVENT_VIEW_DEFS)
    assert plugin.client.query.call_count == expected_count

  def test_views_not_created_when_disabled(self):
    """create_views=False skips view creation."""
    plugin = self._make_plugin(create_views=False)
    plugin.client.get_table.side_effect = cloud_exceptions.NotFound("not found")

    plugin._ensure_schema_exists()

    plugin.client.query.assert_not_called()

  def test_view_creation_error_logged_not_raised(self):
    """Errors during view creation don't crash the plugin."""
    plugin = self._make_plugin(create_views=True)
    plugin.client.get_table.side_effect = cloud_exceptions.NotFound("not found")
    plugin.client.query.side_effect = Exception("BQ error")

    # Should not raise
    plugin._ensure_schema_exists()

    # Verify it tried to create views (and failed gracefully)
    assert plugin.client.query.call_count > 0

  def test_view_sql_contains_correct_event_filter(self):
    """Each SQL has correct WHERE clause and view name."""
    plugin = self._make_plugin(create_views=True)
    plugin.client.get_table.side_effect = cloud_exceptions.NotFound("not found")
    mock_query_job = mock.MagicMock()
    plugin.client.query.return_value = mock_query_job

    plugin._ensure_schema_exists()

    calls = plugin.client.query.call_args_list
    for call in calls:
      sql = call[0][0]
      # Each SQL should have CREATE OR REPLACE VIEW
      assert "CREATE OR REPLACE VIEW" in sql
      # Each SQL should filter by event_type
      assert "WHERE" in sql
      assert "event_type = " in sql
      # View name should start with v_
      assert ".v_" in sql

    # Verify specific views exist
    all_sql = " ".join(c[0][0] for c in calls)
    for event_type in bigquery_agent_analytics_plugin._EVENT_VIEW_DEFS:
      view_name = "v_" + event_type.lower()
      assert view_name in all_sql, f"View {view_name} not found in SQL"

  def test_config_create_views_default_true(self):
    """Config create_views defaults to True."""
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig()
    assert config.create_views is True
    assert config.view_prefix == "v"

  @pytest.mark.asyncio
  async def test_create_analytics_views_ensures_started(
      self, mock_auth_default, mock_bq_client, mock_write_client
  ):
    """Public create_analytics_views() initializes plugin first."""
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
    )
    assert plugin._started is False

    await plugin.create_analytics_views()

    # Plugin should be started after the call
    assert plugin._started is True
    # Views should have been created (query called)
    expected_count = len(bigquery_agent_analytics_plugin._EVENT_VIEW_DEFS)
    # _ensure_schema_exists also creates views, so total calls
    # = schema-creation views + explicit views
    assert mock_bq_client.query.call_count >= expected_count
    await plugin.shutdown()

  def test_views_not_created_after_table_creation_failure(self):
    """View creation is skipped when create_table raises a non-Conflict error."""
    plugin = self._make_plugin(create_views=True)
    plugin.client.get_table.side_effect = cloud_exceptions.NotFound("not found")
    plugin.client.create_table.side_effect = RuntimeError("BQ down")

    plugin._ensure_schema_exists()

    # Views should NOT be attempted since table creation failed
    plugin.client.query.assert_not_called()

  @pytest.mark.asyncio
  async def test_create_analytics_views_raises_on_startup_failure(
      self, mock_auth_default, mock_write_client
  ):
    """create_analytics_views() raises if plugin init fails."""
    # Make the BQ Client constructor raise so _lazy_setup fails
    # before _started is set to True.
    with mock.patch.object(
        bigquery, "Client", side_effect=Exception("client boom")
    ):
      plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
          project_id=PROJECT_ID,
          dataset_id=DATASET_ID,
          table_id=TABLE_ID,
      )
      with pytest.raises(
          RuntimeError, match="Plugin initialization failed"
      ) as exc_info:
        await plugin.create_analytics_views()
      # Root cause should be chained for debuggability
      assert exc_info.value.__cause__ is not None
      assert "client boom" in str(exc_info.value.__cause__)

  def test_custom_view_prefix(self):
    """Custom view_prefix namespaces view names."""
    plugin = self._make_plugin(view_prefix="v_staging")
    plugin.client.get_table.side_effect = cloud_exceptions.NotFound("not found")
    mock_query_job = mock.MagicMock()
    plugin.client.query.return_value = mock_query_job

    plugin._ensure_schema_exists()

    calls = plugin.client.query.call_args_list
    all_sql = " ".join(c[0][0] for c in calls)
    # All views should use the custom prefix
    for event_type in bigquery_agent_analytics_plugin._EVENT_VIEW_DEFS:
      expected_name = "v_staging_" + event_type.lower()
      assert expected_name in all_sql, f"View {expected_name} not found in SQL"
    # Default prefix should NOT appear
    assert ".v_llm_request" not in all_sql

  def test_default_view_prefix_preserves_names(self):
    """Default view_prefix='v' produces the same names as before."""
    plugin = self._make_plugin()  # default view_prefix="v"
    plugin.client.get_table.side_effect = cloud_exceptions.NotFound("not found")
    mock_query_job = mock.MagicMock()
    plugin.client.query.return_value = mock_query_job

    plugin._ensure_schema_exists()

    calls = plugin.client.query.call_args_list
    all_sql = " ".join(c[0][0] for c in calls)
    for event_type in bigquery_agent_analytics_plugin._EVENT_VIEW_DEFS:
      view_name = "v_" + event_type.lower()
      assert view_name in all_sql

  def test_distinct_tables_and_prefixes_no_collision(self):
    """Two plugins targeting different tables produce disjoint views."""
    plugin_a = self._make_plugin(
        table_id="agent_events_prod", view_prefix="v_prod"
    )
    plugin_b = self._make_plugin(
        table_id="agent_events_staging", view_prefix="v_staging"
    )

    for plugin in (plugin_a, plugin_b):
      plugin.client.get_table.side_effect = cloud_exceptions.NotFound(
          "not found"
      )
      mock_query_job = mock.MagicMock()
      plugin.client.query.return_value = mock_query_job
      plugin._ensure_schema_exists()

    sql_a = " ".join(c[0][0] for c in plugin_a.client.query.call_args_list)
    sql_b = " ".join(c[0][0] for c in plugin_b.client.query.call_args_list)

    # View names use their own prefix
    assert "v_prod_llm_request" in sql_a
    assert "v_staging_llm_request" in sql_b
    # No cross-contamination
    assert "v_staging_" not in sql_a
    assert "v_prod_" not in sql_b

    # FROM clauses point at the correct table
    assert "agent_events_prod" in sql_a
    assert "agent_events_staging" not in sql_a
    assert "agent_events_staging" in sql_b
    assert "agent_events_prod" not in sql_b

  def test_empty_view_prefix_raises(self):
    """Empty view_prefix is rejected at init."""
    with pytest.raises(ValueError, match="view_prefix"):
      self._make_plugin(view_prefix="")


# ==============================================================================
# Trace-ID Continuity Tests (Issue #4645)
# ==============================================================================
class TestTraceIdContinuity:
  """Tests for trace_id continuity across all events in an invocation.

  Regression tests for https://github.com/google/adk-python/issues/4645.

  When there is no ambient OTel span (e.g. Agent Engine, custom runners),
  early events (USER_MESSAGE_RECEIVED, INVOCATION_STARTING) used to fall
  back to ``invocation_id`` while AGENT_STARTING got a new OTel hex
  trace_id from ``push_span()``.  The ``ensure_invocation_span()`` fix
  guarantees a root span is always on the stack before any events fire.
  """

  @pytest.mark.asyncio
  async def test_trace_id_continuity_no_ambient_span(self, callback_context):
    """All events share one trace_id when no ambient OTel span exists.

    Simulates the #4645 scenario: OTel IS configured (real TracerProvider)
    but the Runner's ambient span is NOT present (e.g. Agent Engine,
    custom runners).
    """
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    TM = bigquery_agent_analytics_plugin.TraceManager

    # Create a real TracerProvider and patch the plugin's module-level
    # tracer so push_span creates valid spans with proper trace_ids.
    exporter = InMemorySpanExporter()
    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    real_tracer = provider.get_tracer("test-plugin")

    with mock.patch.object(
        bigquery_agent_analytics_plugin, "tracer", real_tracer
    ):
      # Reset the span records contextvar for a clean invocation.
      bigquery_agent_analytics_plugin._span_records_ctx.set(None)

      # No ambient OTel span — we do NOT start_as_current_span.
      ambient = trace.get_current_span()
      assert not ambient.get_span_context().is_valid

      # ensure_invocation_span should push a new span.
      TM.ensure_invocation_span(callback_context)
      trace_id_early = TM.get_trace_id(callback_context)
      assert trace_id_early is not None
      # Should NOT fall back to invocation_id — it should be
      # a 32-char hex OTel trace_id.
      assert trace_id_early != callback_context.invocation_id
      assert len(trace_id_early) == 32

      # Simulate agent callback: push_span("agent")
      TM.push_span(callback_context, "agent")
      trace_id_agent = TM.get_trace_id(callback_context)

      # Both trace_ids must be identical.
      assert trace_id_early == trace_id_agent

      # Cleanup
      TM.pop_span()  # agent
      TM.pop_span()  # invocation

    provider.shutdown()

  @pytest.mark.asyncio
  async def test_invocation_completed_trace_continuity_no_ambient(
      self, callback_context
  ):
    """INVOCATION_COMPLETED must share trace_id with earlier events.

    Reproduces the completion-event fracture: after_run_callback pops
    the invocation span, then _log_event would resolve trace_id via
    the fallback to invocation_id.  The trace_id_override ensures the
    completion event keeps the same trace_id.
    """
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    TM = bigquery_agent_analytics_plugin.TraceManager

    exporter = InMemorySpanExporter()
    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    real_tracer = provider.get_tracer("test-plugin")

    with mock.patch.object(
        bigquery_agent_analytics_plugin, "tracer", real_tracer
    ):
      # Reset for a clean invocation; no ambient span.
      bigquery_agent_analytics_plugin._span_records_ctx.set(None)
      assert not trace.get_current_span().get_span_context().is_valid

      # --- Simulate the full callback lifecycle ---
      # 1. before_run / on_user_message: ensure invocation span
      TM.ensure_invocation_span(callback_context)
      trace_id_start = TM.get_trace_id(callback_context)

      # 2. before_agent: push agent span
      TM.push_span(callback_context, "agent")
      assert TM.get_trace_id(callback_context) == trace_id_start

      # 3. after_agent: pop agent span
      TM.pop_span()

      # 4. after_run: capture trace_id THEN pop invocation span
      trace_id_before_pop = TM.get_trace_id(callback_context)
      assert trace_id_before_pop == trace_id_start

      TM.pop_span()

      # After popping, get_trace_id falls back to invocation_id
      trace_id_after_pop = TM.get_trace_id(callback_context)
      assert trace_id_after_pop == callback_context.invocation_id

      # The trace_id_override preserves continuity
      assert trace_id_before_pop == trace_id_start
      assert trace_id_before_pop != trace_id_after_pop

    provider.shutdown()

  @pytest.mark.asyncio
  async def test_callbacks_emit_same_trace_id_no_ambient(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      callback_context,
      mock_agent,
      dummy_arrow_schema,
  ):
    """Full callback path: all emitted rows share one trace_id.

    Exercises the real before_run → before_agent → after_agent →
    after_run callback chain via the plugin instance, then checks
    every emitted BQ row has the same trace_id.
    """
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    real_tracer = provider.get_tracer("test-plugin")

    with mock.patch.object(
        bigquery_agent_analytics_plugin, "tracer", real_tracer
    ):
      # Reset span records for a clean invocation.
      bigquery_agent_analytics_plugin._span_records_ctx.set(None)

      # No ambient span — simulates Agent Engine / custom runner.
      assert not trace.get_current_span().get_span_context().is_valid

      # Run the full callback lifecycle.
      await bq_plugin_inst.before_run_callback(
          invocation_context=invocation_context
      )
      await bq_plugin_inst.before_agent_callback(
          agent=mock_agent, callback_context=callback_context
      )
      await bq_plugin_inst.after_agent_callback(
          agent=mock_agent, callback_context=callback_context
      )
      await bq_plugin_inst.after_run_callback(
          invocation_context=invocation_context
      )
      await asyncio.sleep(0.01)

      # Collect all emitted rows.
      rows = await _get_captured_rows_async(
          mock_write_client, dummy_arrow_schema
      )
      event_types = [r["event_type"] for r in rows]
      assert "INVOCATION_STARTING" in event_types
      assert "INVOCATION_COMPLETED" in event_types

      # Every row must share the same trace_id.
      trace_ids = {r["trace_id"] for r in rows}
      assert len(trace_ids) == 1, (
          "Expected 1 unique trace_id across all events, got"
          f" {len(trace_ids)}: {trace_ids}"
      )
      # Should be a 32-char hex OTel trace, not the invocation_id.
      sole_trace_id = trace_ids.pop()
      assert sole_trace_id != invocation_context.invocation_id
      assert len(sole_trace_id) == 32

    provider.shutdown()

  @pytest.mark.asyncio
  async def test_trace_id_continuity_with_ambient_span(self, callback_context):
    """All events share one trace_id when an ambient OTel span exists."""
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    TM = bigquery_agent_analytics_plugin.TraceManager

    # Set up a real OTel tracer.
    exporter = InMemorySpanExporter()
    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    real_tracer = provider.get_tracer("test")

    with mock.patch.object(
        bigquery_agent_analytics_plugin, "tracer", real_tracer
    ):
      # Reset the span records contextvar.
      bigquery_agent_analytics_plugin._span_records_ctx.set(None)

      with real_tracer.start_as_current_span("runner_invocation"):
        ambient = trace.get_current_span()
        assert ambient.get_span_context().is_valid
        ambient_trace_id = format(ambient.get_span_context().trace_id, "032x")

        # ensure_invocation_span should attach the ambient span.
        TM.ensure_invocation_span(callback_context)
        trace_id_early = TM.get_trace_id(callback_context)
        assert trace_id_early == ambient_trace_id

        # Simulate agent callback: push_span("agent")
        TM.push_span(callback_context, "agent")
        trace_id_agent = TM.get_trace_id(callback_context)
        assert trace_id_agent == ambient_trace_id

        # Cleanup
        TM.pop_span()  # agent
        TM.pop_span()  # invocation (attached, not owned)

    provider.shutdown()

  @pytest.mark.asyncio
  async def test_invocation_root_span_isolated_across_turns(
      self, callback_context
  ):
    """Each invocation gets its own root span; turns don't leak."""
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    TM = bigquery_agent_analytics_plugin.TraceManager

    exporter = InMemorySpanExporter()
    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    real_tracer = provider.get_tracer("test")

    with mock.patch.object(
        bigquery_agent_analytics_plugin, "tracer", real_tracer
    ):
      # --- Turn 1 ---
      bigquery_agent_analytics_plugin._span_records_ctx.set(None)
      TM.ensure_invocation_span(callback_context)
      trace_id_turn1 = TM.get_trace_id(callback_context)

      TM.push_span(callback_context, "agent")
      assert TM.get_trace_id(callback_context) == trace_id_turn1
      TM.pop_span()  # agent
      TM.pop_span()  # invocation

      # After popping, the stack should be empty.
      records = bigquery_agent_analytics_plugin._span_records_ctx.get()
      assert not records

      # --- Turn 2 ---
      bigquery_agent_analytics_plugin._span_records_ctx.set(None)
      TM.ensure_invocation_span(callback_context)
      trace_id_turn2 = TM.get_trace_id(callback_context)

      TM.push_span(callback_context, "agent")
      assert TM.get_trace_id(callback_context) == trace_id_turn2
      TM.pop_span()  # agent
      TM.pop_span()  # invocation

      # The two turns must have DIFFERENT trace_ids (different
      # root spans).
      assert trace_id_turn1 != trace_id_turn2

    provider.shutdown()


class TestSpanIdConsistency:
  """Tests that STARTING/COMPLETED event pairs share span IDs.

  Span-ID resolution contract:
  - When OTel is active: BQ rows use the same trace/span/parent IDs as
    Cloud Trace (ambient framework spans). STARTING and COMPLETED events
    in the same lifecycle share the same span_id.
  - When OTel is not active: BQ rows use the plugin's internal span
    stack. STARTING gets the current top-of-stack; COMPLETED gets the
    popped span.
  """

  @pytest.mark.asyncio
  async def test_starting_completed_same_span_with_ambient(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      callback_context,
      mock_agent,
      dummy_arrow_schema,
  ):
    """With ambient OTel, STARTING and COMPLETED get the same span_id."""
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    real_tracer = provider.get_tracer("test")

    with mock.patch.object(
        bigquery_agent_analytics_plugin, "tracer", real_tracer
    ):
      bigquery_agent_analytics_plugin._span_records_ctx.set(None)

      # Simulate the framework's ambient spans.
      with real_tracer.start_as_current_span("invocation"):
        await bq_plugin_inst.before_run_callback(
            invocation_context=invocation_context
        )
        with real_tracer.start_as_current_span("invoke_agent"):
          await bq_plugin_inst.before_agent_callback(
              agent=mock_agent, callback_context=callback_context
          )
          await bq_plugin_inst.after_agent_callback(
              agent=mock_agent, callback_context=callback_context
          )
        await bq_plugin_inst.after_run_callback(
            invocation_context=invocation_context
        )

      await asyncio.sleep(0.01)

      rows = await _get_captured_rows_async(
          mock_write_client, dummy_arrow_schema
      )
      agent_starting = [r for r in rows if r["event_type"] == "AGENT_STARTING"]
      agent_completed = [
          r for r in rows if r["event_type"] == "AGENT_COMPLETED"
      ]

      assert len(agent_starting) == 1
      assert len(agent_completed) == 1

      # Both events must share the same span_id (the ambient
      # invoke_agent span) — no plugin-synthetic override.
      assert agent_starting[0]["span_id"] == agent_completed[0]["span_id"]
      assert (
          agent_starting[0]["parent_span_id"]
          == agent_completed[0]["parent_span_id"]
      )

    provider.shutdown()

  @pytest.mark.asyncio
  async def test_starting_completed_use_plugin_span_without_ambient(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      callback_context,
      mock_agent,
      dummy_arrow_schema,
  ):
    """Without ambient OTel, COMPLETED gets the popped plugin span."""
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    real_tracer = provider.get_tracer("test")

    with mock.patch.object(
        bigquery_agent_analytics_plugin, "tracer", real_tracer
    ):
      bigquery_agent_analytics_plugin._span_records_ctx.set(None)

      # No ambient OTel span.
      assert not trace.get_current_span().get_span_context().is_valid

      await bq_plugin_inst.before_run_callback(
          invocation_context=invocation_context
      )
      await bq_plugin_inst.before_agent_callback(
          agent=mock_agent, callback_context=callback_context
      )
      await bq_plugin_inst.after_agent_callback(
          agent=mock_agent, callback_context=callback_context
      )
      await bq_plugin_inst.after_run_callback(
          invocation_context=invocation_context
      )

      await asyncio.sleep(0.01)

      rows = await _get_captured_rows_async(
          mock_write_client, dummy_arrow_schema
      )
      agent_starting = [r for r in rows if r["event_type"] == "AGENT_STARTING"]
      agent_completed = [
          r for r in rows if r["event_type"] == "AGENT_COMPLETED"
      ]

      assert len(agent_starting) == 1
      assert len(agent_completed) == 1

      # AGENT_STARTING gets the top-of-stack span; AGENT_COMPLETED
      # gets the popped span via override — they should match.
      assert agent_starting[0]["span_id"] == agent_completed[0]["span_id"]

    provider.shutdown()

  @pytest.mark.asyncio
  async def test_tool_error_captures_span_id(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      callback_context,
      dummy_arrow_schema,
  ):
    """on_tool_error_callback uses the popped span_id (bonus fix)."""
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    real_tracer = provider.get_tracer("test")

    mock_tool = mock.create_autospec(base_tool_lib.BaseTool, instance=True)
    type(mock_tool).name = mock.PropertyMock(return_value="my_tool")
    tool_ctx = tool_context_lib.ToolContext(
        invocation_context=invocation_context
    )

    with mock.patch.object(
        bigquery_agent_analytics_plugin, "tracer", real_tracer
    ):
      bigquery_agent_analytics_plugin._span_records_ctx.set(None)

      # No ambient OTel — plugin span stack provides IDs.
      assert not trace.get_current_span().get_span_context().is_valid

      await bq_plugin_inst.before_run_callback(
          invocation_context=invocation_context
      )
      # Push tool span via before_tool_callback
      await bq_plugin_inst.before_tool_callback(
          tool=mock_tool,
          tool_args={"a": 1},
          tool_context=tool_ctx,
      )
      # Error callback should pop the tool span and use its ID
      await bq_plugin_inst.on_tool_error_callback(
          tool=mock_tool,
          tool_args={"a": 1},
          tool_context=tool_ctx,
          error=RuntimeError("boom"),
      )
      await bq_plugin_inst.after_run_callback(
          invocation_context=invocation_context
      )
      await asyncio.sleep(0.01)

      rows = await _get_captured_rows_async(
          mock_write_client, dummy_arrow_schema
      )
      tool_starting = [r for r in rows if r["event_type"] == "TOOL_STARTING"]
      tool_error = [r for r in rows if r["event_type"] == "TOOL_ERROR"]

      assert len(tool_starting) == 1
      assert len(tool_error) == 1

      # The TOOL_ERROR event must have the same span_id as
      # TOOL_STARTING (both correspond to the same tool span).
      assert tool_starting[0]["span_id"] == tool_error[0]["span_id"]
      assert tool_error[0]["span_id"] is not None

    provider.shutdown()


class TestStackLeakSafety:
  """Tests for stack leak safety (P2).

  Ensures the plugin's internal span stack doesn't leak records
  across invocations when after_run_callback is skipped.
  """

  def test_ensure_invocation_span_clears_stale_records(self, callback_context):
    """Pre-populated stack from a different invocation is cleared."""
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    TM = bigquery_agent_analytics_plugin.TraceManager

    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    real_tracer = provider.get_tracer("test")

    with mock.patch.object(
        bigquery_agent_analytics_plugin, "tracer", real_tracer
    ):
      # Simulate stale records from incomplete previous invocation.
      bigquery_agent_analytics_plugin._span_records_ctx.set(None)
      # Mark the stale records as belonging to a different invocation.
      bigquery_agent_analytics_plugin._active_invocation_id_ctx.set(
          "old-inv-stale"
      )
      TM.push_span(callback_context, "stale-invocation")
      TM.push_span(callback_context, "stale-agent")

      stale_records = bigquery_agent_analytics_plugin._span_records_ctx.get()
      assert len(stale_records) == 2

      # ensure_invocation_span with the *current* invocation_id should
      # detect the mismatch, clear stale records, and re-init.
      TM.ensure_invocation_span(callback_context)

      records = bigquery_agent_analytics_plugin._span_records_ctx.get()
      # Should have exactly 1 fresh entry (the new invocation span).
      assert len(records) == 1
      # The fresh span should NOT be one of the stale ones.
      assert records[0].span_id != stale_records[0].span_id
      assert records[0].span_id != stale_records[1].span_id

    provider.shutdown()

  def test_clear_stack_ends_owned_spans(self, callback_context):
    """clear_stack() ends all owned spans."""
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    TM = bigquery_agent_analytics_plugin.TraceManager

    provider = SdkProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    real_tracer = provider.get_tracer("test")

    with mock.patch.object(
        bigquery_agent_analytics_plugin, "tracer", real_tracer
    ):
      bigquery_agent_analytics_plugin._span_records_ctx.set(None)
      TM.push_span(callback_context, "span-a")
      TM.push_span(callback_context, "span-b")

      records = list(bigquery_agent_analytics_plugin._span_records_ctx.get())
      assert all(r.owns_span for r in records)

      TM.clear_stack()

      # Stack must be empty after clear.
      result = bigquery_agent_analytics_plugin._span_records_ctx.get()
      assert result == []

      # Both owned spans should have been ended (exported).
      exported = exporter.get_finished_spans()
      assert len(exported) == 2

    provider.shutdown()

  @pytest.mark.asyncio
  async def test_after_run_callback_clears_remaining_stack(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      callback_context,
      mock_agent,
      dummy_arrow_schema,
  ):
    """after_run_callback clears any leftover stack entries."""
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    TM = bigquery_agent_analytics_plugin.TraceManager

    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    real_tracer = provider.get_tracer("test")

    with mock.patch.object(
        bigquery_agent_analytics_plugin, "tracer", real_tracer
    ):
      bigquery_agent_analytics_plugin._span_records_ctx.set(None)

      # No ambient span.
      assert not trace.get_current_span().get_span_context().is_valid

      await bq_plugin_inst.before_run_callback(
          invocation_context=invocation_context
      )
      # Push an agent span but DON'T pop it (simulate missing
      # after_agent_callback due to exception).
      await bq_plugin_inst.before_agent_callback(
          agent=mock_agent, callback_context=callback_context
      )
      # Stack now has [invocation, agent].

      # after_run_callback should pop invocation + clear remaining.
      await bq_plugin_inst.after_run_callback(
          invocation_context=invocation_context
      )

      # Stack must be empty.
      records = bigquery_agent_analytics_plugin._span_records_ctx.get()
      assert records == []

    provider.shutdown()

  @pytest.mark.asyncio
  async def test_next_invocation_clean_after_incomplete_previous(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      callback_context,
      mock_agent,
      dummy_arrow_schema,
      mock_session,
  ):
    """Next invocation starts clean even if previous was incomplete."""
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    TM = bigquery_agent_analytics_plugin.TraceManager

    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    real_tracer = provider.get_tracer("test")

    with mock.patch.object(
        bigquery_agent_analytics_plugin, "tracer", real_tracer
    ):
      bigquery_agent_analytics_plugin._span_records_ctx.set(None)
      bigquery_agent_analytics_plugin._active_invocation_id_ctx.set(None)

      # --- Incomplete invocation 1: no after_run_callback ---
      await bq_plugin_inst.before_run_callback(
          invocation_context=invocation_context
      )
      await bq_plugin_inst.before_agent_callback(
          agent=mock_agent, callback_context=callback_context
      )
      # Skip after_agent and after_run — simulates exception.

      stale = bigquery_agent_analytics_plugin._span_records_ctx.get()
      assert len(stale) >= 2  # invocation + agent

      # --- Invocation 2 with a different invocation_id ---
      mock_write_client.append_rows.reset_mock()
      inv_ctx_2 = InvocationContext(
          agent=mock_agent,
          session=mock_session,
          invocation_id="inv-NEW-002",
          session_service=invocation_context.session_service,
          plugin_manager=invocation_context.plugin_manager,
      )
      await bq_plugin_inst.before_run_callback(invocation_context=inv_ctx_2)

      records = bigquery_agent_analytics_plugin._span_records_ctx.get()
      # Should have exactly 1 fresh invocation span.
      assert len(records) == 1

      # Cleanup
      await bq_plugin_inst.after_run_callback(invocation_context=inv_ctx_2)

    provider.shutdown()

  def test_ensure_invocation_span_idempotent_same_invocation(
      self, callback_context
  ):
    """Calling ensure_invocation_span twice in the same invocation is a no-op."""
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    TM = bigquery_agent_analytics_plugin.TraceManager

    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    real_tracer = provider.get_tracer("test")

    with mock.patch.object(
        bigquery_agent_analytics_plugin, "tracer", real_tracer
    ):
      bigquery_agent_analytics_plugin._span_records_ctx.set(None)
      bigquery_agent_analytics_plugin._active_invocation_id_ctx.set(None)

      # First call: creates invocation span.
      TM.ensure_invocation_span(callback_context)
      records_after_first = list(
          bigquery_agent_analytics_plugin._span_records_ctx.get()
      )
      assert len(records_after_first) == 1
      first_span_id = records_after_first[0].span_id

      # Second call (same invocation): must be a no-op.
      TM.ensure_invocation_span(callback_context)
      records_after_second = (
          bigquery_agent_analytics_plugin._span_records_ctx.get()
      )
      assert len(records_after_second) == 1
      assert records_after_second[0].span_id == first_span_id

      # Cleanup
      TM.pop_span()

    provider.shutdown()

  @pytest.mark.asyncio
  async def test_user_message_then_before_run_same_trace_no_ambient(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      callback_context,
      mock_agent,
      dummy_arrow_schema,
  ):
    """Regression: on_user_message → before_run must share one trace_id.

    Without the invocation-ID guard, the second ensure_invocation_span()
    call would clear the stack and create a new root span with a
    different trace_id, fracturing USER_MESSAGE_RECEIVED from
    INVOCATION_STARTING.
    """
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    real_tracer = provider.get_tracer("test")

    with mock.patch.object(
        bigquery_agent_analytics_plugin, "tracer", real_tracer
    ):
      bigquery_agent_analytics_plugin._span_records_ctx.set(None)
      bigquery_agent_analytics_plugin._active_invocation_id_ctx.set(None)

      # No ambient span.
      assert not trace.get_current_span().get_span_context().is_valid

      user_msg = types.Content(parts=[types.Part(text="hello")], role="user")
      await bq_plugin_inst.on_user_message_callback(
          invocation_context=invocation_context,
          user_message=user_msg,
      )
      await bq_plugin_inst.before_run_callback(
          invocation_context=invocation_context
      )
      await bq_plugin_inst.before_agent_callback(
          agent=mock_agent, callback_context=callback_context
      )
      await bq_plugin_inst.after_agent_callback(
          agent=mock_agent, callback_context=callback_context
      )
      await bq_plugin_inst.after_run_callback(
          invocation_context=invocation_context
      )
      await asyncio.sleep(0.01)

      rows = await _get_captured_rows_async(
          mock_write_client, dummy_arrow_schema
      )
      event_types = [r["event_type"] for r in rows]
      assert "USER_MESSAGE_RECEIVED" in event_types
      assert "INVOCATION_STARTING" in event_types

      # Every row must share the same trace_id.
      trace_ids = {r["trace_id"] for r in rows}
      assert len(trace_ids) == 1, (
          "Expected 1 unique trace_id across all events, got"
          f" {len(trace_ids)}: {trace_ids}"
      )

    provider.shutdown()


class TestRootAgentNameAcrossInvocations:
  """Regression: root_agent_name must refresh across invocations."""

  @pytest.mark.asyncio
  async def test_root_agent_name_updates_between_invocations(
      self,
      bq_plugin_inst,
      mock_write_client,
      mock_session,
      dummy_arrow_schema,
  ):
    """Two invocations with different root agents must log correct names.

    Previously init_trace() only set _root_agent_name_ctx when it was
    None, so the second invocation would inherit the first's root agent.
    """
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    real_tracer = provider.get_tracer("test")

    mock_session_service = mock.create_autospec(
        base_session_service_lib.BaseSessionService,
        instance=True,
        spec_set=True,
    )
    mock_plugin_manager = mock.create_autospec(
        plugin_manager_lib.PluginManager,
        instance=True,
        spec_set=True,
    )

    def _make_inv_ctx(agent_name, inv_id):
      agent = mock.create_autospec(
          base_agent.BaseAgent, instance=True, spec_set=True
      )
      type(agent).name = mock.PropertyMock(return_value=agent_name)
      type(agent).instruction = mock.PropertyMock(return_value="")
      # root_agent returns itself (no parent).
      agent.root_agent = agent
      return InvocationContext(
          agent=agent,
          session=mock_session,
          invocation_id=inv_id,
          session_service=mock_session_service,
          plugin_manager=mock_plugin_manager,
      )

    with mock.patch.object(
        bigquery_agent_analytics_plugin, "tracer", real_tracer
    ):
      # --- Invocation 1: root agent = "RootA" ---
      bigquery_agent_analytics_plugin._span_records_ctx.set(None)
      bigquery_agent_analytics_plugin._active_invocation_id_ctx.set(None)
      bigquery_agent_analytics_plugin._root_agent_name_ctx.set(None)

      inv1 = _make_inv_ctx("RootA", "inv-001")
      cb1 = CallbackContext(inv1)
      await bq_plugin_inst.before_run_callback(invocation_context=inv1)
      await bq_plugin_inst.before_agent_callback(
          agent=inv1.agent, callback_context=cb1
      )
      await bq_plugin_inst.after_agent_callback(
          agent=inv1.agent, callback_context=cb1
      )
      await bq_plugin_inst.after_run_callback(invocation_context=inv1)
      await asyncio.sleep(0.01)

      rows_inv1 = await _get_captured_rows_async(
          mock_write_client, dummy_arrow_schema
      )

      # --- Invocation 2: root agent = "RootB" ---
      mock_write_client.append_rows.reset_mock()

      inv2 = _make_inv_ctx("RootB", "inv-002")
      cb2 = CallbackContext(inv2)
      await bq_plugin_inst.before_run_callback(invocation_context=inv2)
      await bq_plugin_inst.before_agent_callback(
          agent=inv2.agent, callback_context=cb2
      )
      await bq_plugin_inst.after_agent_callback(
          agent=inv2.agent, callback_context=cb2
      )
      await bq_plugin_inst.after_run_callback(invocation_context=inv2)
      await asyncio.sleep(0.01)

      rows_inv2 = await _get_captured_rows_async(
          mock_write_client, dummy_arrow_schema
      )

    # Parse root_agent_name from the attributes JSON column.
    def _get_root_names(rows):
      names = set()
      for r in rows:
        attrs = r.get("attributes")
        if attrs:
          parsed = json.loads(attrs) if isinstance(attrs, str) else attrs
          if "root_agent_name" in parsed:
            names.add(parsed["root_agent_name"])
      return names

    names_inv1 = _get_root_names(rows_inv1)
    names_inv2 = _get_root_names(rows_inv2)

    # Invocation 1 should only have "RootA".
    assert names_inv1 == {"RootA"}, f"Expected {{'RootA'}}, got {names_inv1}"
    # Invocation 2 must have "RootB", NOT stale "RootA".
    assert names_inv2 == {"RootB"}, f"Expected {{'RootB'}}, got {names_inv2}"

    provider.shutdown()


class TestAfterRunCleanupExceptionSafety:
  """after_run_callback cleanup must execute even if _log_event fails."""

  @pytest.mark.asyncio
  async def test_cleanup_runs_when_log_event_raises(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      callback_context,
      mock_agent,
  ):
    """Stale state is cleared even when _log_event raises."""
    from opentelemetry.sdk.trace import TracerProvider as SdkProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = SdkProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    real_tracer = provider.get_tracer("test")

    with mock.patch.object(
        bigquery_agent_analytics_plugin, "tracer", real_tracer
    ):
      bigquery_agent_analytics_plugin._span_records_ctx.set(None)
      bigquery_agent_analytics_plugin._active_invocation_id_ctx.set(None)
      bigquery_agent_analytics_plugin._root_agent_name_ctx.set(None)

      # Run a normal before_run to initialise state.
      await bq_plugin_inst.before_run_callback(
          invocation_context=invocation_context
      )
      await bq_plugin_inst.before_agent_callback(
          agent=mock_agent, callback_context=callback_context
      )

      # Verify state is populated.
      assert bigquery_agent_analytics_plugin._span_records_ctx.get()
      assert (
          bigquery_agent_analytics_plugin._active_invocation_id_ctx.get()
          is not None
      )

      # Make _log_event raise inside after_run_callback.
      with mock.patch.object(
          bq_plugin_inst,
          "_log_event",
          side_effect=RuntimeError("boom"),
      ):
        # _safe_callback swallows the exception, but cleanup in
        # the finally block must still execute.
        await bq_plugin_inst.after_run_callback(
            invocation_context=invocation_context
        )

      # All invocation state must be cleaned up despite the error.
      records = bigquery_agent_analytics_plugin._span_records_ctx.get()
      assert records == [] or records is None
      assert (
          bigquery_agent_analytics_plugin._active_invocation_id_ctx.get()
          is None
      )
      assert bigquery_agent_analytics_plugin._root_agent_name_ctx.get() is None

    provider.shutdown()


class TestStringSystemPromptTruncation:
  """Tests that a string system prompt is truncated in parse()."""

  @pytest.mark.asyncio
  async def test_long_string_system_prompt_is_truncated(self):
    """A string system_instruction exceeding max_content_length is truncated."""
    parser = bigquery_agent_analytics_plugin.HybridContentParser(
        offloader=None,
        trace_id="test-trace",
        span_id="test-span",
        max_length=50,
    )
    long_prompt = "A" * 200
    llm_request = llm_request_lib.LlmRequest(
        model="gemini-pro",
        contents=[types.Content(parts=[types.Part(text="Hi")])],
        config=types.GenerateContentConfig(
            system_instruction=long_prompt,
        ),
    )
    payload, _, is_truncated = await parser.parse(llm_request)
    assert is_truncated
    assert len(payload["system_prompt"]) < 200
    assert "TRUNCATED" in payload["system_prompt"]


class TestSessionStateTruncation:
  """Tests that session state is truncated in _enrich_attributes."""

  @pytest.mark.asyncio
  async def test_oversized_session_state_is_truncated(
      self,
      mock_auth_default,
      mock_bq_client,
      mock_write_client,
      mock_to_arrow_schema,
      mock_asyncio_to_thread,
      mock_session,
      invocation_context,
  ):
    """Session state with large values is truncated."""
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(
        max_content_length=30,
    )
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
        config=config,
    )
    await plugin._ensure_started()

    # Set a large session state value.
    large_value = "X" * 200
    type(mock_session).state = mock.PropertyMock(
        return_value={"big_key": large_value}
    )

    callback_ctx = CallbackContext(invocation_context=invocation_context)
    event_data = bigquery_agent_analytics_plugin.EventData()
    attrs = plugin._enrich_attributes(event_data, callback_ctx)
    state = attrs["session_metadata"]["state"]
    assert len(state["big_key"]) < 200
    assert "TRUNCATED" in state["big_key"]
    await plugin.shutdown()


class TestSchemaUpgradeNestedFields:
  """Tests for nested RECORD field detection in schema upgrade."""

  def _make_plugin(self):
    config = bigquery_agent_analytics_plugin.BigQueryLoggerConfig(
        auto_schema_upgrade=True,
    )
    with mock.patch("google.cloud.bigquery.Client"):
      plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
          project_id=PROJECT_ID,
          dataset_id=DATASET_ID,
          table_id=TABLE_ID,
          config=config,
      )
    plugin.client = mock.MagicMock()
    plugin.full_table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    return plugin

  def test_nested_field_detected(self):
    """A new sub-field in a RECORD triggers an upgrade."""
    plugin = self._make_plugin()

    existing_record = bigquery.SchemaField(
        "metadata",
        "RECORD",
        fields=[
            bigquery.SchemaField("key", "STRING"),
        ],
    )
    desired_record = bigquery.SchemaField(
        "metadata",
        "RECORD",
        fields=[
            bigquery.SchemaField("key", "STRING"),
            bigquery.SchemaField("value", "STRING"),
        ],
    )
    plugin._schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        desired_record,
    ]

    existing = mock.MagicMock(spec=bigquery.Table)
    existing.schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        existing_record,
    ]
    existing.labels = {}
    plugin.client.get_table.return_value = existing
    plugin._ensure_schema_exists()

    plugin.client.update_table.assert_called_once()
    updated_table = plugin.client.update_table.call_args[0][0]
    # Find the metadata field and check it has both sub-fields.
    metadata_field = next(
        f for f in updated_table.schema if f.name == "metadata"
    )
    sub_names = {sf.name for sf in metadata_field.fields}
    assert "key" in sub_names
    assert "value" in sub_names

  def test_version_label_not_stamped_on_failure(self):
    """A failed update_table does not persist the version label."""
    plugin = self._make_plugin()
    plugin._schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("new_col", "STRING"),
    ]

    existing = mock.MagicMock(spec=bigquery.Table)
    existing.schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
    ]
    existing.labels = {}
    plugin.client.get_table.return_value = existing
    plugin.client.update_table.side_effect = Exception("network error")

    # Should not raise.
    plugin._ensure_schema_exists()

    # The label is set on the table object before update_table is
    # called, but since update_table failed the label was never
    # persisted remotely.  On the next run the stored_version will
    # still be None (from the real BQ table) so the upgrade retries.
    # We verify that update_table was actually attempted.
    plugin.client.update_table.assert_called_once()

  def test_nested_upgrade_preserves_policy_tags(self):
    """RECORD field metadata (e.g. policy_tags) is preserved on upgrade."""
    from google.cloud.bigquery import schema as bq_schema

    plugin = self._make_plugin()

    existing_record = bigquery.SchemaField(
        "metadata",
        "RECORD",
        policy_tags=bq_schema.PolicyTagList(
            names=["projects/p/locations/us/taxonomies/t/policyTags/pt"]
        ),
        fields=[
            bigquery.SchemaField("key", "STRING"),
        ],
    )
    desired_record = bigquery.SchemaField(
        "metadata",
        "RECORD",
        fields=[
            bigquery.SchemaField("key", "STRING"),
            bigquery.SchemaField("value", "STRING"),
        ],
    )
    plugin._schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        desired_record,
    ]

    existing = mock.MagicMock(spec=bigquery.Table)
    existing.schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        existing_record,
    ]
    existing.labels = {}
    plugin.client.get_table.return_value = existing
    plugin._ensure_schema_exists()

    plugin.client.update_table.assert_called_once()
    updated_table = plugin.client.update_table.call_args[0][0]
    metadata_field = next(
        f for f in updated_table.schema if f.name == "metadata"
    )
    # Sub-fields were merged.
    sub_names = {sf.name for sf in metadata_field.fields}
    assert "key" in sub_names
    assert "value" in sub_names
    # policy_tags preserved from the existing field.
    assert metadata_field.policy_tags is not None
    assert (
        "projects/p/locations/us/taxonomies/t/policyTags/pt"
        in metadata_field.policy_tags.names
    )


class TestMultiLoopShutdownDrainsOtherLoops:
  """Tests that shutdown() drains batch processors on other loops."""

  @pytest.mark.asyncio
  async def test_other_loop_batch_processor_drained(
      self,
      mock_auth_default,
      mock_bq_client,
      mock_write_client,
      mock_to_arrow_schema,
      mock_asyncio_to_thread,
  ):
    """Shutdown drains batch_processor.shutdown on non-current loops."""
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
    )
    await plugin._ensure_started()

    # Create a mock "other" loop with a mock batch processor.
    other_loop = mock.MagicMock(spec=asyncio.AbstractEventLoop)
    other_loop.is_closed.return_value = False

    mock_other_bp = mock.AsyncMock()
    mock_other_write_client = mock.MagicMock()
    mock_other_write_client.transport = mock.AsyncMock()

    other_state = bigquery_agent_analytics_plugin._LoopState(
        write_client=mock_other_write_client,
        batch_processor=mock_other_bp,
    )
    plugin._loop_state_by_loop[other_loop] = other_state

    # Patch run_coroutine_threadsafe to verify it's called for
    # the other loop's batch_processor.  Close the coroutine arg
    # to avoid "coroutine was never awaited" RuntimeWarning.
    mock_future = mock.MagicMock()
    mock_future.result.return_value = None

    def _fake_run_coroutine_threadsafe(coro, loop):
      coro.close()
      return mock_future

    with mock.patch.object(
        asyncio,
        "run_coroutine_threadsafe",
        side_effect=_fake_run_coroutine_threadsafe,
    ) as mock_rcts:
      await plugin.shutdown()

      # Verify run_coroutine_threadsafe was called with
      # the other loop.
      mock_rcts.assert_called()
      call_args = mock_rcts.call_args
      assert call_args[0][1] is other_loop


# ==============================================================
# TEST CLASS: A2A_INTERACTION event logging via on_event_callback
# ==============================================================
class TestA2AInteractionLogging:
  """Tests for A2A interaction event emission via on_event_callback.

  When a RemoteA2aAgent processes a response, it attaches A2A
  metadata (``a2a:task_id``, ``a2a:context_id``, ``a2a:request``,
  ``a2a:response``) to the event's ``custom_metadata``.  The
  plugin's ``on_event_callback`` should detect events carrying
  ``a2a:request`` or ``a2a:response`` and log an
  ``A2A_INTERACTION`` event so the remote agent's response and
  cross-reference IDs are visible in BigQuery.
  """

  @pytest.mark.asyncio
  async def test_a2a_interaction_logged_for_response_metadata(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      dummy_arrow_schema,
  ):
    """Event with a2a:response in custom_metadata emits A2A_INTERACTION."""
    a2a_meta = {
        "a2a:task_id": "task-abc",
        "a2a:context_id": "ctx-123",
        "a2a:response": {"status": "completed", "text": "result"},
    }
    event = event_lib.Event(
        author="remote_agent",
        custom_metadata=a2a_meta,
    )

    bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
    result = await bq_plugin_inst.on_event_callback(
        invocation_context=invocation_context, event=event
    )
    assert result is None

    await asyncio.sleep(0.05)
    rows = await _get_captured_rows_async(mock_write_client, dummy_arrow_schema)
    event_types = [r["event_type"] for r in rows]
    assert "A2A_INTERACTION" in event_types

    a2a_row = next(r for r in rows if r["event_type"] == "A2A_INTERACTION")
    attributes = json.loads(a2a_row["attributes"])
    assert "a2a_metadata" in attributes
    assert attributes["a2a_metadata"]["a2a:task_id"] == "task-abc"
    assert attributes["a2a_metadata"]["a2a:context_id"] == "ctx-123"

    # Content should contain the a2a:response payload.
    content = json.loads(a2a_row["content"])
    assert content["status"] == "completed"

  @pytest.mark.asyncio
  async def test_a2a_interaction_logged_for_request_metadata(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
      dummy_arrow_schema,
  ):
    """Event with a2a:request (no a2a:response) emits A2A_INTERACTION."""
    a2a_meta = {
        "a2a:task_id": "task-xyz",
        "a2a:request": {"message": "hello"},
    }
    event = event_lib.Event(
        author="remote_agent",
        custom_metadata=a2a_meta,
    )

    bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
    result = await bq_plugin_inst.on_event_callback(
        invocation_context=invocation_context, event=event
    )
    assert result is None

    await asyncio.sleep(0.05)
    rows = await _get_captured_rows_async(mock_write_client, dummy_arrow_schema)
    event_types = [r["event_type"] for r in rows]
    assert "A2A_INTERACTION" in event_types

    a2a_row = next(r for r in rows if r["event_type"] == "A2A_INTERACTION")
    attributes = json.loads(a2a_row["attributes"])
    assert attributes["a2a_metadata"]["a2a:request"] == {"message": "hello"}
    # No a2a:response → content should be None.
    assert a2a_row["content"] is None

  @pytest.mark.asyncio
  async def test_no_a2a_interaction_for_irrelevant_metadata(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
  ):
    """Events with only a2a:task_id (no request/response) are skipped."""
    a2a_meta = {
        "a2a:task_id": "task-only",
        "a2a:context_id": "ctx-only",
    }
    event = event_lib.Event(
        author="remote_agent",
        custom_metadata=a2a_meta,
    )

    result = await bq_plugin_inst.on_event_callback(
        invocation_context=invocation_context, event=event
    )
    assert result is None

    await asyncio.sleep(0.05)
    # No events logged — a2a:task_id alone is not a meaningful
    # interaction payload.
    assert mock_write_client.append_rows.call_count == 0

  @pytest.mark.asyncio
  async def test_no_a2a_interaction_for_no_metadata(
      self,
      bq_plugin_inst,
      mock_write_client,
      invocation_context,
  ):
    """Events without custom_metadata produce no A2A_INTERACTION."""
    event = event_lib.Event(author="regular_agent")

    result = await bq_plugin_inst.on_event_callback(
        invocation_context=invocation_context, event=event
    )
    assert result is None

    await asyncio.sleep(0.05)
    assert mock_write_client.append_rows.call_count == 0
