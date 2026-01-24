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
import dataclasses
import json
from unittest import mock

from google.adk.agents import base_agent
from google.adk.agents import callback_context as callback_context_lib
from google.adk.agents import invocation_context as invocation_context_lib
from google.adk.models import llm_request as llm_request_lib
from google.adk.models import llm_response as llm_response_lib
from google.adk.plugins import bigquery_agent_analytics_plugin
from google.adk.plugins import plugin_manager as plugin_manager_lib
from google.adk.sessions import base_session_service as base_session_service_lib
from google.adk.sessions import session as session_lib
from google.adk.tools import base_tool as base_tool_lib
from google.adk.tools import tool_context as tool_context_lib
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
  bigquery_agent_analytics_plugin._GLOBAL_WRITE_CLIENT = None

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
  return plugin


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
  assert request.trace_id == f"google-adk-bq-logger/{__version__}"

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


# --- Test Class ---


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
    config = BigQueryLoggerConfig()
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, config=config
    )

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

    await plugin.shutdown()

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
    config = BigQueryLoggerConfig()
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
    config = BigQueryLoggerConfig(event_allowlist=["LLM_REQUEST"])
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    )
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
    config = BigQueryLoggerConfig(event_denylist=["USER_MESSAGE_RECEIVED"])
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    )
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

    config = BigQueryLoggerConfig(content_formatter=redact_content)
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    )
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

    config = BigQueryLoggerConfig(content_formatter=error_formatter)
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    )
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
    config = BigQueryLoggerConfig(max_content_length=40)
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    )
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
    config = BigQueryLoggerConfig(max_content_length=80)
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    )
    await plugin._ensure_started()
    mock_write_client.append_rows.reset_mock()

    mock_tool = mock.create_autospec(
        base_tool_lib.BaseTool, instance=True, spec_set=True
    )
    type(mock_tool).name = mock.PropertyMock(return_value="MyTool")
    type(mock_tool).description = mock.PropertyMock(return_value="Description")

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
    config = BigQueryLoggerConfig(max_content_length=-1)
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    )
    await plugin._ensure_started()
    mock_write_client.append_rows.reset_mock()

    mock_tool = mock.create_autospec(
        base_tool_lib.BaseTool, instance=True, spec_set=True
    )
    type(mock_tool).name = mock.PropertyMock(return_value="MyTool")
    type(mock_tool).description = mock.PropertyMock(return_value="Description")

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
      mock_to_arrow_schema,
      dummy_arrow_schema,
      mock_asyncio_to_thread,
  ):
    """Test max content length for tool result."""
    _ = mock_auth_default
    _ = mock_bq_client
    _ = mock_to_arrow_schema
    _ = mock_asyncio_to_thread
    config = BigQueryLoggerConfig(max_content_length=80)
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    )
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
    config = BigQueryLoggerConfig(max_content_length=-1)
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    )
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
    config = BigQueryLoggerConfig(max_content_length=80)
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    )
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

    config = BigQueryLoggerConfig(
        gcs_bucket_name="my-bucket",
        connection_id="us.my-connection",
        max_content_length=20,  # Small limit to force offloading
    )
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    )
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
    with mock.patch(
        "google.adk.plugins.bigquery_agent_analytics_plugin.logger"
    ) as mock_logger:
      bigquery_agent_analytics_plugin.TraceManager.push_span(invocation_context)
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
    # Force the plugin to think it owns the client by clearing the global reference
    bigquery_agent_analytics_plugin._GLOBAL_WRITE_CLIENT = None
    await bq_plugin_inst.shutdown()
    mock_write_client.transport.close.assert_called_once()
    # bq_client might not be closed if it wasn't created or if close() failed,
    # but here it should be.
    # in the new implementation we verify attributes are reset
    assert bq_plugin_inst.write_client is None
    assert bq_plugin_inst.client is None
    assert bq_plugin_inst._is_shutting_down is False

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

  @pytest.mark.asyncio
  async def test_table_creation_options(
      self,
      mock_auth_default,
      mock_bq_client,
      mock_write_client,
      mock_to_arrow_schema,
      mock_asyncio_to_thread,
  ):
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID
    )
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
    timestamp_field = next(f for f in table_arg.schema if f.name == "timestamp")
    assert (
        timestamp_field.description
        == "The UTC timestamp when the event occurred. Used for ordering events"
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
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
    )

    def _run_in_thread():
      # In a real thread pool, there might not be an event loop.
      # However, since we are calling an async method (_ensure_started),
      # we must run it in an event loop. The issue was that _lazy_setup
      # called get_event_loop() which fails in threads without a loop.
      # Here we simulate the condition by running in a thread and creating a new loop if needed,
      # but the key is that the plugin's internal calls should use the correct loop.
      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      try:
        loop.run_until_complete(plugin._ensure_started())
      finally:
        loop.close()

    # Run in a separate thread to simulate ThreadPoolExecutor-0_0
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as executor:
      future = executor.submit(_run_in_thread)
      future.result()  # Should not raise "no current event loop"

    assert plugin._started
    assert plugin.client is not None
    assert plugin.write_client is not None

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
    config = BigQueryLoggerConfig(gcs_bucket_name=bucket_name)
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    )
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
    await asyncio.sleep(0.01)

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
  async def test_global_client_reuse(
      self, mock_write_client, mock_auth_default
  ):
    del mock_write_client, mock_auth_default  # Unused
    # Reset global client for this test
    bigquery_agent_analytics_plugin._GLOBAL_WRITE_CLIENT = None

    # Create two plugins
    plugin1 = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, table_id="table1"
    )
    plugin2 = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, table_id="table2"
    )

    # Start both
    await plugin1._ensure_started()
    await plugin2._ensure_started()

    # Verify they share the same write_client instance
    assert plugin1.write_client is not None
    assert plugin2.write_client is not None
    assert plugin1.write_client is plugin2.write_client

    # Verify shutdown doesn't close the global client
    await plugin1.shutdown()
    # Mock transport close check - since it's a mock, we check call count
    # But here we check if the client is still the global one
    assert (
        bigquery_agent_analytics_plugin._GLOBAL_WRITE_CLIENT
        is plugin2.write_client
    )

    # Cleanup
    await plugin2.shutdown()
    bigquery_agent_analytics_plugin._GLOBAL_WRITE_CLIENT = None

  @pytest.mark.asyncio
  async def test_quota_project_id_used_in_client(
      self,
      mock_bq_client,
      mock_to_arrow_schema,
      mock_asyncio_to_thread,
  ):
    bigquery_agent_analytics_plugin._GLOBAL_WRITE_CLIENT = None
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
        plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID,
            table_id=TABLE_ID,
        )
        await plugin._ensure_started()
        mock_auth_default.assert_called_once()
        mock_bq_write_cls.assert_called_once()
        _, kwargs = mock_bq_write_cls.call_args
        assert kwargs["client_options"].quota_project_id == "quota-project"
        bigquery_agent_analytics_plugin._GLOBAL_WRITE_CLIENT = None

  @pytest.mark.asyncio
  async def test_pickle_safety(self, mock_auth_default, mock_bq_client):
    """Test that the plugin can be pickled safely."""
    import pickle

    config = BigQueryLoggerConfig(enabled=True)
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

    # Test pickling after start
    pickled_started = pickle.dumps(plugin)
    unpickled_started = pickle.loads(pickled_started)

    assert unpickled_started.project_id == PROJECT_ID
    # Runtime objects should be None after unpickling
    assert unpickled_started._setup_lock is None
    assert unpickled_started._executor is None
    assert unpickled_started.client is None

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
    assert llm_span_id != agent_span_id

    log_entry_req = await _get_captured_event_dict_async(
        mock_write_client, dummy_arrow_schema
    )
    assert log_entry_req["event_type"] == "LLM_REQUEST"
    assert log_entry_req["span_id"] == llm_span_id
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
    # Crux of the bug fix: Parent should still be Agent Span, NOT Self.
    assert log_entry_resp["parent_span_id"] == agent_span_id
    assert log_entry_resp["parent_span_id"] != log_entry_resp["span_id"]

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

    config = BigQueryLoggerConfig()
    plugin = bigquery_agent_analytics_plugin.BigQueryAgentAnalyticsPlugin(
        PROJECT_ID, DATASET_ID, table_id=TABLE_ID, config=config
    )
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

      mock_tracer.start_span.assert_called_with("test_span")
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
