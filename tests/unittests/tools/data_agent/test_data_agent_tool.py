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

import inspect
from unittest import mock

from google.adk.tools.data_agent import data_agent_tool
from google.adk.tools.data_agent.config import DataAgentToolConfig
from google.adk.tools.tool_context import ToolContext
import pytest
import requests


@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
def test_list_accessible_data_agents_success(mock_get_session):
  """Tests list_accessible_data_agents success path."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_response = mock.Mock()
  mock_response.json.return_value = {"dataAgents": ["agent1", "agent2"]}
  mock_response.raise_for_status.return_value = None
  mock_session.get.return_value = mock_response
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.googleapis.com",
  )
  result = data_agent_tool.list_accessible_data_agents(
      "test-project", mock_creds
  )
  assert result["status"] == "SUCCESS"
  assert result["response"] == ["agent1", "agent2"]
  mock_get_session.assert_called_once_with(mock_creds, location="global")
  mock_session.get.assert_called_once_with(
      "https://geminidataanalytics.googleapis.com/v1/projects/test-project/locations/global/dataAgents:listAccessible",
      headers={
          "Content-Type": "application/json",
          "X-Goog-API-Client": "GOOGLE_ADK",
      },
      timeout=mock.ANY,
  )


@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
def test_list_accessible_data_agents_exception(mock_get_session):
  """Tests list_accessible_data_agents exception path."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_session.get.side_effect = Exception("List failed!")
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.googleapis.com",
  )
  result = data_agent_tool.list_accessible_data_agents(
      "test-project", mock_creds
  )
  assert result["status"] == "ERROR"
  assert "List failed!" in result["error_details"]
  mock_get_session.assert_called_once_with(mock_creds, location="global")
  mock_session.get.assert_called_once()


@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_endpoint", autospec=True
)
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
def test_get_data_agent_info_success(mock_get_session, mock_get_endpoint):
  """Tests get_data_agent_info success path."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_response = mock.Mock()
  mock_response.json.return_value = "agent_info"
  mock_response.raise_for_status.return_value = None
  mock_session.get.return_value = mock_response
  mock_get_endpoint.return_value = "https://geminidataanalytics.googleapis.com"
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.googleapis.com",
  )
  result = data_agent_tool.get_data_agent_info("agent_name", mock_creds)
  assert result["status"] == "SUCCESS"
  assert result["response"] == "agent_info"
  mock_get_session.assert_called_once_with(mock_creds)
  mock_get_endpoint.assert_called_once()
  mock_session.get.assert_called_once_with(
      "https://geminidataanalytics.googleapis.com/v1/agent_name",
      headers={
          "Content-Type": "application/json",
          "X-Goog-API-Client": "GOOGLE_ADK",
      },
      timeout=mock.ANY,
  )


@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_endpoint", autospec=True
)
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
def test_get_data_agent_info_exception(mock_get_session, mock_get_endpoint):
  """Tests get_data_agent_info exception path."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_session.get.side_effect = Exception("Get failed!")
  mock_get_endpoint.return_value = "https://geminidataanalytics.googleapis.com"
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.googleapis.com",
  )
  result = data_agent_tool.get_data_agent_info("agent_name", mock_creds)
  assert result["status"] == "ERROR"
  assert "Get failed!" in result["error_details"]
  mock_get_session.assert_called_once_with(mock_creds)
  mock_get_endpoint.assert_called_once()
  mock_session.get.assert_called_once()


@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_stream", autospec=True
)
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
@mock.patch.object(data_agent_tool, "_get_data_agent_info", autospec=True)
def test_ask_data_agent_success(
    mock_get_agent_info, mock_get_session, mock_get_stream
):
  """Tests ask_data_agent success path."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.googleapis.com",
  )
  mock_get_agent_info.return_value = {"status": "SUCCESS", "response": {}}
  mock_get_stream.return_value = [
      {"text": {"parts": ["response1"], "textType": "THOUGHT"}},
      {"text": {"parts": ["response2"], "textType": "FINAL_RESPONSE"}},
  ]
  mock_invocation_context = mock.Mock()
  mock_invocation_context.session.state = {}
  mock_context = ToolContext(mock_invocation_context)
  mock_settings = mock.Mock()

  result = data_agent_tool.ask_data_agent(
      "projects/p/locations/l/dataAgents/a",
      "query",
      credentials=mock_creds,
      tool_context=mock_context,
      settings=mock_settings,
  )
  assert result["status"] == "SUCCESS"
  assert result["response"] == [
      {"text": {"parts": ["response1"], "textType": "THOUGHT"}},
      {"text": {"parts": ["response2"], "textType": "FINAL_RESPONSE"}},
  ]
  mock_get_agent_info.assert_called_once_with(
      "projects/p/locations/l/dataAgents/a",
      mock_creds,
      session=mock_session,
      settings=mock_settings,
  )
  mock_get_session.assert_called_once_with(mock_creds, location="l")
  mock_get_stream.assert_called_once_with(
      mock_session,
      "https://geminidataanalytics.googleapis.com/v1/projects/p/locations/l:chat",
      {
          "messages": [{"userMessage": {"text": "query"}}],
          "dataAgentContext": {
              "dataAgent": "projects/p/locations/l/dataAgents/a",
          },
          "clientIdEnum": "GOOGLE_ADK",
      },
      {
          "Content-Type": "application/json",
          "X-Goog-API-Client": "GOOGLE_ADK",
      },
      mock_settings.max_query_result_rows,
  )


@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_stream", autospec=True
)
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
@mock.patch.object(data_agent_tool, "_get_data_agent_info", autospec=True)
def test_ask_data_agent_exception(
    mock_get_agent_info, mock_get_session, mock_get_stream
):
  """Tests ask_data_agent exception path."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.googleapis.com",
  )
  mock_get_agent_info.return_value = {"status": "SUCCESS", "response": {}}
  mock_get_stream.side_effect = Exception("Chat failed!")
  mock_invocation_context = mock.Mock()
  mock_invocation_context.session.state = {}
  mock_context = ToolContext(mock_invocation_context)
  mock_settings = mock.Mock()

  result = data_agent_tool.ask_data_agent(
      "projects/p/locations/l/dataAgents/a",
      "query",
      credentials=mock_creds,
      tool_context=mock_context,
      settings=mock_settings,
  )
  assert result["status"] == "ERROR"
  assert "Chat failed!" in result["error_details"]
  mock_get_session.assert_called_once_with(mock_creds, location="l")
  mock_get_stream.assert_called_once()


def test_extract_location_from_resource_name():
  """Tests location extraction helper function."""
  extract = data_agent_tool._extract_location_from_resource_name
  assert extract("projects/p/locations/eu/dataAgents/agent_1") == "eu"
  assert extract("projects/p/locations/us/dataAgents/agent_2") == "us"
  assert extract("projects/p/locations/global/dataAgents/agent_3") == "global"
  assert extract("invalid_name") is None


@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_endpoint", autospec=True
)
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
def test_get_data_agent_info_auto_extract_location(
    mock_get_session, mock_get_endpoint
):
  """Tests automatic location extraction from resource name when settings location is None."""

  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_response = mock.Mock()
  mock_response.json.return_value = {"name": "agent_eu"}
  mock_session.get.return_value = mock_response
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.eu.rep.googleapis.com",
  )
  mock_get_endpoint.return_value = (
      "https://geminidataanalytics.eu.rep.googleapis.com"
  )

  settings = DataAgentToolConfig(location=None)
  result = data_agent_tool._get_data_agent_info(
      "projects/my-proj/locations/eu/dataAgents/my-agent",
      mock_creds,
      settings=settings,
  )

  mock_get_endpoint.assert_called_once_with(location="eu")
  mock_get_session.assert_called_once_with(mock_creds, location="eu")
  assert result["status"] == "SUCCESS"


@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
def test_list_accessible_data_agents_regional(mock_get_session):
  """Tests list_accessible_data_agents with regional settings."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_response = mock.Mock()
  mock_response.json.return_value = {"dataAgents": ["agent_eu"]}
  mock_response.raise_for_status.return_value = None
  mock_session.get.return_value = mock_response
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.eu.rep.googleapis.com",
  )
  settings = DataAgentToolConfig(location="eu")
  result = data_agent_tool.list_accessible_data_agents(
      "test-project", mock_creds, settings=settings
  )
  assert result["status"] == "SUCCESS"
  assert result["response"] == ["agent_eu"]
  mock_get_session.assert_called_once_with(mock_creds, location="eu")
  mock_session.get.assert_called_once_with(
      "https://geminidataanalytics.eu.rep.googleapis.com/v1/projects/test-project/locations/eu/dataAgents:listAccessible",
      headers={
          "Content-Type": "application/json",
          "X-Goog-API-Client": "GOOGLE_ADK",
      },
      timeout=mock.ANY,
  )


@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
def test_list_accessible_data_agents_explicit_location(mock_get_session):
  """Tests list_accessible_data_agents with explicit location parameter overriding settings."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_response = mock.Mock()
  mock_response.json.return_value = {"dataAgents": ["agent_us"]}
  mock_response.raise_for_status.return_value = None
  mock_session.get.return_value = mock_response
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.us.rep.googleapis.com",
  )
  settings = DataAgentToolConfig(location="eu")
  result = data_agent_tool.list_accessible_data_agents(
      "test-project", mock_creds, location="us", settings=settings
  )
  assert result["status"] == "SUCCESS"
  assert result["response"] == ["agent_us"]
  mock_get_session.assert_called_once_with(mock_creds, location="us")
  mock_session.get.assert_called_once_with(
      "https://geminidataanalytics.us.rep.googleapis.com/v1/projects/test-project/locations/us/dataAgents:listAccessible",
      headers={
          "Content-Type": "application/json",
          "X-Goog-API-Client": "GOOGLE_ADK",
      },
      timeout=mock.ANY,
  )


def test_list_accessible_data_agents_invalid_location():
  """Tests list_accessible_data_agents with invalid location segment."""
  mock_creds = mock.Mock()
  result = data_agent_tool.list_accessible_data_agents(
      "test-project", mock_creds, location="invalid/segment"
  )
  assert result["status"] == "ERROR"
  assert "Invalid location format" in result["error_details"]


def test_list_accessible_data_agents_invalid_project_id():
  """Tests list_accessible_data_agents with invalid project_id segment."""
  mock_creds = mock.Mock()
  result = data_agent_tool.list_accessible_data_agents(
      "invalid/project", mock_creds
  )
  assert result["status"] == "ERROR"
  assert "Invalid project_id format" in result["error_details"]


class _FakeClock:
  """Virtual clock: only asyncio.sleep advances time, so tests run instantly."""

  def __init__(self):
    self.now = 0.0

  def monotonic(self) -> float:
    return self.now

  async def sleep(self, seconds: float) -> None:
    self.now += seconds


@pytest.fixture
def fake_clock():
  clock = _FakeClock()
  with (
      mock.patch.object(
          data_agent_tool.time, "monotonic", side_effect=clock.monotonic
      ),
      mock.patch.object(
          data_agent_tool.asyncio, "sleep", side_effect=clock.sleep
      ),
  ):
    yield clock


@pytest.mark.asyncio
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
async def test_create_data_agent_success(mock_get_session):
  """Tests create_data_agent success path."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_response = mock.Mock()
  mock_response.ok = True
  mock_response.json.return_value = {"name": "agent1"}
  mock_session.post.return_value = mock_response
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.googleapis.com",
  )
  mock_settings = mock.Mock()
  mock_settings.enable_data_agent_modification = True
  mock_settings.data_agent_modification_timeout_seconds = 60
  mock_settings.data_agent_modification_poll_interval_seconds = 2

  result = await data_agent_tool.create_data_agent(
      "test-project",
      "new-agent",
      '{"displayName": "test"}',
      location="us-central1",
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert result["status"] == "SUCCESS"
  assert result["response"] == {"name": "agent1"}
  mock_get_session.assert_called_once_with(mock_creds, location="us-central1")
  mock_session.post.assert_called_once_with(
      "https://geminidataanalytics.googleapis.com/v1/projects/test-project/locations/us-central1/dataAgents",
      params={"dataAgentId": "new-agent"},
      json={"displayName": "test"},
      headers={
          "Content-Type": "application/json",
          "X-Goog-API-Client": "GOOGLE_ADK",
      },
      timeout=mock.ANY,
  )


@pytest.mark.asyncio
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
async def test_create_data_agent_non_2xx(mock_get_session):
  """Tests create_data_agent non-2xx error path."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_response = mock.Mock()
  mock_response.ok = False
  mock_response.status_code = 400
  mock_response.text = "Bad Request"
  mock_session.post.return_value = mock_response
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.googleapis.com",
  )
  mock_settings = mock.Mock()
  mock_settings.enable_data_agent_modification = True
  mock_settings.data_agent_modification_timeout_seconds = 60
  mock_settings.data_agent_modification_poll_interval_seconds = 2

  result = await data_agent_tool.create_data_agent(
      "test-project",
      "new-agent",
      '{"displayName": "test"}',
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert result["status"] == "ERROR"
  assert "API returned error status: 400 Bad Request" in result["error_details"]


@pytest.mark.asyncio
async def test_create_data_agent_malformed_config():
  """Tests create_data_agent with malformed JSON agent_config."""
  mock_creds = mock.Mock()
  mock_settings = mock.Mock()
  mock_settings.enable_data_agent_modification = True

  result = await data_agent_tool.create_data_agent(
      "test-project",
      "new-agent",
      "invalid-json",
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert result["status"] == "ERROR"
  assert "Invalid agent_config:" in result["error_details"]


@pytest.mark.asyncio
async def test_create_data_agent_non_dict_config():
  """Tests create_data_agent with JSON string that is not a dict."""
  mock_creds = mock.Mock()
  mock_settings = mock.Mock()
  mock_settings.enable_data_agent_modification = True

  result = await data_agent_tool.create_data_agent(
      "test-project",
      "new-agent",
      "[1, 2]",
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert result["status"] == "ERROR"
  assert "agent_config must be a dictionary" in result["error_details"]


@pytest.mark.asyncio
async def test_create_data_agent_creation_disabled():
  """Tests create_data_agent when creation is disabled."""
  mock_creds = mock.Mock()
  mock_settings = mock.Mock()
  mock_settings.enable_data_agent_modification = False

  result = await data_agent_tool.create_data_agent(
      "test-project",
      "new-agent",
      '{"displayName": "test"}',
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert result["status"] == "ERROR"
  assert "Data agent mutation is disabled" in result["error_details"]


@pytest.mark.asyncio
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
async def test_create_data_agent_exception(mock_get_session):
  """Tests create_data_agent exception path."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_session.post.side_effect = Exception("Post failed!")
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.googleapis.com",
  )
  mock_settings = mock.Mock()
  mock_settings.enable_data_agent_modification = True
  mock_settings.data_agent_modification_timeout_seconds = 60
  mock_settings.data_agent_modification_poll_interval_seconds = 2

  result = await data_agent_tool.create_data_agent(
      "test-project",
      "new-agent",
      '{"displayName": "test"}',
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert result["status"] == "ERROR"
  assert "Post failed!" in result["error_details"]


def test_create_data_agent_is_coroutine_function():
  """Verifies create_data_agent is an async coroutine function."""
  assert inspect.iscoroutinefunction(data_agent_tool.create_data_agent)


@pytest.mark.asyncio
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
async def test_create_data_agent_lro_polls_until_done(
    mock_get_session, fake_clock
):
  """Tests create_data_agent LRO polling until operation completes."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()

  post_resp = mock.Mock(ok=True)
  post_resp.json.return_value = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": False,
  }
  mock_session.post.return_value = post_resp

  poll_resp_1 = mock.Mock(ok=True)
  poll_resp_1.json.return_value = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": False,
  }
  poll_resp_2 = mock.Mock(ok=True)
  poll_resp_2.json.return_value = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": True,
      "response": {"name": "projects/p/locations/g/dataAgents/new-agent"},
  }
  mock_session.get.side_effect = [poll_resp_1, poll_resp_2]

  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.googleapis.com",
  )
  mock_settings = mock.Mock(
      enable_data_agent_modification=True,
      data_agent_modification_timeout_seconds=60,
      data_agent_modification_poll_interval_seconds=2,
  )

  result = await data_agent_tool.create_data_agent(
      "p",
      "new-agent",
      '{"displayName": "test"}',
      credentials=mock_creds,
      settings=mock_settings,
  )

  assert result["status"] == "SUCCESS"
  assert result["response"] == {
      "name": "projects/p/locations/g/dataAgents/new-agent"
  }
  assert mock_session.get.call_count == 2
  mock_session.get.assert_called_with(
      "https://geminidataanalytics.googleapis.com/v1/projects/p/locations/g/operations/op-1",
      headers={
          "Content-Type": "application/json",
          "X-Goog-API-Client": "GOOGLE_ADK",
      },
      timeout=mock.ANY,
  )


@pytest.mark.asyncio
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
async def test_create_data_agent_accepts_dict_from_programmatic_caller(
    mock_get_session,
):
  """Tests create_data_agent accepts dict from programmatic Python callers or AI middleware."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_response = mock.Mock(ok=True)
  mock_response.json.return_value = {"name": "agent1", "done": True}
  mock_session.post.return_value = mock_response
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.googleapis.com",
  )
  mock_settings = mock.Mock(
      enable_data_agent_modification=True,
      data_agent_modification_timeout_seconds=60,
      data_agent_modification_poll_interval_seconds=2,
  )

  result = await data_agent_tool.create_data_agent(
      "p",
      "new-agent",
      {"displayName": "test"},
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert result["status"] == "SUCCESS"
  mock_session.post.assert_called_once_with(
      "https://geminidataanalytics.googleapis.com/v1/projects/p/locations/global/dataAgents",
      params={"dataAgentId": "new-agent"},
      json={"displayName": "test"},
      headers=mock.ANY,
      timeout=mock.ANY,
  )


# ==============================================================================
# LRO Polling (_await_lro) Unit Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_await_lro_returns_immediately_when_done():
  """Tests _await_lro returns immediately when initial operation has done=True."""
  mock_session = mock.MagicMock()
  op = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": True,
      "response": {"name": "projects/p/locations/g/dataAgents/agent-1"},
  }
  result = await data_agent_tool._await_lro(
      session=mock_session,
      base_url="https://example.com/v1",
      headers={},
      resp=mock.Mock(ok=True, json=mock.Mock(return_value=op)),
      deadline=100.0,
      poll_interval=2.0,
      total_timeout=60.0,
  )
  assert result["status"] == "SUCCESS"
  assert result["response"] == {
      "name": "projects/p/locations/g/dataAgents/agent-1"
  }
  mock_session.get.assert_not_called()


@pytest.mark.asyncio
async def test_await_lro_non_operation_name_returns_immediately():
  """Tests _await_lro returns immediately when resource is not an operation."""
  mock_session = mock.MagicMock()
  resource = {"name": "projects/p/locations/g/dataAgents/agent-1"}
  result = await data_agent_tool._await_lro(
      session=mock_session,
      base_url="https://example.com/v1",
      headers={},
      resp=mock.Mock(ok=True, json=mock.Mock(return_value=resource)),
      deadline=100.0,
      poll_interval=2.0,
      total_timeout=60.0,
  )
  assert result["status"] == "SUCCESS"
  assert result["response"] == resource
  mock_session.get.assert_not_called()


@pytest.mark.asyncio
async def test_await_lro_polls_until_done(fake_clock):
  """Tests _await_lro polling until operation completes successfully."""
  mock_session = mock.MagicMock()
  poll_resp_1 = mock.Mock(ok=True)
  poll_resp_1.json.return_value = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": False,
  }
  poll_resp_2 = mock.Mock(ok=True)
  poll_resp_2.json.return_value = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": True,
      "response": {"name": "projects/p/locations/g/dataAgents/agent-1"},
  }
  mock_session.get.side_effect = [poll_resp_1, poll_resp_2]
  op = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": False,
  }
  result = await data_agent_tool._await_lro(
      session=mock_session,
      base_url="https://geminidataanalytics.googleapis.com/v1",
      headers={"X-Test": "1"},
      resp=mock.Mock(ok=True, json=mock.Mock(return_value=op)),
      deadline=100.0,
      poll_interval=0.1,
      total_timeout=60.0,
  )
  assert result["status"] == "SUCCESS"
  assert result["response"] == {
      "name": "projects/p/locations/g/dataAgents/agent-1"
  }
  assert mock_session.get.call_count == 2
  mock_session.get.assert_called_with(
      "https://geminidataanalytics.googleapis.com/v1/projects/p/locations/g/operations/op-1",
      headers={"X-Test": "1"},
      timeout=mock.ANY,
  )


@pytest.mark.asyncio
async def test_await_lro_operation_error(fake_clock):
  """Tests _await_lro returning ERROR when operation finishes with error."""
  mock_session = mock.MagicMock()
  poll_resp = mock.Mock(ok=True)
  poll_resp.json.return_value = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": True,
      "error": {"code": 400, "message": "Mutation invalid"},
  }
  mock_session.get.return_value = poll_resp
  op = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": False,
  }
  result = await data_agent_tool._await_lro(
      session=mock_session,
      base_url="https://geminidataanalytics.googleapis.com/v1",
      headers={},
      resp=mock.Mock(ok=True, json=mock.Mock(return_value=op)),
      deadline=100.0,
      poll_interval=0.1,
      total_timeout=60.0,
  )
  assert result["status"] == "ERROR"
  assert "Mutation invalid" in result["error_details"]
  assert result["operation_name"] == "projects/p/locations/g/operations/op-1"


@pytest.mark.asyncio
async def test_await_lro_poll_http_error(fake_clock):
  """Tests _await_lro handling non-retryable HTTP error during polling."""
  mock_session = mock.MagicMock()
  poll_resp = mock.Mock(ok=False, status_code=400, text="Bad Request")
  mock_session.get.return_value = poll_resp
  op = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": False,
  }
  result = await data_agent_tool._await_lro(
      session=mock_session,
      base_url="https://geminidataanalytics.googleapis.com/v1",
      headers={},
      resp=mock.Mock(ok=True, json=mock.Mock(return_value=op)),
      deadline=100.0,
      poll_interval=0.1,
      total_timeout=60.0,
  )
  assert result["status"] == "ERROR"
  assert (
      "Polling failed with status: 400 Bad Request" in result["error_details"]
  )
  assert result["operation_name"] == "projects/p/locations/g/operations/op-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
async def test_await_lro_retryable_http_error_recovers(fake_clock, code):
  """Tests _await_lro retrying on retryable HTTP error and recovering on next poll."""
  mock_session = mock.MagicMock()
  poll_resp_1 = mock.Mock(ok=False, status_code=code, text="Retryable Error")
  poll_resp_2 = mock.Mock(ok=True)
  poll_resp_2.json.return_value = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": True,
      "response": {"name": "projects/p/locations/g/dataAgents/agent-1"},
  }
  mock_session.get.side_effect = [poll_resp_1, poll_resp_2]

  op = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": False,
  }
  result = await data_agent_tool._await_lro(
      session=mock_session,
      base_url="https://geminidataanalytics.googleapis.com/v1",
      headers={},
      resp=mock.Mock(ok=True, json=mock.Mock(return_value=op)),
      deadline=100.0,
      poll_interval=0.1,
      total_timeout=60.0,
  )
  assert result["status"] == "SUCCESS"
  assert result["response"] == {
      "name": "projects/p/locations/g/dataAgents/agent-1"
  }


@pytest.mark.asyncio
async def test_await_lro_connection_error_retries_and_recovers(fake_clock):
  """Tests _await_lro retrying on ConnectionError and recovering on next poll."""
  mock_session = mock.MagicMock()
  poll_resp_2 = mock.Mock(ok=True)
  poll_resp_2.json.return_value = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": True,
      "response": {"name": "projects/p/locations/g/dataAgents/agent-1"},
  }
  mock_session.get.side_effect = [
      requests.ConnectionError("Temporary network failure"),
      poll_resp_2,
  ]

  op = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": False,
  }
  result = await data_agent_tool._await_lro(
      session=mock_session,
      base_url="https://geminidataanalytics.googleapis.com/v1",
      headers={},
      resp=mock.Mock(ok=True, json=mock.Mock(return_value=op)),
      deadline=100.0,
      poll_interval=0.1,
      total_timeout=60.0,
  )
  assert result["status"] == "SUCCESS"
  assert result["response"] == {
      "name": "projects/p/locations/g/dataAgents/agent-1"
  }
  assert mock_session.get.call_count == 2


@pytest.mark.asyncio
async def test_await_lro_poll_invalid_json(fake_clock):
  """Tests _await_lro returns ERROR when polling response is invalid JSON."""
  mock_session = mock.MagicMock()
  poll_resp = mock.Mock(ok=True)
  poll_resp.json.side_effect = ValueError("Expecting value")
  mock_session.get.return_value = poll_resp

  op = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": False,
  }
  result = await data_agent_tool._await_lro(
      session=mock_session,
      base_url="https://geminidataanalytics.googleapis.com/v1",
      headers={},
      resp=mock.Mock(ok=True, json=mock.Mock(return_value=op)),
      deadline=100.0,
      poll_interval=0.1,
      total_timeout=60.0,
  )
  assert result["status"] == "ERROR"
  assert "Polling returned invalid JSON" in result["error_details"]
  assert result["operation_name"] == "projects/p/locations/g/operations/op-1"


@pytest.mark.parametrize(
    "bad_name",
    [
        "agent-1",
        "projects/p/locations/g/dataAgents/",
        "projects/p/locations/g/dataAgents/a/extra",
        "projects/p/locations/g/dataAgents/a\n",
        "projects/p/locations/g/dataAgents/..",
        "projects/../locations/../dataAgents/x",
    ],
)
def test_validate_data_agent_name_invalid(bad_name):
  """Tests _validate_data_agent_name rejects invalid resource names."""
  err = data_agent_tool._validate_data_agent_name(bad_name)
  assert err is not None
  assert err["status"] == "ERROR"
  assert "Invalid data_agent_name format" in err["error_details"]


@pytest.mark.asyncio
async def test_await_lro_unpollable_operation_not_done_returns_error():
  """Tests _await_lro returns ERROR when operation is not pollable and not done."""
  mock_session = mock.MagicMock()
  op = {
      "name": "invalid-op-name",
      "done": False,
  }
  result = await data_agent_tool._await_lro(
      session=mock_session,
      base_url="https://geminidataanalytics.googleapis.com/v1",
      headers={},
      resp=mock.Mock(ok=True, json=mock.Mock(return_value=op)),
      deadline=100.0,
      poll_interval=0.1,
      total_timeout=60.0,
  )
  assert result["status"] == "ERROR"
  assert (
      "Operation is not completed and does not contain a pollable"
      in result["error_details"]
  )


@pytest.mark.asyncio
async def test_await_lro_timeout(fake_clock):
  """Tests _await_lro timing out after reaching deadline."""
  mock_session = mock.MagicMock()

  def get_side_effect(*args, **kwargs):
    fake_clock.now += 30.0
    res = mock.Mock(ok=True)
    res.json.return_value = {
        "name": "projects/p/locations/g/operations/op-1",
        "done": False,
    }
    return res

  mock_session.get.side_effect = get_side_effect
  op = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": False,
  }
  result = await data_agent_tool._await_lro(
      session=mock_session,
      base_url="https://geminidataanalytics.googleapis.com/v1",
      headers={},
      resp=mock.Mock(ok=True, json=mock.Mock(return_value=op)),
      deadline=fake_clock.now + 10.0,
      poll_interval=0.1,
      total_timeout=10.0,
  )
  assert result["status"] == "ERROR"
  assert "did not complete within" in result["error_details"]
  assert result["operation_name"] == "projects/p/locations/g/operations/op-1"


@pytest.mark.asyncio
async def test_await_lro_poll_network_exception(fake_clock):
  """Tests _await_lro catching network exception during poll and returning ERROR."""
  mock_session = mock.MagicMock()
  mock_session.get.side_effect = Exception("Network unreachable")
  op = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": False,
  }
  result = await data_agent_tool._await_lro(
      session=mock_session,
      base_url="https://geminidataanalytics.googleapis.com/v1",
      headers={},
      resp=mock.Mock(ok=True, json=mock.Mock(return_value=op)),
      deadline=100.0,
      poll_interval=0.1,
      total_timeout=60.0,
  )
  assert result["status"] == "ERROR"
  assert "Network unreachable" in result["error_details"]
  assert result["operation_name"] == "projects/p/locations/g/operations/op-1"


@pytest.mark.asyncio
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
async def test_delete_data_agent_success(mock_get_session):
  """Tests delete_data_agent success path."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_response = mock.Mock(ok=True)
  mock_response.json.return_value = {"name": "operations/op-1", "done": True}
  mock_session.delete.return_value = mock_response
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.googleapis.com",
  )
  mock_settings = mock.Mock(
      enable_data_agent_modification=True,
      data_agent_modification_timeout_seconds=60,
      data_agent_modification_poll_interval_seconds=2,
  )

  result = await data_agent_tool.delete_data_agent(
      "projects/p/locations/g/dataAgents/agent-1",
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert result["status"] == "SUCCESS"
  mock_get_session.assert_called_once_with(mock_creds, location="g")
  mock_session.delete.assert_called_once_with(
      "https://geminidataanalytics.googleapis.com/v1/projects/p/locations/g/dataAgents/agent-1",
      headers=mock.ANY,
      timeout=mock.ANY,
  )


@pytest.mark.asyncio
async def test_delete_data_agent_disabled():
  """Tests delete_data_agent when disabled."""
  mock_creds = mock.Mock()
  mock_settings = mock.Mock(enable_data_agent_modification=False)
  result = await data_agent_tool.delete_data_agent(
      "projects/p/locations/g/dataAgents/agent-1",
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert "mutation is disabled" in result["error_details"]


@pytest.mark.asyncio
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
async def test_delete_data_agent_endpoint_matches_resource_name(
    mock_get_session,
):
  """Tests delete_data_agent resource name location overrides settings.location."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_response = mock.Mock(ok=True)
  mock_response.json.return_value = {"name": "op-1", "done": True}
  mock_session.delete.return_value = mock_response
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.googleapis.com",
  )
  mock_settings = mock.Mock(
      enable_data_agent_modification=True,
      data_agent_modification_timeout_seconds=60,
      data_agent_modification_poll_interval_seconds=2,
      location="eu",
  )

  result = await data_agent_tool.delete_data_agent(
      "projects/p/locations/us/dataAgents/agent-1",
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert result["status"] == "SUCCESS"
  mock_get_session.assert_called_once_with(mock_creds, location="us")


@pytest.mark.parametrize(
    "value,field_name,expected_valid",
    [
        ("my-project", "project_id", True),
        ("global", "location", True),
        ("agent-123", "data_agent_id", True),
        ("p/locations", "project_id", False),
        ("..", "location", False),
        ("agent?x=1", "data_agent_id", False),
    ],
)
def test_validate_path_segment(value, field_name, expected_valid):
  """Tests _validate_path_segment for various inputs."""
  err = data_agent_tool._validate_path_segment(value, field_name)
  if expected_valid:
    assert err is None
  else:
    assert err is not None
    assert "Invalid " + field_name + " format" in err["error_details"]


@pytest.mark.asyncio
async def test_create_data_agent_invalid_path_segment():
  """Tests create_data_agent with invalid project_id path segment."""
  mock_creds = mock.Mock()
  mock_settings = mock.Mock(enable_data_agent_modification=True)
  result = await data_agent_tool.create_data_agent(
      project_id="my/project",
      data_agent_id="my-agent",
      agent_config={"displayName": "Test"},
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert result["status"] == "ERROR"
  assert "Invalid project_id format" in result["error_details"]


@pytest.mark.asyncio
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
async def test_update_data_agent_success(mock_get_session):
  """Tests update_data_agent success path."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_response = mock.Mock(ok=True)
  mock_response.json.return_value = {"name": "operations/op-1", "done": True}
  mock_session.patch.return_value = mock_response
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.googleapis.com",
  )
  mock_settings = mock.Mock(
      enable_data_agent_modification=True,
      data_agent_modification_timeout_seconds=60,
      data_agent_modification_poll_interval_seconds=2,
  )

  result = await data_agent_tool.update_data_agent(
      "projects/p/locations/g/dataAgents/agent-1",
      '{"displayName": "updated"}',
      "displayName",
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert result["status"] == "SUCCESS"
  mock_get_session.assert_called_once_with(mock_creds, location="g")
  mock_session.patch.assert_called_once_with(
      "https://geminidataanalytics.googleapis.com/v1/projects/p/locations/g/dataAgents/agent-1",
      params={"updateMask": "displayName"},
      json={"displayName": "updated"},
      headers=mock.ANY,
      timeout=mock.ANY,
  )


@pytest.mark.asyncio
async def test_update_data_agent_disabled():
  """Tests update_data_agent when disabled."""
  mock_creds = mock.Mock()
  mock_settings = mock.Mock(enable_data_agent_modification=False)
  result = await data_agent_tool.update_data_agent(
      "projects/p/locations/g/dataAgents/agent-1",
      '{"displayName": "updated"}',
      "displayName",
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert result["status"] == "ERROR"
  assert "mutation is disabled" in result["error_details"]


@pytest.mark.asyncio
async def test_update_data_agent_missing_mask_field_rejected():
  """Tests update_data_agent rejects update_mask fields absent from agent_config."""
  mock_creds = mock.Mock()
  mock_settings = mock.Mock(enable_data_agent_modification=True)
  result = await data_agent_tool.update_data_agent(
      "projects/p/locations/g/dataAgents/agent-1",
      '{"displayName": "updated"}',
      "displayName,description",
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert result["status"] == "ERROR"
  assert (
      "update_mask fields ['description'] are not present"
      in result["error_details"]
  )


@pytest.mark.asyncio
async def test_update_data_agent_missing_nested_mask_field_rejected():
  """Tests update_data_agent rejects nested update_mask fields absent from agent_config."""
  mock_creds = mock.Mock()
  mock_settings = mock.Mock(enable_data_agent_modification=True)
  result = await data_agent_tool.update_data_agent(
      "projects/p/locations/g/dataAgents/agent-1",
      '{"dataAnalyticsAgent": {}}',
      "dataAnalyticsAgent.publishedContext.systemInstruction",
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert result["status"] == "ERROR"
  assert (
      "update_mask fields"
      " ['dataAnalyticsAgent.publishedContext.systemInstruction'] are not"
      " present"
      in result["error_details"]
  )


@pytest.mark.asyncio
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
async def test_update_data_agent_nested_mask_field_success(mock_get_session):
  """Tests update_data_agent succeeds when nested update_mask fields are present."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_response = mock.Mock(ok=True)
  mock_response.json.return_value = {"name": "operations/op-1", "done": True}
  mock_session.patch.return_value = mock_response
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.googleapis.com",
  )
  mock_settings = mock.Mock(
      enable_data_agent_modification=True,
      data_agent_modification_timeout_seconds=60,
      data_agent_modification_poll_interval_seconds=2,
  )

  result = await data_agent_tool.update_data_agent(
      "projects/p/locations/g/dataAgents/agent-1",
      '{"dataAnalyticsAgent": {"publishedContext": {"systemInstruction":'
      ' "test"}}}',
      "dataAnalyticsAgent.publishedContext.systemInstruction",
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert result["status"] == "SUCCESS"


@pytest.mark.asyncio
async def test_update_data_agent_empty_mask_error():
  """Tests update_data_agent with an empty update_mask."""
  result = await data_agent_tool.update_data_agent(
      "projects/p/locations/g/dataAgents/agent-1",
      '{"displayName": "New"}',
      "   ",
      credentials=mock.Mock(),
      settings=mock.Mock(enable_data_agent_modification=True),
  )
  assert result["status"] == "ERROR"
  assert (
      "update_mask must be a non-empty comma-separated list"
      in result["error_details"]
  )


@pytest.mark.asyncio
async def test_update_data_agent_invalid_name_error():
  """Tests update_data_agent with an invalid data_agent_name."""
  result = await data_agent_tool.update_data_agent(
      "invalid-name",
      '{"displayName": "New"}',
      "displayName",
      credentials=mock.Mock(),
      settings=mock.Mock(enable_data_agent_modification=True),
  )
  assert "Invalid data_agent_name format" in result["error_details"]


@pytest.mark.asyncio
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
async def test_update_data_agent_accepts_dict_from_programmatic_caller(
    mock_get_session,
):
  """Tests update_data_agent accepts dict for agent_config from Python callers."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_response = mock.Mock(ok=True)
  mock_response.json.return_value = {"name": "op-1", "done": True}
  mock_session.patch.return_value = mock_response
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.googleapis.com",
  )
  mock_settings = mock.Mock(
      enable_data_agent_modification=True,
      data_agent_modification_timeout_seconds=60,
      data_agent_modification_poll_interval_seconds=2,
  )

  result = await data_agent_tool.update_data_agent(
      "projects/p/locations/g/dataAgents/agent-1",
      {"displayName": "dict-config"},
      "displayName",
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert result["status"] == "SUCCESS"
  mock_session.patch.assert_called_once_with(
      "https://geminidataanalytics.googleapis.com/v1/projects/p/locations/g/dataAgents/agent-1",
      params={"updateMask": "displayName"},
      json={"displayName": "dict-config"},
      headers=mock.ANY,
      timeout=mock.ANY,
  )


@pytest.mark.asyncio
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
async def test_update_data_agent_endpoint_matches_resource_name(
    mock_get_session,
):
  """Tests update_data_agent resource name location overrides settings.location."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()
  mock_response = mock.Mock(ok=True)
  mock_response.json.return_value = {"name": "op-1", "done": True}
  mock_session.patch.return_value = mock_response
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.googleapis.com",
  )
  mock_settings = mock.Mock(
      enable_data_agent_modification=True,
      data_agent_modification_timeout_seconds=60,
      data_agent_modification_poll_interval_seconds=2,
      location="eu",
  )

  result = await data_agent_tool.update_data_agent(
      "projects/p/locations/us/dataAgents/agent-1",
      '{"displayName": "New"}',
      "displayName",
      credentials=mock_creds,
      settings=mock_settings,
  )
  assert result["status"] == "SUCCESS"
  mock_get_session.assert_called_once_with(mock_creds, location="us")
