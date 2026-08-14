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
  mock_get_session.assert_called_once_with(mock_creds)
  mock_session.get.assert_called_once_with(
      "https://geminidataanalytics.googleapis.com/v1/projects/test-project/locations/global/dataAgents:listAccessible",
      headers={
          "Content-Type": "application/json",
          "X-Goog-API-Client": "GOOGLE_ADK",
      },
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
  mock_get_session.assert_called_once_with(mock_creds)
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
      "projects/p/locations/l/dataAgents/a", mock_creds, session=mock_session
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
  from google.adk.tools.data_agent.config import DataAgentToolConfig

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
  )


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


@pytest.mark.asyncio
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
async def test_create_data_agent_lro_operation_error(
    mock_get_session, fake_clock
):
  """Tests create_data_agent LRO returning error in operation."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()

  post_resp = mock.Mock(ok=True)
  post_resp.json.return_value = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": False,
  }
  mock_session.post.return_value = post_resp

  poll_resp = mock.Mock(ok=True)
  poll_resp.json.return_value = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": True,
      "error": {"code": 400, "message": "Creation invalid"},
  }
  mock_session.get.return_value = poll_resp

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

  assert result["status"] == "ERROR"
  assert "Creation invalid" in result["error_details"]


@pytest.mark.asyncio
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
async def test_create_data_agent_lro_poll_http_error(
    mock_get_session, fake_clock
):
  """Tests create_data_agent LRO polling encountering HTTP error."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()

  post_resp = mock.Mock(ok=True)
  post_resp.json.return_value = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": False,
  }
  mock_session.post.return_value = post_resp

  poll_resp = mock.Mock(ok=False, status_code=500, text="Internal Error")
  mock_session.get.return_value = poll_resp

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

  assert result["status"] == "ERROR"
  assert (
      "Polling failed with status: 500 Internal Error"
      in result["error_details"]
  )


@pytest.mark.asyncio
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
async def test_create_data_agent_lro_timeout(mock_get_session, fake_clock):
  """Tests create_data_agent LRO timing out after reaching deadline."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()

  post_resp = mock.Mock(ok=True)
  post_resp.json.return_value = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": False,
  }
  mock_session.post.return_value = post_resp

  def get_side_effect(*args, **kwargs):
    fake_clock.now += 30.0
    res = mock.Mock(ok=True)
    res.json.return_value = {
        "name": "projects/p/locations/g/operations/op-1",
        "done": False,
    }
    return res

  mock_session.get.side_effect = get_side_effect

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

  assert result["status"] == "ERROR"
  assert "did not complete within 60 seconds" in result["error_details"]
  assert result["operation_name"] == "projects/p/locations/g/operations/op-1"


@pytest.mark.asyncio
@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_gda_session", autospec=True
)
async def test_create_data_agent_returns_immediately_when_done(
    mock_get_session,
):
  """Tests create_data_agent returns immediately when POST response has done=True."""
  mock_creds = mock.Mock()
  mock_session = mock.MagicMock()

  post_resp = mock.Mock(ok=True)
  post_resp.json.return_value = {
      "name": "projects/p/locations/g/operations/op-1",
      "done": True,
      "response": {"name": "projects/p/locations/g/dataAgents/agent-1"},
  }
  mock_session.post.return_value = post_resp

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
      "agent-1",
      '{"displayName": "test"}',
      credentials=mock_creds,
      settings=mock_settings,
  )

  assert result["status"] == "SUCCESS"
  assert result["response"] == {
      "name": "projects/p/locations/g/dataAgents/agent-1"
  }
  mock_session.get.assert_not_called()
