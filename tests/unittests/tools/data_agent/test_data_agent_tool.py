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

from unittest import mock

from google.adk.tools.data_agent import data_agent_tool
from google.adk.tools.tool_context import ToolContext


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
  mock_get_session.assert_called_once_with(mock_creds)
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
  mock_get_session.assert_called_once_with(mock_creds)
  mock_get_stream.assert_called_once()
