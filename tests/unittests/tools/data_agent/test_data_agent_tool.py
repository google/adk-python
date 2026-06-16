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

import pathlib
from unittest import mock

from google.adk.tools.data_agent import data_agent_tool
from google.adk.tools.tool_context import ToolContext
import pytest
import requests
import yaml


@mock.patch.object(data_agent_tool, "requests", autospec=True)
def test_list_accessible_data_agents_success(mock_requests):
  """Tests list_accessible_data_agents success path."""
  mock_creds = mock.Mock()
  mock_creds.token = "fake-token"
  mock_response = mock.Mock()
  mock_response.json.return_value = {"dataAgents": ["agent1", "agent2"]}
  mock_response.raise_for_status.return_value = None
  mock_requests.get.return_value = mock_response
  result = data_agent_tool.list_accessible_data_agents(
      "test-project", mock_creds
  )
  assert result["status"] == "SUCCESS"
  assert result["response"] == ["agent1", "agent2"]
  mock_requests.get.assert_called_once_with(
      "https://geminidataanalytics.googleapis.com/v1beta/projects/test-project/locations/global/dataAgents:listAccessible",
      headers={
          "Authorization": "Bearer fake-token",
          "Content-Type": "application/json",
          "X-Goog-API-Client": "GOOGLE_ADK",
      },
  )


@mock.patch.object(data_agent_tool, "requests", autospec=True)
def test_list_accessible_data_agents_exception(mock_requests):
  """Tests list_accessible_data_agents exception path."""
  mock_creds = mock.Mock()
  mock_creds.token = "fake-token"
  mock_requests.get.side_effect = Exception("List failed!")
  result = data_agent_tool.list_accessible_data_agents(
      "test-project", mock_creds
  )
  assert result["status"] == "ERROR"
  assert "List failed!" in result["error_details"]
  mock_requests.get.assert_called_once()


@mock.patch.object(data_agent_tool, "requests", autospec=True)
def test_get_data_agent_info_success(mock_requests):
  """Tests get_data_agent_info success path."""
  mock_creds = mock.Mock()
  mock_creds.token = "fake-token"
  mock_response = mock.Mock()
  mock_response.json.return_value = "agent_info"
  mock_response.raise_for_status.return_value = None
  mock_requests.get.return_value = mock_response
  result = data_agent_tool.get_data_agent_info("agent_name", mock_creds)
  assert result["status"] == "SUCCESS"
  assert result["response"] == "agent_info"
  mock_requests.get.assert_called_once_with(
      "https://geminidataanalytics.googleapis.com/v1beta/agent_name",
      headers={
          "Authorization": "Bearer fake-token",
          "Content-Type": "application/json",
          "X-Goog-API-Client": "GOOGLE_ADK",
      },
  )


@mock.patch.object(data_agent_tool, "requests", autospec=True)
def test_get_data_agent_info_exception(mock_requests):
  """Tests get_data_agent_info exception path."""
  mock_creds = mock.Mock()
  mock_creds.token = "fake-token"
  mock_requests.get.side_effect = Exception("Get failed!")
  result = data_agent_tool.get_data_agent_info("agent_name", mock_creds)
  assert result["status"] == "ERROR"
  assert "Get failed!" in result["error_details"]
  mock_requests.get.assert_called_once()


@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_stream", autospec=True
)
@mock.patch.object(data_agent_tool, "requests", autospec=True)
@mock.patch.object(data_agent_tool, "get_data_agent_info", autospec=True)
def test_ask_data_agent_success(
    mock_get_agent_info, mock_requests, mock_get_stream
):
  """Tests ask_data_agent success path."""
  mock_creds = mock.Mock()
  mock_creds.token = "fake-token"
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
  mock_get_agent_info.assert_called_once()
  mock_get_stream.assert_called_once_with(
      "https://geminidataanalytics.googleapis.com/v1beta/projects/p/locations/l:chat",
      {
          "messages": [{"userMessage": {"text": "query"}}],
          "dataAgentContext": {
              "dataAgent": "projects/p/locations/l/dataAgents/a",
          },
          "clientIdEnum": "GOOGLE_ADK",
      },
      {
          "Authorization": "Bearer fake-token",
          "Content-Type": "application/json",
          "X-Goog-API-Client": "GOOGLE_ADK",
      },
      mock_settings.max_query_result_rows,
  )


@mock.patch.object(
    data_agent_tool._gda_stream_util, "get_stream", autospec=True
)
@mock.patch.object(data_agent_tool, "requests", autospec=True)
@mock.patch.object(data_agent_tool, "get_data_agent_info", autospec=True)
def test_ask_data_agent_exception(
    mock_get_agent_info, mock_requests, mock_get_stream
):
  """Tests ask_data_agent exception path."""
  mock_creds = mock.Mock()
  mock_creds.token = "fake-token"
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
  mock_get_stream.assert_called_once()
