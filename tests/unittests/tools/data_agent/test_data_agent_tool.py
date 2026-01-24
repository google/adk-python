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
  mock_requests.get.assert_called_once()


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
  mock_requests.get.assert_called_once()


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


@mock.patch.object(data_agent_tool, "_get_stream", autospec=True)
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
      {"Answer": "response1"},
      {"Answer": "response2"},
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
      {"Answer": "response1"},
      {"Answer": "response2"},
  ]
  mock_get_agent_info.assert_called_once()
  mock_get_stream.assert_called_once()


@mock.patch.object(data_agent_tool, "_get_stream", autospec=True)
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


@pytest.mark.parametrize(
    "case_file_path",
    [
        pytest.param("test_data/ask_data_insights_penguins_highest_mass.yaml"),
    ],
)
@mock.patch.object(requests.Session, "post")
def test_get_stream_from_file(mock_post, case_file_path):
  """Runs a full integration test for the _get_stream function using data from a specific file."""
  # 1. Construct the full, absolute path to the data file
  full_path = pathlib.Path(__file__).parent.parent / "bigquery" / case_file_path

  # 2. Load the test case data from the specified YAML file
  with open(full_path, "r", encoding="utf-8") as f:
    case_data = yaml.safe_load(f)

  # 3. Prepare the mock stream and expected output from the loaded data
  mock_stream_str = case_data["mock_api_stream"]
  fake_stream_lines = [
      line.encode("utf-8") for line in mock_stream_str.splitlines()
  ]
  # Load the expected output as a list of dictionaries, not a single string
  expected_final_list = case_data["expected_output"]
  data_retrieved = {
      "Data Retrieved": {
          "headers": ["island", "average_body_mass"],
          "rows": [
              ["Biscoe", "4716.017964071853"],
              ["Dream", "3712.9032258064512"],
              ["Torgersen", "3706.3725490196075"],
          ],
          "summary": "Showing all 3 rows.",
      }
  }
  expected_final_list.insert(-1, data_retrieved)

  # 4. Configure the mock for requests.post
  mock_response = mock.Mock()
  mock_response.iter_lines.return_value = fake_stream_lines
  # Add raise_for_status mock which is called in the updated code
  mock_response.raise_for_status.return_value = None
  mock_post.return_value.__enter__.return_value = mock_response

  # 5. Call the function under test
  result = data_agent_tool._get_stream(  # pylint: disable=protected-access
      url="fake_url",
      ca_payload={},
      headers={},
      max_query_result_rows=50,
  )

  # 6. Assert that the final list of dicts matches the expected output
  assert result == expected_final_list
