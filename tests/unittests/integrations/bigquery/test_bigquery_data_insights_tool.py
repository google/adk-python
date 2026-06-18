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

from google.adk.integrations.bigquery import data_insights_tool
import pytest
import yaml


@pytest.mark.parametrize(
    "case_file_path",
    [
        pytest.param("test_data/ask_data_insights_penguins_highest_mass.yaml"),
    ],
)
@mock.patch.object(data_insights_tool.requests.Session, "post")
def test_ask_data_insights_pipeline_from_file(mock_post, case_file_path):
  """Runs a full integration test for the ask_data_insights pipeline using data from a specific file."""
  # 1. Construct the full, absolute path to the data file
  full_path = pathlib.Path(__file__).parent / case_file_path

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

  # 4. Configure the mock for requests.post
  mock_response = mock.Mock()
  mock_response.iter_lines.return_value = fake_stream_lines
  # Add raise_for_status mock which is called in the updated code
  mock_response.raise_for_status.return_value = None
  mock_post.return_value.__enter__.return_value = mock_response

  # 5. Call the function under test
  mock_creds = mock.Mock()
  mock_creds.token = "fake-token"
  mock_settings = mock.Mock()
  mock_settings.max_query_result_rows = 50
  result = data_insights_tool.ask_data_insights(
      project_id="test-project",
      user_query_with_context=case_data["user_question"],
      table_references=[],
      credentials=mock_creds,
      settings=mock_settings,
  )

  # 6. Assert that the final list of dicts matches the expected output
  assert result["status"] == "SUCCESS"
  assert result["response"] == expected_final_list


@mock.patch.object(data_insights_tool._gda_stream_util, "get_stream")
def test_ask_data_insights_success(mock_get_stream):
  """Tests the success path of ask_data_insights using decorators."""
  # 1. Configure the behavior of the mocked functions
  mock_stream = [
      {"text": {"parts": ["response1"], "textType": "THOUGHT"}},
      {"text": {"parts": ["response2"], "textType": "FINAL_RESPONSE"}},
  ]
  mock_get_stream.return_value = mock_stream

  # 2. Create mock inputs for the function call
  mock_creds = mock.Mock()
  mock_creds.token = "fake-token"
  mock_settings = mock.Mock()
  mock_settings.max_query_result_rows = 100

  # 3. Call the function under test
  result = data_insights_tool.ask_data_insights(
      project_id="test-project",
      user_query_with_context="test query",
      table_references=[],
      credentials=mock_creds,
      settings=mock_settings,
  )

  # 4. Assert the results are as expected
  assert result["status"] == "SUCCESS"
  assert result["response"] == mock_stream
  mock_get_stream.assert_called_once()

  # Verify that the correct headers and client ID were passed to _get_stream
  args, _ = mock_get_stream.call_args
  headers = args[2]
  assert headers["X-Goog-API-Client"] == "GOOGLE_ADK"
  assert headers["Authorization"] == "Bearer fake-token"


@mock.patch.object(data_insights_tool._gda_stream_util, "get_stream")
def test_ask_data_insights_handles_exception(mock_get_stream):
  """Tests the exception path of ask_data_insights using decorators."""
  # 1. Configure one of the mocks to raise an error
  mock_get_stream.side_effect = Exception("API call failed!")

  # 2. Create mock inputs
  mock_creds = mock.Mock()
  mock_creds.token = "fake-token"
  mock_settings = mock.Mock()

  # 3. Call the function
  result = data_insights_tool.ask_data_insights(
      project_id="test-project",
      user_query_with_context="test query",
      table_references=[],
      credentials=mock_creds,
      settings=mock_settings,
  )

  # 4. Assert that the error was caught and formatted correctly
  assert result["status"] == "ERROR"
  assert "API call failed!" in result["error_details"]
  mock_get_stream.assert_called_once()
