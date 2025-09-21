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

import pathlib
from unittest import mock

from google.adk.tools.bigquery import data_insights_tool
import pytest
import yaml


@pytest.mark.parametrize(
    "case_file_path",
    [
        pytest.param("test_data/ask_data_insights_penguins_highest_mass.yaml"),
    ],
)
@mock.patch(
    "google.adk.tools.bigquery.data_insights_tool.requests.Session.post"
)
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
  result = data_insights_tool._get_stream(  # pylint: disable=protected-access
      url="fake_url",
      ca_payload={},
      headers={},
      max_query_result_rows=50,
  )

  # 6. Assert that the final list of dicts matches the expected output
  assert result == expected_final_list


@mock.patch("google.adk.tools.bigquery.data_insights_tool._get_stream")
def test_ask_data_insights_success(mock_get_stream):
  """Tests the success path of ask_data_insights using decorators."""
  # 1. Configure the behavior of the mocked functions
  mock_get_stream.return_value = "Final formatted string from stream"

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
  assert result["response"] == "Final formatted string from stream"
  mock_get_stream.assert_called_once()


@mock.patch("google.adk.tools.bigquery.data_insights_tool._get_stream")
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


@pytest.mark.parametrize(
    "initial_messages, new_message, expected_list",
    [
        pytest.param(
            [{"Thinking": None}, {"Schema Resolved": {}}],
            {"SQL Generated": "SELECT 1"},
            [
                {"Thinking": None},
                {"Schema Resolved": {}},
                {"SQL Generated": "SELECT 1"},
            ],
            id="append_when_last_message_is_not_data",
        ),
        pytest.param(
            [{"Thinking": None}, {"Data Retrieved": {"rows": [1]}}],
            {"Data Retrieved": {"rows": [1, 2]}},
            [{"Thinking": None}, {"Data Retrieved": {"rows": [1, 2]}}],
            id="replace_when_last_message_is_data",
        ),
        pytest.param(
            [],
            {"Answer": "First Message"},
            [{"Answer": "First Message"}],
            id="append_to_an_empty_list",
        ),
        pytest.param(
            [{"Data Retrieved": {}}],
            {},
            [{"Data Retrieved": {}}],
            id="should_not_append_an_empty_new_message",
        ),
    ],
)
def test_append_message(initial_messages, new_message, expected_list):
  """Tests the logic of replacing the last message if it's a data message."""
  messages_copy = initial_messages.copy()
  data_insights_tool._append_message(messages_copy, new_message)  # pylint: disable=protected-access
  assert messages_copy == expected_list


@pytest.mark.parametrize(
    "response_dict, expected_output",
    [
        pytest.param(
            {"parts": ["The answer", " is 42."]},
            {"Answer": "The answer is 42."},
            id="multiple_parts",
        ),
        pytest.param(
            {"parts": ["Hello"]}, {"Answer": "Hello"}, id="single_part"
        ),
        pytest.param({}, {"Answer": ""}, id="empty_response"),
    ],
)
def test_handle_text_response(response_dict, expected_output):
  """Tests the text response handler."""
  result = data_insights_tool._handle_text_response(response_dict)  # pylint: disable=protected-access
  assert result == expected_output


@pytest.mark.parametrize(
    "response_dict, expected_output",
    [
        pytest.param(
            {"query": {"question": "What is the schema?"}},
            {"Question": "What is the schema?"},
            id="schema_query_path",
        ),
        pytest.param(
            {
                "result": {
                    "datasources": [{
                        "bigqueryTableReference": {
                            "projectId": "p",
                            "datasetId": "d",
                            "tableId": "t",
                        },
                        "schema": {
                            "fields": [{"name": "col1", "type": "STRING"}]
                        },
                    }]
                }
            },
            {
                "Schema Resolved": [{
                    "source_name": "p.d.t",
                    "schema": {
                        "headers": ["Column", "Type", "Description", "Mode"],
                        "rows": [["col1", "STRING", "", ""]],
                    },
                }]
            },
            id="schema_result_path",
        ),
    ],
)
def test_handle_schema_response(response_dict, expected_output):
  """Tests different paths of the schema response handler."""
  result = data_insights_tool._handle_schema_response(response_dict)  # pylint: disable=protected-access
  assert result == expected_output


@pytest.mark.parametrize(
    "response_dict, expected_output",
    [
        pytest.param(
            {"generatedSql": "SELECT 1;"},
            {"SQL Generated": "SELECT 1;"},
            id="format_generated_sql",
        ),
        pytest.param(
            {
                "result": {
                    "schema": {"fields": [{"name": "id"}, {"name": "name"}]},
                    "data": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
                }
            },
            {
                "Data Retrieved": {
                    "headers": ["id", "name"],
                    "rows": [[1, "A"], [2, "B"]],
                    "summary": "Showing all 2 rows.",
                }
            },
            id="format_data_result_table",
        ),
    ],
)
def test_handle_data_response(response_dict, expected_output):
  """Tests different paths of the data response handler, including truncation."""
  result = data_insights_tool._handle_data_response(response_dict, 100)  # pylint: disable=protected-access
  assert result == expected_output


@pytest.mark.parametrize(
    "response_dict, expected_output",
    [
        pytest.param(
            {"code": 404, "message": "Not Found"},
            {"Error": {"Code": 404, "Message": "Not Found"}},
            id="full_error_message",
        ),
        pytest.param(
            {"code": 500},
            {"Error": {"Code": 500, "Message": "No message provided."}},
            id="error_with_missing_message",
        ),
    ],
)
def test_handle_error(response_dict, expected_output):
  """Tests the error response handler."""
  result = data_insights_tool._handle_error(response_dict)  # pylint: disable=protected-access
  assert result == expected_output
