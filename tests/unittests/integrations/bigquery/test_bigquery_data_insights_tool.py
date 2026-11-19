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
from typing import Any
from unittest import mock

from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.integrations.bigquery import BigQueryToolset
from google.adk.integrations.bigquery import data_insights_tool
from google.adk.integrations.bigquery.config import BigQueryToolConfig
from google.adk.tools import _gda_stream_util
from google.auth.credentials import Credentials
import pytest
import yaml

_ENDPOINT = "https://geminidataanalytics.mtls.googleapis.com"


@pytest.mark.parametrize(
    "case_file_path",
    [
        pytest.param("test_data/ask_data_insights_penguins_highest_mass.yaml"),
    ],
)
@mock.patch.object(
    data_insights_tool._gda_stream_util, "get_gda_session", autospec=True
)
def test_ask_data_insights_pipeline_from_file(mock_get_session, case_file_path):
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
  mock_session = mock.MagicMock()
  mock_response = mock.Mock()
  mock_response.iter_lines.return_value = fake_stream_lines
  # Add raise_for_status mock which is called in the updated code
  mock_response.raise_for_status.return_value = None
  mock_session.post.return_value.__enter__.return_value = mock_response
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.mtls.googleapis.com",
  )

  # 5. Call the function under test
  mock_creds = mock.Mock()
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
  mock_get_session.assert_called_once_with(mock_creds)
  mock_session.post.assert_called_once_with(
      "https://geminidataanalytics.mtls.googleapis.com/v1/projects/test-project/locations/global:chat",
      json={
          "messages": [{"userMessage": {"text": case_data["user_question"]}}],
          "inlineContext": {
              "datasourceReferences": {"bq": {"tableReferences": []}},
              "systemInstruction": mock.ANY,
          },
          "clientIdEnum": "GOOGLE_ADK",
      },
      headers={
          "Content-Type": "application/json",
          "X-Goog-API-Client": "GOOGLE_ADK",
      },
      stream=True,
  )


@mock.patch.object(data_insights_tool._gda_stream_util, "get_stream")
@mock.patch.object(
    data_insights_tool._gda_stream_util, "get_gda_session", autospec=True
)
def test_ask_data_insights_success(mock_get_session, mock_get_stream):
  """Tests the success path of ask_data_insights using decorators."""
  # 1. Configure the behavior of the mocked functions
  mock_stream = [
      {"text": {"parts": ["response1"], "textType": "THOUGHT"}},
      {"text": {"parts": ["response2"], "textType": "FINAL_RESPONSE"}},
  ]
  mock_get_stream.return_value = mock_stream
  mock_session = mock.MagicMock()
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.mtls.googleapis.com",
  )

  # 2. Create mock inputs for the function call
  mock_creds = mock.Mock()
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
  mock_get_session.assert_called_once_with(mock_creds)
  mock_get_stream.assert_called_once_with(
      mock_session,
      "https://geminidataanalytics.mtls.googleapis.com/v1/projects/test-project/locations/global:chat",
      {
          "messages": [{"userMessage": {"text": "test query"}}],
          "inlineContext": {
              "datasourceReferences": {"bq": {"tableReferences": []}},
              "systemInstruction": mock.ANY,
          },
          "clientIdEnum": "GOOGLE_ADK",
      },
      {
          "Content-Type": "application/json",
          "X-Goog-API-Client": "GOOGLE_ADK",
      },
      100,
  )


@mock.patch.object(data_insights_tool._gda_stream_util, "get_stream")
@mock.patch.object(
    data_insights_tool._gda_stream_util, "get_gda_session", autospec=True
)
def test_ask_data_insights_handles_exception(mock_get_session, mock_get_stream):
  """Tests the exception path of ask_data_insights using decorators."""
  # 1. Configure one of the mocks to raise an error
  mock_get_stream.side_effect = Exception("API call failed!")
  mock_session = mock.MagicMock()
  mock_get_session.return_value = (
      mock_session,
      "https://geminidataanalytics.mtls.googleapis.com",
  )

  # 2. Create mock inputs
  mock_creds = mock.Mock()
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
  mock_get_session.assert_called_once_with(mock_creds)
  mock_get_stream.assert_called_once()


@pytest.fixture(name="gda_api")
def _gda_api():
  """Fakes the GDA API session and stream, yielding both mocks."""
  with (
      mock.patch.object(
          _gda_stream_util, "get_gda_session", autospec=True
      ) as get_session,
      mock.patch.object(
          _gda_stream_util, "get_stream", autospec=True
      ) as get_stream,
  ):
    get_session.return_value = (mock.MagicMock(), _ENDPOINT)
    get_stream.return_value = [{"Answer": "Total sales were $42."}]
    yield get_session, get_stream


def _ask(**kwargs: Any) -> dict[str, Any]:
  """Calls ask_data_insights with fake credentials and default settings."""
  kwargs.setdefault(
      "credentials", mock.create_autospec(Credentials, instance=True)
  )
  kwargs.setdefault("settings", BigQueryToolConfig())
  return data_insights_tool.ask_data_insights(
      project_id="test-project",
      user_query_with_context="What were total sales?",
      **kwargs,
  )


def _sent_bq_references(get_stream: mock.MagicMock) -> dict[str, Any]:
  """Returns the BigQuery datasource references of the sent chat request."""
  ca_payload = get_stream.call_args.args[2]
  return ca_payload["inlineContext"]["datasourceReferences"]["bq"]


def test_ask_data_insights_with_search_scope_include_project_ids(gda_api):
  """A project search scope is sent in place of table references."""
  get_session, get_stream = gda_api

  result = _ask(
      search_scope={"includeProjectIds": ["my_project", "other_project"]}
  )

  assert result == {
      "status": "SUCCESS",
      "response": [{"Answer": "Total sales were $42."}],
  }
  get_stream.assert_called_once_with(
      get_session.return_value[0],
      f"{_ENDPOINT}/v1/projects/test-project/locations/global:chat",
      {
          "messages": [{"userMessage": {"text": "What were total sales?"}}],
          "inlineContext": {
              "datasourceReferences": {
                  "bq": {
                      "searchScope": {
                          "includeProjectIds": ["my_project", "other_project"]
                      }
                  }
              },
              "systemInstruction": mock.ANY,
          },
          "clientIdEnum": "GOOGLE_ADK",
      },
      {"Content-Type": "application/json", "X-Goog-API-Client": "GOOGLE_ADK"},
      BigQueryToolConfig().max_query_result_rows,
  )


def test_ask_data_insights_with_search_scope_include_dataset_references(
    gda_api,
):
  """A dataset search scope is sent in place of table references."""
  _, get_stream = gda_api

  result = _ask(
      search_scope={
          "includeDatasetReferences": [
              {"projectId": "my_project", "datasetId": "sales_ds"}
          ]
      }
  )

  assert result["status"] == "SUCCESS"
  assert _sent_bq_references(get_stream) == {
      "searchScope": {
          "includeDatasetReferences": [
              {"projectId": "my_project", "datasetId": "sales_ds"}
          ]
      }
  }


def test_ask_data_insights_with_snake_case_search_scope_normalization(gda_api):
  """Snake_case search scope keys are sent as camelCase REST JSON keys."""
  _, get_stream = gda_api

  result = _ask(
      search_scope={
          "include_project_ids": ["p1"],
          "include_dataset_references": [
              {"project_id": "p1", "dataset_id": "d1"}
          ],
      }
  )

  assert result["status"] == "SUCCESS"
  assert _sent_bq_references(get_stream) == {
      "searchScope": {
          "includeProjectIds": ["p1"],
          "includeDatasetReferences": [{"projectId": "p1", "datasetId": "d1"}],
      }
  }


def test_ask_data_insights_with_search_scope_strips_surrounding_whitespace(
    gda_api,
):
  """Surrounding whitespace on project and dataset IDs is stripped."""
  _, get_stream = gda_api

  result = _ask(
      search_scope={
          "includeProjectIds": ["  my_project  "],
          "includeDatasetReferences": [
              {"projectId": " my_project ", "datasetId": " sales_ds "}
          ],
      }
  )

  assert result["status"] == "SUCCESS"
  assert _sent_bq_references(get_stream) == {
      "searchScope": {
          "includeProjectIds": ["my_project"],
          "includeDatasetReferences": [
              {"projectId": "my_project", "datasetId": "sales_ds"}
          ],
      }
  }


def test_ask_data_insights_with_search_scope_at_max_entries_limit(
    gda_api,
):
  """A search scope with 100 project IDs and 100 datasets is accepted."""
  _, get_stream = gda_api
  project_ids = [f"project_{i}" for i in range(100)]
  dataset_references = [
      {"projectId": "my_project", "datasetId": f"ds_{i}"} for i in range(100)
  ]

  result = _ask(
      search_scope={
          "includeProjectIds": project_ids,
          "includeDatasetReferences": dataset_references,
      }
  )

  assert result["status"] == "SUCCESS"
  assert _sent_bq_references(get_stream) == {
      "searchScope": {
          "includeProjectIds": project_ids,
          "includeDatasetReferences": dataset_references,
      }
  }


def test_ask_data_insights_with_both_table_references_and_search_scope(
    gda_api,
):
  """Table references and a search scope are sent together."""
  _, get_stream = gda_api
  table_references = [
      {"projectId": "my_project", "datasetId": "sales_ds", "tableId": "orders"}
  ]

  result = _ask(
      table_references=table_references,
      search_scope={"includeProjectIds": ["other_project"]},
  )

  assert result["status"] == "SUCCESS"
  assert _sent_bq_references(get_stream) == {
      "tableReferences": table_references,
      "searchScope": {"includeProjectIds": ["other_project"]},
  }


def test_ask_data_insights_missing_both_table_references_and_search_scope_returns_error(
    gda_api,
):
  """Without table references or a search scope, no request is sent."""
  get_session, get_stream = gda_api

  result = _ask()

  assert result == {
      "status": "ERROR",
      "error_details": (
          "At least one of table_references or search_scope must be provided."
      ),
  }
  get_session.assert_not_called()
  get_stream.assert_not_called()


@pytest.mark.parametrize(
    "table_references",
    [
        pytest.param(None, id="without-table-references"),
        pytest.param(
            [{"projectId": "p1", "datasetId": "d1", "tableId": "t1"}],
            id="with-table-references",
        ),
    ],
)
def test_ask_data_insights_empty_search_scope_returns_error(
    gda_api, table_references
):
  """An empty search scope is rejected, even next to table references."""
  get_session, get_stream = gda_api

  result = _ask(table_references=table_references, search_scope={})

  assert result == {
      "status": "ERROR",
      "error_details": (
          "search_scope must contain non-empty includeProjectIds or"
          " includeDatasetReferences."
      ),
  }
  get_session.assert_not_called()
  get_stream.assert_not_called()


@pytest.mark.parametrize(
    ("search_scope", "error"),
    [
        pytest.param(
            "my_project", "search_scope must be a dictionary", id="not-a-dict"
        ),
        pytest.param(
            {"includeProjectIds": [], "includeDatasetReferences": []},
            "search_scope must contain non-empty includeProjectIds",
            id="empty-lists",
        ),
        pytest.param(
            {"projectIds": ["my_project"]},
            "search_scope has unsupported key 'projectIds'",
            id="unsupported-key",
        ),
        pytest.param(
            {"includeProjectIds": ["p1"], "include_project_ids": ["p2"]},
            "search_scope sets includeProjectIds more than once",
            id="duplicate-key",
        ),
        pytest.param(
            {"includeProjectIds": "my_project"},
            "includeProjectIds must be a list of non-blank project IDs",
            id="project-ids-not-a-list",
        ),
        pytest.param(
            {"includeProjectIds": ""},
            "includeProjectIds must be a list of non-blank project IDs",
            id="project-ids-empty-string",
        ),
        pytest.param(
            {"includeProjectIds": 0},
            "includeProjectIds must be a list of non-blank project IDs",
            id="project-ids-zero",
        ),
        pytest.param(
            {"includeProjectIds": False},
            "includeProjectIds must be a list of non-blank project IDs",
            id="project-ids-false",
        ),
        pytest.param(
            {"includeProjectIds": {}},
            "includeProjectIds must be a list of non-blank project IDs",
            id="project-ids-empty-dict",
        ),
        pytest.param(
            {
                "includeProjectIds": "",
                "includeDatasetReferences": [
                    {"projectId": "p1", "datasetId": "d1"}
                ],
            },
            "includeProjectIds must be a list of non-blank project IDs",
            id="falsy-project-ids-with-valid-dataset-references",
        ),
        pytest.param(
            {"includeProjectIds": ["my_project", " "]},
            "includeProjectIds must be a list of non-blank project IDs",
            id="blank-project-id",
        ),
        pytest.param(
            {"includeProjectIds": [f"project_{i}" for i in range(101)]},
            "includeProjectIds must not contain more than 100 project IDs",
            id="too-many-project-ids",
        ),
        pytest.param(
            {"includeDatasetReferences": ""},
            "includeDatasetReferences must be a list of dictionaries",
            id="dataset-references-empty-string",
        ),
        pytest.param(
            {"includeDatasetReferences": 0},
            "includeDatasetReferences must be a list of dictionaries",
            id="dataset-references-zero",
        ),
        pytest.param(
            {"includeDatasetReferences": False},
            "includeDatasetReferences must be a list of dictionaries",
            id="dataset-references-false",
        ),
        pytest.param(
            {"includeDatasetReferences": {}},
            "includeDatasetReferences must be a list of dictionaries",
            id="dataset-references-empty-dict",
        ),
        pytest.param(
            {
                "includeProjectIds": ["my_project"],
                "includeDatasetReferences": {},
            },
            "includeDatasetReferences must be a list of dictionaries",
            id="falsy-dataset-references-with-valid-project-ids",
        ),
        pytest.param(
            {"includeDatasetReferences": ["my_project.sales_ds"]},
            "includeDatasetReferences must be a list of dictionaries",
            id="dataset-reference-not-a-dict",
        ),
        pytest.param(
            {"includeDatasetReferences": [{"projectId": "my_project"}]},
            "must have a non-blank projectId and datasetId",
            id="missing-dataset-id",
        ),
        pytest.param(
            {
                "includeDatasetReferences": [
                    {"projectId": " ", "datasetId": "sales_ds"}
                ]
            },
            "must have a non-blank projectId and datasetId",
            id="blank-project-id-in-dataset-reference",
        ),
        pytest.param(
            {
                "includeDatasetReferences": [{
                    "projectId": "my_project",
                    "datasetId": "sales_ds",
                    "tableId": "orders",
                }]
            },
            "has unsupported key 'tableId'",
            id="unsupported-dataset-reference-key",
        ),
        pytest.param(
            {
                "includeDatasetReferences": [
                    {"projectId": "my_project", "datasetId": f"ds_{i}"}
                    for i in range(101)
                ]
            },
            (
                "includeDatasetReferences must not contain more than 100"
                " dataset references"
            ),
            id="too-many-dataset-references",
        ),
    ],
)
def test_ask_data_insights_invalid_search_scope_returns_error(
    gda_api, search_scope, error
):
  """A malformed search scope is rejected before any request is sent."""
  get_session, get_stream = gda_api

  result = _ask(search_scope=search_scope)

  assert result["status"] == "ERROR"
  assert error in result["error_details"]
  get_session.assert_not_called()
  get_stream.assert_not_called()


def test_ask_data_insights_without_credentials_returns_error(gda_api):
  """Without credentials, no request is sent."""
  get_session, get_stream = gda_api

  result = _ask(
      search_scope={"includeProjectIds": ["my_project"]}, credentials=None
  )

  assert result == {
      "status": "ERROR",
      "error_details": (
          "credentials are required to call the Conversational Analytics API."
      ),
  }
  get_session.assert_not_called()
  get_stream.assert_not_called()


def test_ask_data_insights_without_settings_uses_default_row_limit(gda_api):
  """Without settings, the default BigQueryToolConfig row limit applies."""
  _, get_stream = gda_api

  result = _ask(
      search_scope={"includeProjectIds": ["my_project"]}, settings=None
  )

  assert result["status"] == "SUCCESS"
  assert (
      get_stream.call_args.args[4] == BigQueryToolConfig().max_query_result_rows
  )


def test_ask_data_insights_accepts_positional_arguments(gda_api):
  """Table references, credentials and settings still bind positionally."""
  get_session, get_stream = gda_api
  credentials = mock.create_autospec(Credentials, instance=True)
  table_references = [{"projectId": "p1", "datasetId": "d1", "tableId": "t1"}]

  result = data_insights_tool.ask_data_insights(
      "test-project",
      "What were total sales?",
      table_references,
      credentials,
      BigQueryToolConfig(max_query_result_rows=7),
  )

  assert result["status"] == "SUCCESS"
  get_session.assert_called_once_with(credentials)
  assert _sent_bq_references(get_stream) == {
      "tableReferences": table_references
  }
  assert get_stream.call_args.args[4] == 7


@pytest.mark.parametrize(
    "json_schema", [True, False], ids=["json-schema", "schema-object"]
)
async def test_ask_data_insights_declaration_offers_optional_search_scope(
    json_schema,
):
  """The model can send table references, a search scope, or both."""
  with temporary_feature_override(
      FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, json_schema
  ):
    [tool] = await BigQueryToolset(
        tool_filter=["ask_data_insights"]
    ).get_tools()
    declaration = tool._get_declaration()  # pylint: disable=protected-access

  if json_schema:
    assert declaration.parameters is None
    assert declaration.parameters_json_schema is not None
    schema = declaration.parameters_json_schema
    properties = schema.get("properties") or {}
    required = schema.get("required") or []
  else:
    assert declaration.parameters_json_schema is None
    assert declaration.parameters is not None
    properties = declaration.parameters.properties or {}
    required = declaration.parameters.required or []
  assert set(properties) == {
      "project_id",
      "user_query_with_context",
      "table_references",
      "search_scope",
  }
  assert set(required) == {"project_id", "user_query_with_context"}
