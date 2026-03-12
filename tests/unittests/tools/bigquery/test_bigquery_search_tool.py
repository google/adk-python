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

import sys
from typing import Any
import unittest
from unittest import mock

from absl.testing import parameterized

# Mock google.genai and pydantic if not available, before importing google.adk modules
try:
  import google.genai
except ImportError:
  m = mock.MagicMock()
  m.__path__ = []
  sys.modules["google.genai"] = m
  sys.modules["google.genai.types"] = mock.MagicMock()
  sys.modules["google.genai.errors"] = mock.MagicMock()

try:
  import pydantic
except ImportError:
  m_pydantic = mock.MagicMock()

  class MockBaseModel:
    pass

  m_pydantic.BaseModel = MockBaseModel
  sys.modules["pydantic"] = m_pydantic

try:
  import fastapi
  import fastapi.openapi.models
except ImportError:
  m_fastapi = mock.MagicMock()
  m_fastapi.openapi.models = mock.MagicMock()
  sys.modules["fastapi"] = m_fastapi
  sys.modules["fastapi.openapi"] = mock.MagicMock()
  sys.modules["fastapi.openapi.models"] = mock.MagicMock()


from google.adk.tools.bigquery import search_tool
from google.adk.tools.bigquery.config import BigQueryToolConfig
from google.api_core import exceptions as api_exceptions
from google.auth.credentials import Credentials
from google.cloud import dataplex_v1


def _mock_creds():
  return mock.create_autospec(Credentials, instance=True)


def _mock_settings(app_name: str | None = "test-app"):
  return BigQueryToolConfig(application_name=app_name)


def _mock_search_entries_response(results: list[dict[str, Any]]):
  mock_response = mock.MagicMock(spec=dataplex_v1.SearchEntriesResponse)
  mock_results = []
  for r in results:
    mock_result = mock.create_autospec(
        dataplex_v1.SearchEntriesResult, instance=True
    )
    # Manually attach dataplex_entry since it's not visible in dir() of the proto class
    mock_entry = mock.create_autospec(dataplex_v1.Entry, instance=True)
    mock_result.dataplex_entry = mock_entry

    mock_entry.name = r.get("name")
    mock_entry.entry_type = r.get("entry_type")
    mock_entry.update_time = r.get("update_time", "2026-01-14T05:00:00Z")

    # Manually attach entry_source since it's not visible in dir() of the proto class
    mock_source = mock.create_autospec(dataplex_v1.EntrySource, instance=True)
    mock_entry.entry_source = mock_source

    mock_source.display_name = r.get("display_name")
    mock_source.resource = r.get("linked_resource")
    mock_source.description = r.get("description")
    mock_source.location = r.get("location")
    mock_results.append(mock_result)
  mock_response.results = mock_results
  return mock_response


class TestSearchCatalog(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_dataplex_client = mock.create_autospec(
        dataplex_v1.CatalogServiceClient, instance=True
    )

    # Patch get_dataplex_catalog_client
    self.mock_get_dataplex_client = self.enter_context(
        mock.patch(
            "google.adk.tools.bigquery.client.get_dataplex_catalog_client",
            autospec=True,
        )
    )
    self.mock_get_dataplex_client.return_value = self.mock_dataplex_client
    self.mock_dataplex_client.__enter__.return_value = self.mock_dataplex_client

    # Patch SearchEntriesRequest
    self.mock_search_request = self.enter_context(
        mock.patch(
            "google.cloud.dataplex_v1.SearchEntriesRequest", autospec=True
        )
    )

  def test_search_catalog_success(self):
    """Test the successful path of search_catalog."""
    creds = _mock_creds()
    settings = _mock_settings()
    prompt = "customer data"
    project_id = "test-project"
    location = "us"

    mock_api_results = [{
        "name": "entry1",
        "entry_type": "TABLE",
        "display_name": "Cust Table",
        "linked_resource": (
            "//bigquery.googleapis.com/projects/p/datasets/d/tables/t1"
        ),
        "description": "Table 1",
        "location": "us",
    }]
    self.mock_dataplex_client.search_entries.return_value = (
        _mock_search_entries_response(mock_api_results)
    )

    result = search_tool.search_catalog(
        prompt=prompt,
        project_id=project_id,
        credentials=creds,
        settings=settings,
        location=location,
    )

    with self.subTest("Test result content"):
      self.assertEqual(result["status"], "SUCCESS")
      self.assertLen(result["results"], 1)
      self.assertEqual(result["results"][0]["name"], "entry1")
      self.assertEqual(result["results"][0]["display_name"], "Cust Table")

    with self.subTest("Test mock calls"):
      self.mock_get_dataplex_client.assert_called_once_with(
          credentials=creds, user_agent=["test-app", "search_catalog"]
      )

      expected_query = (
          '(customer data) AND projectid="test-project" AND system=BIGQUERY'
      )
      self.mock_search_request.assert_called_once_with(
          name=f"projects/{project_id}/locations/us",
          query=expected_query,
          page_size=10,
          semantic_search=True,
      )
      self.mock_dataplex_client.search_entries.assert_called_once_with(
          request=self.mock_search_request.return_value
      )

  def test_search_catalog_no_project_id(self):
    """Test search_catalog with missing project_id."""
    result = search_tool.search_catalog(
        prompt="test",
        project_id="",
        credentials=_mock_creds(),
        settings=_mock_settings(),
        location="us",
    )
    self.assertEqual(result["status"], "ERROR")
    self.assertIn("project_id must be provided", result["error_details"])
    self.mock_get_dataplex_client.assert_not_called()

  def test_search_catalog_api_error(self):
    """Test search_catalog handling API exceptions."""
    self.mock_dataplex_client.search_entries.side_effect = (
        api_exceptions.BadRequest("Invalid query")
    )

    result = search_tool.search_catalog(
        prompt="test",
        project_id="test-project",
        credentials=_mock_creds(),
        settings=_mock_settings(),
        location="us",
    )
    self.assertEqual(result["status"], "ERROR")
    self.assertIn(
        "Dataplex API Error: 400 Invalid query", result["error_details"]
    )

  def test_search_catalog_other_exception(self):
    """Test search_catalog handling unexpected exceptions."""
    self.mock_get_dataplex_client.side_effect = Exception(
        "Something went wrong"
    )

    result = search_tool.search_catalog(
        prompt="test",
        project_id="test-project",
        credentials=_mock_creds(),
        settings=_mock_settings(),
        location="us",
    )
    self.assertEqual(result["status"], "ERROR")
    self.assertIn("Something went wrong", result["error_details"])

  @parameterized.named_parameters(
      ("project_filter", "p", ["proj1"], None, None, 'projectid="proj1"'),
      (
          "multi_project_filter",
          "p",
          ["p1", "p2"],
          None,
          None,
          '(projectid="p1" OR projectid="p2")',
      ),
      ("type_filter", "p", None, None, ["TABLE"], 'type="TABLE"'),
      (
          "multi_type_filter",
          "p",
          None,
          None,
          ["TABLE", "DATASET"],
          '(type="TABLE" OR type="DATASET")',
      ),
      (
          "project_and_dataset_filters",
          "inventory",
          ["proj1", "proj2"],
          ["dsetA"],
          None,
          (
              '(projectid="proj1" OR projectid="proj2") AND'
              ' (linked_resource:"//bigquery.googleapis.com/projects/proj1/datasets/dsetA/*"'
              ' OR linked_resource:"//bigquery.googleapis.com/projects/proj2/datasets/dsetA/*")'
          ),
      ),
  )
  def test_search_catalog_query_construction(
      self, prompt, project_ids, dataset_ids, types, expected_query_part
  ):
    """Test different query constructions based on filters."""
    search_tool.search_catalog(
        prompt=prompt,
        project_id="test-project",
        credentials=_mock_creds(),
        settings=_mock_settings(),
        location="us",
        project_ids_filter=project_ids,
        dataset_ids_filter=dataset_ids,
        types_filter=types,
    )

    self.mock_search_request.assert_called_once()
    _, kwargs = self.mock_search_request.call_args
    query = kwargs["query"]

    if prompt:
      assert f"({prompt})" in query
    assert "system=BIGQUERY" in query
    assert expected_query_part in query

  def test_search_catalog_no_app_name(self):
    """Test search_catalog when settings.application_name is None."""
    creds = _mock_creds()
    settings = _mock_settings(app_name=None)
    search_tool.search_catalog(
        prompt="test",
        project_id="test-project",
        credentials=creds,
        settings=settings,
        location="us",
    )

    self.mock_get_dataplex_client.assert_called_once_with(
        credentials=creds, user_agent=[None, "search_catalog"]
    )

  def test_search_catalog_multi_project_filter_semantic(self):
    """Test semantic search with a multi-project filter."""
    creds = _mock_creds()
    settings = _mock_settings()
    prompt = "What datasets store user profiles?"
    project_id = "main-project"
    project_filters = ["user-data-proj", "shared-infra-proj"]
    location = "global"

    self.mock_dataplex_client.search_entries.return_value = (
        _mock_search_entries_response([])
    )

    search_tool.search_catalog(
        prompt=prompt,
        project_id=project_id,
        credentials=creds,
        settings=settings,
        location=location,
        project_ids_filter=project_filters,
        types_filter=["DATASET"],
    )

    expected_query = (
        f"({prompt}) AND "
        '(projectid="user-data-proj" OR projectid="shared-infra-proj") AND '
        'type="DATASET" AND system=BIGQUERY'
    )
    self.mock_search_request.assert_called_once_with(
        name=f"projects/{project_id}/locations/{location}",
        query=expected_query,
        page_size=10,
        semantic_search=True,
    )
    self.mock_dataplex_client.search_entries.assert_called_once()

  def test_search_catalog_natural_language_semantic(self):
    """Test natural language prompts with semantic search enabled and check output."""
    creds = _mock_creds()
    settings = _mock_settings()
    prompt = "Find tables about football matches"
    project_id = "sports-analytics"
    location = "europe-west1"

    # Mock the results that the API would return for this semantic query
    mock_api_results = [
        {
            "name": (
                "projects/sports-analytics/locations/europe-west1/entryGroups/@bigquery/entries/fb1"
            ),
            "display_name": "uk_football_premiership",
            "entry_type": (
                "projects/655216118709/locations/global/entryTypes/bigquery-table"
            ),
            "linked_resource": (
                "//bigquery.googleapis.com/projects/sports-analytics/datasets/uk/tables/premiership"
            ),
            "description": "Stats for UK Premier League matches.",
            "location": "europe-west1",
        },
        {
            "name": (
                "projects/sports-analytics/locations/europe-west1/entryGroups/@bigquery/entries/fb2"
            ),
            "display_name": "serie_a_matches",
            "entry_type": (
                "projects/655216118709/locations/global/entryTypes/bigquery-table"
            ),
            "linked_resource": (
                "//bigquery.googleapis.com/projects/sports-analytics/datasets/italy/tables/serie_a"
            ),
            "description": "Italian Serie A football results.",
            "location": "europe-west1",
        },
    ]
    self.mock_dataplex_client.search_entries.return_value = (
        _mock_search_entries_response(mock_api_results)
    )

    result = search_tool.search_catalog(
        prompt=prompt,
        project_id=project_id,
        credentials=creds,
        settings=settings,
        location=location,
    )

    with self.subTest("Query Construction"):
      # Assert the request was made as expected
      expected_query = (
          f'({prompt}) AND projectid="{project_id}" AND system=BIGQUERY'
      )
      self.mock_search_request.assert_called_once_with(
          name=f"projects/{project_id}/locations/{location}",
          query=expected_query,
          page_size=10,
          semantic_search=True,
      )
      self.mock_dataplex_client.search_entries.assert_called_once()

    with self.subTest("Response Processing"):
      # Assert the output is processed correctly
      self.assertEqual(result["status"], "SUCCESS")
      self.assertLen(result["results"], 2)
      self.assertEqual(
          result["results"][0]["display_name"], "uk_football_premiership"
      )
      self.assertEqual(result["results"][1]["display_name"], "serie_a_matches")
      self.assertIn("UK Premier League", result["results"][0]["description"])

  def test_search_catalog_default_location(self):
    """Test search_catalog fallback to global location when None is provided."""
    creds = _mock_creds()
    settings = _mock_settings()
    # settings.location is None by default

    self.mock_dataplex_client.search_entries.return_value = (
        _mock_search_entries_response([])
    )

    search_tool.search_catalog(
        prompt="test",
        project_id="test-project",
        credentials=creds,
        settings=settings,
    )

    self.mock_search_request.assert_called_once()
    _, kwargs = self.mock_search_request.call_args
    name_arg = kwargs["name"]
    self.assertIn("locations/global", name_arg)

  def test_search_catalog_settings_location(self):
    """Test search_catalog uses settings.location when provided."""
    creds = _mock_creds()
    settings = BigQueryToolConfig(location="eu")

    self.mock_dataplex_client.search_entries.return_value = (
        _mock_search_entries_response([])
    )

    search_tool.search_catalog(
        prompt="test",
        project_id="test-project",
        credentials=creds,
        settings=settings,
    )

    self.mock_search_request.assert_called_once()
    _, kwargs = self.mock_search_request.call_args
    name_arg = kwargs["name"]
    self.assertIn("locations/eu", name_arg)
