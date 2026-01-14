from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from unittest import mock

from google.adk.tools.bigquery import client
from google.adk.tools.bigquery import search_tool
from google.adk.tools.bigquery.config import BigQueryToolConfig
from google.api_core import exceptions as api_exceptions
from google.auth.credentials import Credentials
from google.cloud import dataplex_v1
import pytest


# Helper function to create mock credentials
def _mock_creds():
  return mock.create_autospec(Credentials, instance=True)


# Helper function to create mock settings
def _mock_settings(app_name: str | None = "test-app"):
  return BigQueryToolConfig(application_name=app_name)


# Mock response for dataplex_client.search_entries
def _mock_search_entries_response(results: List[Dict[str, Any]]):
  mock_response = mock.MagicMock(spec=dataplex_v1.SearchEntriesResponse)
  mock_results = []
  for r in results:
    mock_result = mock.MagicMock()
    mock_entry = mock_result.dataplex_entry
    mock_entry.name = r.get("name")
    mock_entry.entry_type = r.get("entry_type")
    mock_entry.update_time = r.get("update_time", "2026-01-14T05:00:00Z")
    mock_source = mock_entry.entry_source
    mock_source.display_name = r.get("display_name")
    mock_source.resource = r.get("linked_resource")
    mock_source.description = r.get("description")
    mock_source.location = r.get("location")
    mock_results.append(mock_result)
  mock_response.results = mock_results
  return mock_response


class TestSearchCatalog:

  @pytest.fixture(autouse=True)
  def setup_mocks(self):
    self.mock_dataplex_client = mock.MagicMock(
        spec=dataplex_v1.CatalogServiceClient
    )
    self.mock_get_dataplex_client = mock.patch.object(
        client, "get_dataplex_catalog_client", autospec=True
    ).start()
    self.mock_get_dataplex_client.return_value = self.mock_dataplex_client
    self.mock_search_request = mock.patch.object(
        dataplex_v1, "SearchEntriesRequest", autospec=True
    ).start()

    yield

    mock.patch.stopall()

  def test_search_catalog_success(self):
    """Test the successful path of search_catalog."""
    creds = _mock_creds()
    settings = _mock_settings()
    prompt = "customer data"
    project_id = "test-project"

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

    result = search_tool.search_catalog(prompt, project_id, creds, settings)

    assert result["status"] == "SUCCESS"
    assert len(result["results"]) == 1
    assert result["results"][0]["name"] == "entry1"
    assert result["results"][0]["display_name"] == "Cust Table"

    self.mock_get_dataplex_client.assert_called_once_with(
        credentials=creds, user_agent=["test-app", "search_catalog"]
    )

    expected_query = (
        '(customer data) AND projectid="test-project" AND system=BIGQUERY'
    )
    self.mock_search_request.assert_called_once_with(
        name=f"projects/{project_id}/locations/global",
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
        "test", "", _mock_creds(), _mock_settings()
    )
    assert result["status"] == "ERROR"
    assert "project_id must be provided" in result["error_details"]
    self.mock_get_dataplex_client.assert_not_called()

  def test_search_catalog_api_error(self):
    """Test search_catalog handling API exceptions."""
    self.mock_dataplex_client.search_entries.side_effect = (
        api_exceptions.BadRequest("Invalid query")
    )

    result = search_tool.search_catalog(
        "test", "test-project", _mock_creds(), _mock_settings()
    )
    assert result["status"] == "ERROR"
    assert "Dataplex API Error: Invalid query" in result["error_details"]

  def test_search_catalog_other_exception(self):
    """Test search_catalog handling unexpected exceptions."""
    self.mock_get_dataplex_client.side_effect = Exception(
        "Something went wrong"
    )

    result = search_tool.search_catalog(
        "test", "test-project", _mock_creds(), _mock_settings()
    )
    assert result["status"] == "ERROR"
    assert "Something went wrong" in result["error_details"]

  @pytest.mark.parametrize(
      "prompt, project_ids, dataset_ids, types, expected_query_part",
      [
          ("p", None, None, None, 'projectid="test-project"'),
          ("p", ["proj1"], None, None, 'projectid="proj1"'),
          ("p", ["p1", "p2"], None, None, '(projectid="p1" OR projectid="p2")'),
          ("p", None, None, ["TABLE"], 'type="TABLE"'),
          (
              "p",
              None,
              None,
              ["TABLE", "DATASET"],
              '(type="TABLE" OR type="DATASET")',
          ),
      ],
  )
  def test_search_catalog_query_construction(
      self, prompt, project_ids, dataset_ids, types, expected_query_part
  ):
    """Test different query constructions based on filters."""
    search_tool.search_catalog(
        prompt,
        "test-project",
        _mock_creds(),
        _mock_settings(),
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
    search_tool.search_catalog("test", "test-project", creds, settings)

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
        prompt,
        project_id,
        creds,
        settings,
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

    # Call the tool
    result = search_tool.search_catalog(
        prompt, project_id, creds, settings, location=location
    )

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

    # Assert the output is processed correctly
    assert result["status"] == "SUCCESS"
    assert len(result["results"]) == 2
    assert result["results"][0]["display_name"] == "uk_football_premiership"
    assert result["results"][1]["display_name"] == "serie_a_matches"
    assert "UK Premier League" in result["results"][0]["description"]

  def test_query_with_project_and_dataset_filters(self):
    creds = _mock_creds()
    settings = _mock_settings()
    project_id = "proj1"
    location = "us-central1"  # Using a specific location

    search_tool.search_catalog(
        prompt="inventory",
        project_id=project_id,
        credentials=creds,
        settings=settings,
        project_ids_filter=["proj1", "proj2"],
        dataset_ids_filter=["dsetA"],
        location=location,
    )

    self.mock_get_dataplex_client.assert_called_once_with(
        credentials=creds, user_agent=["test-app", "search_catalog"]
    )

    expected_query = (
        '(inventory) AND (projectid="proj1" OR projectid="proj2") AND'
        ' (linked_resource:"//bigquery.googleapis.com/projects/proj1/datasets/dsetA/*"'
        ' OR linked_resource:"//bigquery.googleapis.com/projects/proj2/datasets/dsetA/*")'
        " AND system=BIGQUERY"
    )
    expected_search_scope = f"projects/{project_id}/locations/{location}"
    self.mock_search_request.assert_called_once_with(
        name=expected_search_scope,
        query=expected_query,
        page_size=10,
        semantic_search=True,
    )

    self.mock_dataplex_client.search_entries.assert_called_once_with(
        request=self.mock_search_request.return_value
    )
