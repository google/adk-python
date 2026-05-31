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

from google.adk.tools import discovery_engine_search_tool
from google.adk.tools.discovery_engine_search_tool import DiscoveryEngineSearchTool
from google.adk.tools.discovery_engine_search_tool import SearchResultMode
from google.api_core import exceptions
from google.cloud import discoveryengine_v1beta as discoveryengine
import pytest

from google import auth


@mock.patch(
    "google.auth.default",
    mock.MagicMock(return_value=("credentials", "project")),
)
class TestDiscoveryEngineSearchTool:
  """Test the DiscoveryEngineSearchTool class."""

  def test_init_with_data_store_id(self):
    """Test initialization with data_store_id."""
    tool = DiscoveryEngineSearchTool(data_store_id="test_data_store")
    assert (
        tool._serving_config == "test_data_store/servingConfigs/default_config"
    )

  def test_init_with_search_engine_id(self):
    """Test initialization with search_engine_id."""
    tool = DiscoveryEngineSearchTool(search_engine_id="test_search_engine")
    assert (
        tool._serving_config
        == "test_search_engine/servingConfigs/default_config"
    )

  def test_init_with_no_ids_raises_error(self):
    """Test that initialization with no IDs raises ValueError."""
    with pytest.raises(
        ValueError,
        match="Either data_store_id or search_engine_id must be specified.",
    ):
      DiscoveryEngineSearchTool()

  def test_init_with_both_ids_raises_error(self):
    """Test that initialization with both IDs raises ValueError."""
    with pytest.raises(
        ValueError,
        match="Either data_store_id or search_engine_id must be specified.",
    ):
      DiscoveryEngineSearchTool(
          data_store_id="test_data_store",
          search_engine_id="test_search_engine",
      )

  def test_init_with_data_store_specs_without_search_engine_id_raises_error(
      self,
  ):
    """Test that data_store_specs without search_engine_id raises ValueError."""
    with pytest.raises(
        ValueError,
        match=(
            "search_engine_id must be specified if data_store_specs is"
            " specified."
        ),
    ):
      DiscoveryEngineSearchTool(
          data_store_id="test_data_store", data_store_specs=[{"id": "123"}]
      )

  @pytest.mark.parametrize(
      ("tool_kwargs", "expected_endpoint"),
      [
          (
              {
                  "data_store_id": (
                      "projects/test/locations/eu/collections/default_collection/"
                      "dataStores/test_data_store"
                  )
              },
              "eu-discoveryengine.googleapis.com",
          ),
          (
              {
                  "search_engine_id": (
                      "projects/test/locations/us/collections/default_collection/"
                      "engines/test_search_engine"
                  )
              },
              "us-discoveryengine.googleapis.com",
          ),
          (
              {
                  "data_store_id": (
                      "projects/test/locations/europe-west1/collections/"
                      "default_collection/dataStores/test_data_store"
                  )
              },
              "europe-west1-discoveryengine.googleapis.com",
          ),
      ],
  )
  @mock.patch.object(discovery_engine_search_tool, "client_options")
  @mock.patch.object(discoveryengine, "SearchServiceClient")
  def test_init_with_regional_location_uses_regional_endpoint(
      self,
      mock_search_client,
      mock_client_options,
      tool_kwargs,
      expected_endpoint,
  ):
    """Test initialization uses the expected regional API endpoint."""
    DiscoveryEngineSearchTool(**tool_kwargs)

    mock_client_options.ClientOptions.assert_called_once_with(
        api_endpoint=expected_endpoint
    )
    mock_search_client.assert_called_once_with(
        credentials="credentials",
        client_options=mock_client_options.ClientOptions.return_value,
    )

  @mock.patch.object(discovery_engine_search_tool, "client_options")
  @mock.patch.object(discoveryengine, "SearchServiceClient")
  def test_init_with_explicit_location_override_uses_input_location(
      self, mock_search_client, mock_client_options
  ):
    """Test initialization uses explicit location when resource has none."""
    DiscoveryEngineSearchTool(
        data_store_id="test_data_store",
        location="eu",
    )

    mock_client_options.ClientOptions.assert_called_once_with(
        api_endpoint="eu-discoveryengine.googleapis.com"
    )
    mock_search_client.assert_called_once_with(
        credentials="credentials",
        client_options=mock_client_options.ClientOptions.return_value,
    )

  @mock.patch.object(discoveryengine, "SearchServiceClient")
  def test_init_with_mismatched_location_raises_error(self, mock_search_client):
    """Test initialization rejects mismatched location overrides."""
    with pytest.raises(
        ValueError,
        match=(
            "location must match the location in data_store_id or "
            "search_engine_id."
        ),
    ):
      DiscoveryEngineSearchTool(
          data_store_id=(
              "projects/test/locations/us/collections/default_collection/"
              "dataStores/test_data_store"
          ),
          location="eu",
      )

    mock_search_client.assert_not_called()

  @mock.patch.object(discoveryengine, "SearchServiceClient")
  def test_init_with_empty_location_raises_error(self, mock_search_client):
    """Test initialization rejects an empty location override."""
    with pytest.raises(
        ValueError, match="location must not be empty if specified."
    ):
      DiscoveryEngineSearchTool(
          data_store_id=(
              "projects/test/locations/us/collections/default_collection/"
              "dataStores/test_data_store"
          ),
          location=" ",
      )

    mock_search_client.assert_not_called()

  @mock.patch.object(discoveryengine, "SearchServiceClient")
  def test_init_with_invalid_override_location_raises_error(
      self, mock_search_client
  ):
    """Test initialization rejects invalid override location characters."""
    with pytest.raises(
        ValueError,
        match="location must contain only letters, digits, and hyphens.",
    ):
      DiscoveryEngineSearchTool(
          data_store_id="test_data_store",
          location="attacker.com#",
      )

    mock_search_client.assert_not_called()

  @mock.patch.object(discoveryengine, "SearchServiceClient")
  def test_init_with_invalid_resource_location_raises_error(
      self, mock_search_client
  ):
    """Test initialization rejects invalid resource location characters."""
    with pytest.raises(
        ValueError,
        match="Invalid location in data_store_id or search_engine_id.",
    ):
      DiscoveryEngineSearchTool(
          data_store_id=(
              "projects/test/locations/attacker.com#/collections/"
              "default_collection/dataStores/test_data_store"
          )
      )

    mock_search_client.assert_not_called()

  @mock.patch.object(discovery_engine_search_tool, "client_options")
  @mock.patch.object(discoveryengine, "SearchServiceClient")
  def test_init_with_global_location_keeps_default_endpoint(
      self, mock_search_client, mock_client_options
  ):
    """Test initialization keeps default API endpoint for global location."""
    DiscoveryEngineSearchTool(
        data_store_id=(
            "projects/test/locations/global/collections/default_collection/"
            "dataStores/test_data_store"
        )
    )

    mock_client_options.ClientOptions.assert_not_called()
    mock_search_client.assert_called_once_with(
        credentials="credentials", client_options=None
    )

  @mock.patch.object(discovery_engine_search_tool, "client_options")
  @mock.patch.object(discoveryengine, "SearchServiceClient")
  def test_init_with_regional_location_and_quota_project_id(
      self, mock_search_client, mock_client_options
  ):
    """Test initialization uses endpoint and quota project id together."""
    mock_credentials = mock.MagicMock()
    mock_credentials.quota_project_id = "test-quota-project"

    with mock.patch.object(
        auth, "default", return_value=(mock_credentials, "project")
    ):
      DiscoveryEngineSearchTool(
          data_store_id=(
              "projects/test/locations/eu/collections/default_collection/"
              "dataStores/test_data_store"
          )
      )

    mock_client_options.ClientOptions.assert_called_once_with(
        api_endpoint="eu-discoveryengine.googleapis.com",
        quota_project_id="test-quota-project",
    )
    mock_search_client.assert_called_once_with(
        credentials=mock_credentials,
        client_options=mock_client_options.ClientOptions.return_value,
    )

  @mock.patch.object(discovery_engine_search_tool, "client_options")
  @mock.patch.object(
      discoveryengine,
      "SearchServiceClient",
  )
  def test_discovery_engine_search_success(
      self, mock_search_client, mock_client_options
  ):
    """Test successful discovery engine search."""
    mock_response = discoveryengine.SearchResponse()
    mock_response.results = [
        discoveryengine.SearchResponse.SearchResult(
            chunk=discoveryengine.Chunk(
                document_metadata={
                    "title": "Test Title",
                    "uri": "gs://test_bucket/test_file",
                    "struct_data": {
                        "key1": "value1",
                        "uri": "http://example.com",
                    },
                },
                content="Test Content",
            )
        )
    ]
    mock_search_client.return_value.search.return_value = mock_response
    mock_credentials = mock.MagicMock()
    mock_credentials.quota_project_id = "test-quota-project"

    with mock.patch.object(
        auth, "default", return_value=(mock_credentials, "project")
    ) as mock_auth:
      tool = DiscoveryEngineSearchTool(data_store_id="test_data_store")
      result = tool.discovery_engine_search("test query")

      assert result["status"] == "success"
      assert len(result["results"]) == 1
      assert result["results"][0]["title"] == "Test Title"
      assert result["results"][0]["url"] == "http://example.com"
      assert result["results"][0]["content"] == "Test Content"
      mock_auth.assert_called_once()
      mock_client_options.ClientOptions.assert_called_once_with(
          quota_project_id="test-quota-project"
      )
      mock_search_client.assert_called_once_with(
          credentials=mock_credentials,
          client_options=mock_client_options.ClientOptions.return_value,
      )

  @mock.patch.object(
      discoveryengine,
      "SearchServiceClient",
  )
  def test_discovery_engine_search_api_error(self, mock_search_client):
    """Test discovery engine search with API error."""
    mock_search_client.return_value.search.side_effect = (
        exceptions.GoogleAPICallError("API error")
    )

    tool = DiscoveryEngineSearchTool(data_store_id="test_data_store")
    result = tool.discovery_engine_search("test query")

    assert result["status"] == "error"
    assert result["error_message"] == "None API error"

  @mock.patch.object(
      discoveryengine,
      "SearchServiceClient",
  )
  def test_discovery_engine_search_no_results(self, mock_search_client):
    """Test discovery engine search with no results."""
    mock_response = discoveryengine.SearchResponse()
    mock_search_client.return_value.search.return_value = mock_response

    tool = DiscoveryEngineSearchTool(data_store_id="test_data_store")
    result = tool.discovery_engine_search("test query")

    assert result["status"] == "success"
    assert not result["results"]

  def test_init_default_search_result_mode(self):
    """Test default search result mode is None (auto-detect)."""
    tool = DiscoveryEngineSearchTool(data_store_id="test_data_store")
    assert tool._search_result_mode is None

  def test_init_with_documents_mode(self):
    """Test initialization with DOCUMENTS search result mode."""
    tool = DiscoveryEngineSearchTool(
        data_store_id="test_data_store",
        search_result_mode=SearchResultMode.DOCUMENTS,
    )
    assert tool._search_result_mode == SearchResultMode.DOCUMENTS

  @mock.patch.object(
      discoveryengine,
      "SearchServiceClient",
  )
  def test_discovery_engine_search_documents_structured(
      self, mock_search_client
  ):
    """Test DOCUMENTS mode with structured data."""
    mock_doc = discoveryengine.Document(
        name="projects/p/locations/l/doc1",
        id="doc1",
        struct_data={
            "title": "Jira Issue",
            "uri": "https://jira.example.com/123",
            "summary": "Bug fix for login",
        },
    )
    mock_response = discoveryengine.SearchResponse()
    mock_response.results = [
        discoveryengine.SearchResponse.SearchResult(document=mock_doc)
    ]
    mock_search_client.return_value.search.return_value = mock_response

    tool = DiscoveryEngineSearchTool(
        data_store_id="test_data_store",
        search_result_mode=SearchResultMode.DOCUMENTS,
    )
    result = tool.discovery_engine_search("test query")

    assert result["status"] == "success"
    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "Jira Issue"
    assert result["results"][0]["url"] == "https://jira.example.com/123"
    assert "Bug fix for login" in result["results"][0]["content"]

  @mock.patch.object(
      discoveryengine,
      "SearchServiceClient",
  )
  def test_discovery_engine_search_documents_unstructured(
      self, mock_search_client
  ):
    """Test DOCUMENTS mode with unstructured data."""
    mock_doc = discoveryengine.Document(
        name="projects/p/locations/l/doc2",
        id="doc2",
        derived_struct_data={
            "title": "Web Page",
            "link": "https://example.com",
            "snippets": [{"snippet": "Relevant text here"}],
        },
    )
    mock_response = discoveryengine.SearchResponse()
    mock_response.results = [
        discoveryengine.SearchResponse.SearchResult(document=mock_doc)
    ]
    mock_search_client.return_value.search.return_value = mock_response

    tool = DiscoveryEngineSearchTool(
        data_store_id="test_data_store",
        search_result_mode=SearchResultMode.DOCUMENTS,
    )
    result = tool.discovery_engine_search("test query")

    assert result["status"] == "success"
    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "Web Page"
    assert result["results"][0]["url"] == "https://example.com"
    assert "Relevant text here" in result["results"][0]["content"]

  @mock.patch.object(
      discoveryengine,
      "SearchServiceClient",
  )
  def test_discovery_engine_search_documents_no_results(
      self, mock_search_client
  ):
    """Test DOCUMENTS mode with no results."""
    mock_response = discoveryengine.SearchResponse()
    mock_search_client.return_value.search.return_value = mock_response

    tool = DiscoveryEngineSearchTool(
        data_store_id="test_data_store",
        search_result_mode=SearchResultMode.DOCUMENTS,
    )
    result = tool.discovery_engine_search("test query")

    assert result["status"] == "success"
    assert not result["results"]

  @mock.patch.object(
      discoveryengine,
      "SearchServiceClient",
  )
  def test_auto_detect_falls_back_to_documents(self, mock_search_client):
    """Test auto-detect retries with DOCUMENTS on structured store error."""
    structured_error = exceptions.InvalidArgument(
        "`content_search_spec.search_result_mode` must be set to"
        " SearchRequest.ContentSearchSpec.SearchResultMode.DOCUMENTS"
        " when the engine contains structured data store."
    )
    mock_doc = discoveryengine.Document(
        name="projects/p/locations/l/doc1",
        id="doc1",
        struct_data={
            "title": "Jira Issue",
            "uri": "https://jira.example.com/123",
            "summary": "Bug fix",
        },
    )
    mock_doc_response = discoveryengine.SearchResponse()
    mock_doc_response.results = [
        discoveryengine.SearchResponse.SearchResult(document=mock_doc)
    ]
    mock_search_client.return_value.search.side_effect = [
        structured_error,
        mock_doc_response,
    ]

    tool = DiscoveryEngineSearchTool(data_store_id="test_data_store")
    result = tool.discovery_engine_search("test query")

    assert result["status"] == "success"
    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "Jira Issue"
    assert mock_search_client.return_value.search.call_count == 2
    # Mode should be persisted so subsequent calls skip the retry.
    assert tool._search_result_mode == SearchResultMode.DOCUMENTS

  @mock.patch.object(
      discoveryengine,
      "SearchServiceClient",
  )
  def test_auto_detect_does_not_retry_on_unrelated_error(
      self, mock_search_client
  ):
    """Test auto-detect does not retry on unrelated API errors."""
    mock_search_client.return_value.search.side_effect = (
        exceptions.GoogleAPICallError("Permission denied")
    )

    tool = DiscoveryEngineSearchTool(data_store_id="test_data_store")
    result = tool.discovery_engine_search("test query")

    assert result["status"] == "error"
    assert "Permission denied" in result["error_message"]
    assert mock_search_client.return_value.search.call_count == 1
