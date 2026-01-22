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

  @mock.patch(
      "google.cloud.discoveryengine_v1beta.SearchServiceClient",
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

  @mock.patch(
      "google.cloud.discoveryengine_v1beta.SearchServiceClient",
  )
  def test_discovery_engine_search_no_results(self, mock_search_client):
    """Test discovery engine search with no results."""
    mock_response = discoveryengine.SearchResponse()
    mock_search_client.return_value.search.return_value = mock_response

    tool = DiscoveryEngineSearchTool(data_store_id="test_data_store")
    result = tool.discovery_engine_search("test query")

    assert result["status"] == "success"
    assert not result["results"]
