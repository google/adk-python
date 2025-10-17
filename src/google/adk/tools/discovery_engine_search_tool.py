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

from __future__ import annotations

from typing import Any
from typing import Optional

from google.api_core.exceptions import GoogleAPICallError
import google.auth
from google.cloud import discoveryengine_v1beta as discoveryengine
from google.genai import types

from .function_tool import FunctionTool


class DiscoveryEngineSearchTool(FunctionTool):
  """Tool for searching the discovery engine."""

  def __init__(
      self,
      data_store_id: Optional[str] = None,
      data_store_specs: Optional[
          list[types.VertexAISearchDataStoreSpec]
      ] = None,
      search_engine_id: Optional[str] = None,
      filter: Optional[str] = None,
      max_results: Optional[int] = None,
  ):
    """Initializes the DiscoveryEngineSearchTool.

    Args:
      data_store_id: The Vertex AI search data store resource ID in the format
        of
        "projects/{project}/locations/{location}/collections/{collection}/dataStores/{dataStore}".
      data_store_specs: Specifications that define the specific DataStores to be
        searched. It should only be set if engine is used.
      search_engine_id: The Vertex AI search engine resource ID in the format of
        "projects/{project}/locations/{location}/collections/{collection}/engines/{engine}".
      filter: The filter to be applied to the search request. Default is None.
      max_results: The maximum number of results to return. Default is None.
    """
    super().__init__(self.discovery_engine_search)
    if (data_store_id is None and search_engine_id is None) or (
        data_store_id is not None and search_engine_id is not None
    ):
      raise ValueError(
          "Either data_store_id or search_engine_id must be specified."
      )
    if data_store_specs is not None and search_engine_id is None:
      raise ValueError(
          "search_engine_id must be specified if data_store_specs is specified."
      )

    self._serving_config = (
        f"{data_store_id or search_engine_id}/servingConfigs/default_config"
    )
    self._data_store_specs = data_store_specs
    self._search_engine_id = search_engine_id
    self._filter = filter
    self._max_results = max_results

    credentials, _ = google.auth.default()
    self._discovery_engine_client = discoveryengine.SearchServiceClient(
        credentials=credentials
    )

  def discovery_engine_search(
      self,
      query: str,
  ) -> dict[str, Any]:
    """Search through Vertex AI Search's discovery engine search API.

    Args:
      query: The search query.

    Returns:
      A dictionary containing the status of the request and the list of search
      results, which contains the title, url and content.
    """
    request = discoveryengine.SearchRequest(
        serving_config=self._serving_config,
        query=query,
        content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
            search_result_mode=discoveryengine.SearchRequest.ContentSearchSpec.SearchResultMode.CHUNKS,
            chunk_spec=discoveryengine.SearchRequest.ContentSearchSpec.ChunkSpec(
                num_previous_chunks=0,
                num_next_chunks=0,
            ),
        ),
    )

    if self._data_store_specs:
      request.data_store_specs = self._data_store_specs
    if self._filter:
      request.filter = self._filter
    if self._max_results:
      request.page_size = self._max_results

    results = []
    try:
      response = self._discovery_engine_client.search(request)
      for item in response.results:
        chunk = item.chunk
        if not chunk:
          continue

        title = ""
        uri = ""
        doc_metadata = chunk.document_metadata
        if doc_metadata:
          title = doc_metadata.title
          uri = doc_metadata.uri
          # Prioritize URI from struct_data if it exists.
          if doc_metadata.struct_data and "uri" in doc_metadata.struct_data:
            uri = doc_metadata.struct_data["uri"]

        results.append({
            "title": title,
            "url": uri,
            "content": chunk.content,
        })
    except GoogleAPICallError as e:
      return {"status": "error", "error_message": str(e)}
    return {"status": "success", "results": results}
