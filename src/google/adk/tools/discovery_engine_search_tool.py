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

from collections.abc import Mapping
import enum
import json
import logging
import re
from typing import Any
from typing import Optional

from google.api_core import client_options
from google.api_core.exceptions import GoogleAPICallError
import google.auth
from google.cloud import discoveryengine_v1beta as discoveryengine
from google.genai import types

from .function_tool import FunctionTool

logger = logging.getLogger('google_adk.' + __name__)

_STRUCTURED_STORE_ERROR_PATTERN = re.compile(
    r'search_result_mode.*DOCUMENTS', re.IGNORECASE
)

_DEFAULT_ENDPOINT = 'discoveryengine.googleapis.com'
_GLOBAL_LOCATION = 'global'
_LOCATION_PATTERN = re.compile(
    r'/locations/([a-z0-9-]+)(?:/|$)', flags=re.IGNORECASE
)
_VALID_LOCATION_PATTERN = re.compile(r'^[a-z0-9-]+$')


def _normalize_location(location: str, location_type: str) -> str:
  """Normalizes and validates a location value."""
  normalized_location = location.strip().lower()
  if not normalized_location:
    raise ValueError(f'{location_type} must not be empty if specified.')
  if not _VALID_LOCATION_PATTERN.fullmatch(normalized_location):
    raise ValueError(
        f'{location_type} must contain only letters, digits, and hyphens.'
    )
  return normalized_location


def _extract_resource_location(resource_id: str) -> Optional[str]:
  """Extracts and validates location from a resource id."""
  if '/locations/' not in resource_id.lower():
    return None

  location_match = _LOCATION_PATTERN.search(resource_id)
  if not location_match:
    raise ValueError('Invalid location in data_store_id or search_engine_id.')
  return _normalize_location(location_match.group(1), 'resource location')


def _resolve_location(resource_id: str, location: Optional[str]) -> str:
  """Resolves the Discovery Engine location to use for the endpoint."""
  inferred_location = _extract_resource_location(resource_id)

  if location is not None:
    normalized_location = _normalize_location(location, 'location')
    if inferred_location and normalized_location != inferred_location:
      raise ValueError(
          'location must match the location in data_store_id or '
          'search_engine_id.'
      )
    return normalized_location

  if inferred_location:
    return inferred_location
  return _GLOBAL_LOCATION


def _build_client_options(
    resource_id: str,
    quota_project_id: Optional[str],
    location: Optional[str],
) -> Optional[client_options.ClientOptions]:
  """Builds client options for Discovery Engine requests."""
  client_options_kwargs = {}
  resolved_location = _resolve_location(resource_id, location)

  if resolved_location != _GLOBAL_LOCATION:
    client_options_kwargs['api_endpoint'] = (
        f'{resolved_location}-{_DEFAULT_ENDPOINT}'
    )
  if quota_project_id:
    client_options_kwargs['quota_project_id'] = quota_project_id

  if not client_options_kwargs:
    return None
  return client_options.ClientOptions(**client_options_kwargs)


class SearchResultMode(enum.Enum):
  """Search result mode for discovery engine search."""

  CHUNKS = 'CHUNKS'
  """Results as chunks (default). Works for unstructured data."""

  DOCUMENTS = 'DOCUMENTS'
  """Results as documents. Required for structured datastores."""


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
      *,
      search_result_mode: Optional[SearchResultMode] = None,
      location: Optional[str] = None,
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
      search_result_mode: The search result mode. When None (default),
        automatically detects the correct mode by trying CHUNKS first and
        falling back to DOCUMENTS if the datastore requires it. Set explicitly
        to CHUNKS or DOCUMENTS to skip auto-detection.
      location: Optional endpoint location override.
        Examples: "global", "us", "eu". If not specified, location is inferred
          from `data_store_id` or `search_engine_id` and defaults to "global".
    """
    super().__init__(self.discovery_engine_search)
    if (data_store_id is None and search_engine_id is None) or (
        data_store_id is not None and search_engine_id is not None
    ):
      raise ValueError(
          'Either data_store_id or search_engine_id must be specified.'
      )
    if data_store_specs is not None and search_engine_id is None:
      raise ValueError(
          'search_engine_id must be specified if data_store_specs is specified.'
      )

    self._serving_config = (
        f'{data_store_id or search_engine_id}/servingConfigs/default_config'
    )
    self._data_store_specs = data_store_specs
    self._search_engine_id = search_engine_id
    self._filter = filter
    self._max_results = max_results
    self._search_result_mode = search_result_mode
    self._location = location

    credentials, _ = google.auth.default()
    quota_project_id = getattr(credentials, 'quota_project_id', None)
    resource_id = data_store_id or search_engine_id or ''
    options = _build_client_options(
        resource_id=resource_id,
        quota_project_id=quota_project_id,
        location=location,
    )
    self._discovery_engine_client = discoveryengine.SearchServiceClient(
        credentials=credentials, client_options=options
    )

  def discovery_engine_search(
      self,
      query: str,
  ) -> dict[str, Any]:
    """Search through Vertex AI Search's discovery engine search API.

    Args:
      query: The search query.

    Returns:
      A dictionary containing the status of the request and the list of
      search results, which contains the title, url and content.
    """
    try:
      mode = self._search_result_mode
      if mode is not None:
        return self._do_search(query, mode)

      # Auto-detect: try CHUNKS first, fall back to DOCUMENTS
      # if the datastore requires it.
      try:
        return self._do_search(query, SearchResultMode.CHUNKS)
      except GoogleAPICallError as e:
        if _STRUCTURED_STORE_ERROR_PATTERN.search(str(e)):
          logger.info(
              'CHUNKS mode failed for structured datastore,'
              ' retrying with DOCUMENTS mode.'
          )
          self._search_result_mode = SearchResultMode.DOCUMENTS
          return self._do_search(query, SearchResultMode.DOCUMENTS)
        raise
    except GoogleAPICallError as e:
      return {'status': 'error', 'error_message': str(e)}

  def _do_search(
      self,
      query: str,
      mode: SearchResultMode,
  ) -> dict[str, Any]:
    """Executes a search request with the given mode.

    Raises:
      GoogleAPICallError: If the search API call fails.
    """
    content_search_spec = self._build_content_search_spec(mode)
    request = discoveryengine.SearchRequest(
        serving_config=self._serving_config,
        query=query,
        content_search_spec=content_search_spec,
    )

    if self._data_store_specs:
      request.data_store_specs = self._data_store_specs
    if self._filter:
      request.filter = self._filter
    if self._max_results:
      request.page_size = self._max_results

    results = []
    response = self._discovery_engine_client.search(request)
    for item in response.results:
      if mode == SearchResultMode.DOCUMENTS:
        doc = item.document
        if not doc:
          continue
        results.append(self._parse_document_result(doc))
      else:
        chunk = item.chunk
        if not chunk:
          continue
        results.append(self._parse_chunk_result(chunk))
    return {'status': 'success', 'results': results}

  def _build_content_search_spec(
      self,
      mode: SearchResultMode,
  ) -> discoveryengine.SearchRequest.ContentSearchSpec:
    """Builds the ContentSearchSpec based on the search result mode."""
    spec_cls = discoveryengine.SearchRequest.ContentSearchSpec
    if mode == SearchResultMode.DOCUMENTS:
      return spec_cls(
          search_result_mode=spec_cls.SearchResultMode.DOCUMENTS,
      )
    return spec_cls(
        search_result_mode=spec_cls.SearchResultMode.CHUNKS,
        chunk_spec=spec_cls.ChunkSpec(
            num_previous_chunks=0,
            num_next_chunks=0,
        ),
    )

  def _parse_chunk_result(self, chunk: discoveryengine.Chunk) -> dict[str, str]:
    """Parses a chunk search result into a dict."""
    title = ''
    uri = ''
    doc_metadata = chunk.document_metadata
    if doc_metadata:
      title = doc_metadata.title
      uri = doc_metadata.uri
      # Prioritize URI from struct_data if it exists.
      if doc_metadata.struct_data and 'uri' in doc_metadata.struct_data:
        uri = doc_metadata.struct_data['uri']
    return {
        'title': title,
        'url': uri,
        'content': chunk.content,
    }

  def _parse_document_result(
      self, doc: discoveryengine.Document
  ) -> dict[str, str]:
    """Parses a document search result into a dict."""
    title = ''
    uri = ''
    content = ''

    # Structured data: fields are in struct_data.
    if doc.struct_data:
      data = dict(doc.struct_data)
      title = data.pop('title', '')
      uri = data.pop('uri', data.pop('link', ''))
      content = json.dumps(data)
    # Unstructured data: fields are in derived_struct_data.
    elif doc.derived_struct_data:
      data = dict(doc.derived_struct_data)
      title = data.get('title', '')
      uri = data.get('link', '')
      snippets = data.get('snippets', [])
      if snippets:
        snippet_texts = []
        for s in snippets:
          s_snippet = s.get('snippet') if isinstance(s, Mapping) else None
          if s_snippet:
            snippet_texts.append(str(s_snippet))
          else:
            snippet_texts.append(str(s))
        content = '\n'.join(snippet_texts)
      extractive_answers = data.get('extractive_answers', [])
      if not content and extractive_answers:
        content = '\n'.join(str(a) for a in extractive_answers)

    return {
        'title': title,
        'url': uri,
        'content': content,
    }
