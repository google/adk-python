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

import importlib.metadata
import logging
import os
from typing import Any
from typing import Optional

import httpx

from ..version import __version__ as _ADK_VERSION
from .function_tool import FunctionTool

logger = logging.getLogger('google_adk.' + __name__)

_PERPLEXITY_SEARCH_URL = 'https://api.perplexity.ai/search'
_INTEGRATION_SLUG = 'google-adk'
_PACKAGE_NAME = 'google-adk'
_DEFAULT_TIMEOUT_SECONDS = 30.0


def _resolve_package_version() -> str:
  """Returns the installed ADK package version, falling back to the in-tree version."""
  try:
    return importlib.metadata.version(_PACKAGE_NAME)
  except importlib.metadata.PackageNotFoundError:
    return _ADK_VERSION


class PerplexitySearchTool(FunctionTool):
  """Tool that performs web search via the Perplexity Search API.

  This tool wraps the `POST https://api.perplexity.ai/search` endpoint and
  exposes a single `query` argument to the model, while letting the developer
  pin server-side options (recency, domain filter, max results, etc.) at
  construction time.

  This is the recommended general-purpose web search tool when you do not
  have a Google Cloud / Vertex AI Discovery Engine data store set up:
  `DiscoveryEngineSearchTool` requires a GCP project with a Discovery Engine
  app and indexed corpus, whereas `PerplexitySearchTool` works against the
  public web with a single `PERPLEXITY_API_KEY` environment variable.

  See https://docs.perplexity.ai/api-reference/search-post for the request
  and response schema.

  Example:

    ```python
    from google.adk.agents import LlmAgent
    from google.adk.tools.perplexity_search_tool import PerplexitySearchTool

    perplexity_search = PerplexitySearchTool()
    agent = LlmAgent(
        model='gemini-2.5-flash',
        name='research_agent',
        tools=[perplexity_search],
    )
    ```
  """

  def __init__(
      self,
      api_key: Optional[str] = None,
      *,
      max_results: Optional[int] = None,
      max_tokens_per_page: Optional[int] = None,
      country: Optional[str] = None,
      search_recency_filter: Optional[str] = None,
      search_domain_filter: Optional[list[str]] = None,
      search_language_filter: Optional[list[str]] = None,
      last_updated_after_filter: Optional[str] = None,
      last_updated_before_filter: Optional[str] = None,
      search_after_date_filter: Optional[str] = None,
      search_before_date_filter: Optional[str] = None,
      timeout: float = _DEFAULT_TIMEOUT_SECONDS,
  ):
    """Initializes the PerplexitySearchTool.

    Args:
      api_key: The Perplexity API key. If not provided, the value of the
        `PERPLEXITY_API_KEY` environment variable is used.
      max_results: Maximum number of results to return (1-20). Defaults to the
        API default (10) when not set.
      max_tokens_per_page: Maximum tokens per page (1-1,000,000). Defaults to
        the API default (4096) when not set.
      country: Optional ISO 3166-1 alpha-2 country code for localization.
      search_recency_filter: Optional recency filter. One of `hour`, `day`,
        `week`, `month`, or `year`.
      search_domain_filter: Optional list of up to 20 domains to restrict the
        search to.
      search_language_filter: Optional list of ISO 639-1 language codes.
      last_updated_after_filter: Optional `MM/DD/YYYY` lower bound on the
        `last_updated` field of results.
      last_updated_before_filter: Optional `MM/DD/YYYY` upper bound on the
        `last_updated` field of results.
      search_after_date_filter: Optional `MM/DD/YYYY` lower bound on the
        result publication date.
      search_before_date_filter: Optional `MM/DD/YYYY` upper bound on the
        result publication date.
      timeout: HTTP timeout in seconds for each search request.

    Raises:
      ValueError: If no API key is supplied and `PERPLEXITY_API_KEY` is not
        set in the environment.
    """
    super().__init__(self.perplexity_search)
    resolved_api_key = api_key or os.environ.get('PERPLEXITY_API_KEY')
    if not resolved_api_key:
      raise ValueError(
          'Perplexity API key is required: pass `api_key` to '
          'PerplexitySearchTool or set the PERPLEXITY_API_KEY '
          'environment variable.'
      )
    self._api_key = resolved_api_key
    self._max_results = max_results
    self._max_tokens_per_page = max_tokens_per_page
    self._country = country
    self._search_recency_filter = search_recency_filter
    self._search_domain_filter = search_domain_filter
    self._search_language_filter = search_language_filter
    self._last_updated_after_filter = last_updated_after_filter
    self._last_updated_before_filter = last_updated_before_filter
    self._search_after_date_filter = search_after_date_filter
    self._search_before_date_filter = search_before_date_filter
    self._timeout = timeout

  def _build_headers(self) -> dict[str, str]:
    return {
        'Authorization': f'Bearer {self._api_key}',
        'Content-Type': 'application/json',
        'X-Pplx-Integration': (
            f'{_INTEGRATION_SLUG}/{_resolve_package_version()}'
        ),
    }

  def _build_body(self, query: str) -> dict[str, Any]:
    body: dict[str, Any] = {'query': query}
    optional_fields: dict[str, Any] = {
        'max_results': self._max_results,
        'max_tokens_per_page': self._max_tokens_per_page,
        'country': self._country,
        'search_recency_filter': self._search_recency_filter,
        'search_domain_filter': self._search_domain_filter,
        'search_language_filter': self._search_language_filter,
        'last_updated_after_filter': self._last_updated_after_filter,
        'last_updated_before_filter': self._last_updated_before_filter,
        'search_after_date_filter': self._search_after_date_filter,
        'search_before_date_filter': self._search_before_date_filter,
    }
    for key, value in optional_fields.items():
      if value is not None:
        body[key] = value
    return body

  async def perplexity_search(self, query: str) -> dict[str, Any]:
    """Searches the web via the Perplexity Search API.

    Args:
      query: The search query.

    Returns:
      A dictionary with `status` set to either `success` or `error`. On
      success, `results` contains a list of result entries with `title`,
      `url`, `snippet`, `date`, and `last_updated` fields, along with the
      raw `id` and `server_time` returned by the API.
    """
    headers = self._build_headers()
    body = self._build_body(query)

    try:
      async with httpx.AsyncClient(timeout=self._timeout) as client:
        response = await client.post(
            _PERPLEXITY_SEARCH_URL, headers=headers, json=body
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as e:
      logger.exception('Perplexity Search request failed.')
      return {
          'status': 'error',
          'error_message': (
              f'Perplexity Search returned HTTP {e.response.status_code}.'
          ),
      }
    except httpx.HTTPError as e:
      logger.exception('Perplexity Search request failed.')
      return {'status': 'error', 'error_message': str(e)}

    return {
        'status': 'success',
        'results': payload.get('results', []),
        'id': payload.get('id'),
        'server_time': payload.get('server_time'),
    }
