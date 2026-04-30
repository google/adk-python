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

import json
from typing import Any

from google.adk.tools.perplexity_search_tool import PerplexitySearchTool
import httpx
import pytest

_FAKE_PAYLOAD: dict[str, Any] = {
    "results": [{
        "title": "Example",
        "url": "https://example.com",
        "snippet": "An example result.",
        "date": "2026-01-01",
        "last_updated": "2026-01-02",
    }],
    "id": "search-id-123",
    "server_time": "2026-04-30T00:00:00Z",
}


class _CapturedRequest:
  """Holds the most recent request seen by the fake transport."""

  def __init__(self) -> None:
    self.url: str | None = None
    self.headers: httpx.Headers | None = None
    self.json: dict[str, Any] | None = None
    self.method: str | None = None


def _make_transport(
    *,
    captured: _CapturedRequest,
    payload: dict[str, Any] | None = None,
    status_code: int = 200,
    raise_exc: Exception | None = None,
) -> httpx.MockTransport:
  """Builds a MockTransport that captures the outgoing request and returns a fixed response."""

  def handler(request: httpx.Request) -> httpx.Response:
    captured.method = request.method
    captured.url = str(request.url)
    captured.headers = request.headers
    captured.json = json.loads(request.content) if request.content else None
    if raise_exc is not None:
      raise raise_exc
    return httpx.Response(status_code, json=payload or {})

  return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _patch_async_client(monkeypatch):
  """Routes every httpx.AsyncClient created by the tool through a MockTransport."""

  state: dict[str, Any] = {
      "captured": _CapturedRequest(),
      "payload": _FAKE_PAYLOAD,
      "status_code": 200,
      "raise_exc": None,
  }

  original_init = httpx.AsyncClient.__init__

  def _patched_init(self, *args, **kwargs):
    kwargs["transport"] = _make_transport(
        captured=state["captured"],
        payload=state["payload"],
        status_code=state["status_code"],
        raise_exc=state["raise_exc"],
    )
    return original_init(self, *args, **kwargs)

  monkeypatch.setattr(httpx.AsyncClient, "__init__", _patched_init)
  return state


@pytest.mark.asyncio
async def test_init_requires_api_key(monkeypatch):
  monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
  with pytest.raises(ValueError, match="Perplexity API key"):
    PerplexitySearchTool()


@pytest.mark.asyncio
async def test_init_reads_api_key_from_environment(monkeypatch):
  monkeypatch.setenv("PERPLEXITY_API_KEY", "env-key")
  tool = PerplexitySearchTool()
  assert tool._api_key == "env-key"


@pytest.mark.asyncio
async def test_init_prefers_explicit_api_key(monkeypatch):
  monkeypatch.setenv("PERPLEXITY_API_KEY", "env-key")
  tool = PerplexitySearchTool(api_key="explicit-key")
  assert tool._api_key == "explicit-key"


@pytest.mark.asyncio
async def test_search_sends_attribution_header(_patch_async_client):
  tool = PerplexitySearchTool(api_key="test-key")
  result = await tool.perplexity_search("hello world")

  captured = _patch_async_client["captured"]
  assert result["status"] == "success"
  assert captured.headers is not None
  integration_header = captured.headers.get("x-pplx-integration")
  assert integration_header is not None
  assert integration_header.startswith("google-adk/")


@pytest.mark.asyncio
async def test_search_sends_bearer_auth_and_json_content_type(
    _patch_async_client,
):
  tool = PerplexitySearchTool(api_key="test-key")
  await tool.perplexity_search("foo")

  captured = _patch_async_client["captured"]
  assert captured.headers["authorization"] == "Bearer test-key"
  assert captured.headers["content-type"] == "application/json"
  assert captured.method == "POST"
  assert captured.url == "https://api.perplexity.ai/search"


@pytest.mark.asyncio
async def test_search_returns_results_payload(_patch_async_client):
  tool = PerplexitySearchTool(api_key="test-key")
  result = await tool.perplexity_search("hello")

  assert result["status"] == "success"
  assert result["results"] == _FAKE_PAYLOAD["results"]
  assert result["id"] == "search-id-123"
  assert result["server_time"] == "2026-04-30T00:00:00Z"


@pytest.mark.asyncio
async def test_search_body_contains_only_query_when_no_options(
    _patch_async_client,
):
  tool = PerplexitySearchTool(api_key="test-key")
  await tool.perplexity_search("hello world")

  captured = _patch_async_client["captured"]
  assert captured.json == {"query": "hello world"}


@pytest.mark.asyncio
async def test_search_body_includes_configured_options(_patch_async_client):
  tool = PerplexitySearchTool(
      api_key="test-key",
      max_results=5,
      max_tokens_per_page=1024,
      country="US",
      search_recency_filter="week",
      search_domain_filter=["example.com", "wikipedia.org"],
      search_language_filter=["en"],
      last_updated_after_filter="01/01/2026",
      last_updated_before_filter="04/30/2026",
      search_after_date_filter="01/01/2025",
      search_before_date_filter="12/31/2025",
  )
  await tool.perplexity_search("topic")

  body = _patch_async_client["captured"].json
  assert body == {
      "query": "topic",
      "max_results": 5,
      "max_tokens_per_page": 1024,
      "country": "US",
      "search_recency_filter": "week",
      "search_domain_filter": ["example.com", "wikipedia.org"],
      "search_language_filter": ["en"],
      "last_updated_after_filter": "01/01/2026",
      "last_updated_before_filter": "04/30/2026",
      "search_after_date_filter": "01/01/2025",
      "search_before_date_filter": "12/31/2025",
  }


@pytest.mark.asyncio
async def test_search_returns_error_on_http_status(_patch_async_client):
  _patch_async_client["status_code"] = 401
  _patch_async_client["payload"] = {"error": "unauthorized"}

  tool = PerplexitySearchTool(api_key="bad-key")
  result = await tool.perplexity_search("anything")

  assert result["status"] == "error"
  assert "401" in result["error_message"]


@pytest.mark.asyncio
async def test_search_returns_error_on_transport_error(_patch_async_client):
  _patch_async_client["raise_exc"] = httpx.ConnectError("boom")

  tool = PerplexitySearchTool(api_key="test-key")
  result = await tool.perplexity_search("anything")

  assert result["status"] == "error"
  assert "boom" in result["error_message"]


@pytest.mark.asyncio
async def test_tool_function_declaration_uses_query_argument(
    _patch_async_client,
):
  tool = PerplexitySearchTool(api_key="test-key")
  declaration = tool._get_declaration()

  assert declaration is not None
  assert declaration.name == "perplexity_search"
  assert declaration.parameters is not None
  properties = declaration.parameters.properties or {}
  assert "query" in properties


@pytest.mark.asyncio
async def test_tool_is_lazily_exported(monkeypatch):
  monkeypatch.setenv("PERPLEXITY_API_KEY", "env-key")
  from google.adk import tools as adk_tools

  tool_cls = adk_tools.PerplexitySearchTool
  instance = tool_cls()
  assert instance.name == "perplexity_search"
