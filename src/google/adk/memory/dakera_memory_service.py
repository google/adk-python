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

"""DakeraMemoryService — persistent, decay-weighted cross-session memory for ADK
via a self-hosted Dakera server.

See https://dakera.ai for server setup instructions.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import httpx
from google.genai import types as genai_types
from typing_extensions import override

from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..sessions.session import Session

logger = logging.getLogger(__name__)


class DakeraMemoryService(BaseMemoryService):
  """Persistent, decay-weighted cross-session memory backed by a self-hosted
  Dakera server (https://dakera.ai).

  Dakera exposes a REST API for storing and semantically searching memories
  across agent sessions. Unlike in-memory alternatives, memories survive process
  restarts and are ranked by recency and access frequency using an
  access-weighted decay model.

  Usage::

      from google.adk.memory import DakeraMemoryService

      memory_service = DakeraMemoryService(
          base_url="http://localhost:3300",  # or set DAKERA_API_URL
          api_key="sk-...",                  # or set DAKERA_API_KEY
          top_k=10,
      )

  The server can be started via Docker::

      docker run -p 3300:3300 ghcr.io/dakera-ai/dakera:latest

  Or self-hosted — see https://dakera.ai for full deployment instructions.
  """

  def __init__(
      self,
      *,
      base_url: str | None = None,
      api_key: str | None = None,
      top_k: int = 10,
  ) -> None:
    """Initialises the Dakera memory service client.

    Args:
        base_url: Base URL of the Dakera server.  Defaults to the
            ``DAKERA_API_URL`` environment variable, falling back to
            ``http://localhost:3300``.
        api_key: API key for authenticating with the Dakera server.  Defaults
            to the ``DAKERA_API_KEY`` environment variable.
        top_k: Maximum number of memories to return from a search.
    """
    self._base_url = (
        base_url or os.environ.get("DAKERA_API_URL", "http://localhost:3300")
    ).rstrip("/")
    self._api_key = api_key or os.environ.get("DAKERA_API_KEY", "")
    self._top_k = top_k

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if self._api_key:
      headers["Authorization"] = f"Bearer {self._api_key}"

    self._client = httpx.AsyncClient(
        base_url=self._base_url,
        headers=headers,
        timeout=30.0,
    )

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    """Stores all meaningful events from *session* into Dakera memory.

    Each event that carries text content is written to ``POST /v1/memories``
    with the session, app, and user identifiers preserved as metadata so that
    searches can be scoped correctly later.

    Args:
        session: The session whose events should be persisted.
    """
    for event in session.events:
      if not event.content or not event.content.parts:
        continue

      text_parts = [
          part.text
          for part in event.content.parts
          if hasattr(part, "text") and part.text
      ]
      if not text_parts:
        continue

      role = getattr(event, "author", None) or "unknown"
      combined_text = " ".join(text_parts)
      content_str = f"{role}: {combined_text}"

      payload: dict[str, object] = {
          "content": content_str,
          "session_id": session.id,
          "metadata": {
              "app_name": session.app_name,
              "user_id": session.user_id,
          },
      }

      try:
        response = await self._client.post("/v1/memories", json=payload)
        response.raise_for_status()
      except httpx.HTTPError as exc:
        logger.warning(
            "DakeraMemoryService: failed to store memory for session %s: %s",
            session.id,
            exc,
        )

  @override
  async def search_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      query: str,
  ) -> SearchMemoryResponse:
    """Semantically searches Dakera for memories matching *query*.

    Calls ``POST /v1/memories/search`` and maps each result to a
    :class:`~google.adk.memory.MemoryEntry` wrapped inside a
    :class:`~google.adk.memory.base_memory_service.SearchMemoryResponse`.

    Args:
        app_name: Application name used to scope the search.
        user_id: User identifier used to scope the search.
        query: Natural-language query string.

    Returns:
        A :class:`SearchMemoryResponse` containing up to ``top_k`` matching
        memory entries.
    """
    payload: dict[str, object] = {
        "query": query,
        "top_k": self._top_k,
        "filter": {
            "app_name": app_name,
            "user_id": user_id,
        },
    }

    try:
      response = await self._client.post("/v1/memories/search", json=payload)
      response.raise_for_status()
      data = response.json()
    except httpx.HTTPError as exc:
      logger.warning(
          "DakeraMemoryService: search failed for app=%s user=%s: %s",
          app_name,
          user_id,
          exc,
      )
      return SearchMemoryResponse()

    memories: list[MemoryEntry] = []
    for result in data.get("results", []):
      content_text: str = result.get("content", "")
      if not content_text:
        continue

      genai_content = genai_types.Content(
          parts=[genai_types.Part(text=content_text)],
          role="user",
      )
      memories.append(
          MemoryEntry(
              content=genai_content,
              id=result.get("id"),
              timestamp=result.get("created_at"),
              custom_metadata={
                  k: v
                  for k, v in result.get("metadata", {}).items()
                  if k not in ("app_name", "user_id")
              },
          )
      )

    return SearchMemoryResponse(memories=memories)
