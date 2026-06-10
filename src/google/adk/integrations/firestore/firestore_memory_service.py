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

import asyncio
import logging
import os
import re
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from google.cloud.firestore_v1.base_query import FieldFilter
from typing_extensions import override

from ...events.event import Event
from ...memory import _utils
from ...memory.base_memory_service import BaseMemoryService
from ...memory.base_memory_service import SearchMemoryResponse
from ...memory.memory_entry import MemoryEntry
from ._stop_words import DEFAULT_STOP_WORDS

if TYPE_CHECKING:
  from google.cloud import firestore

  from ...sessions.session import Session

logger = logging.getLogger("google_adk." + __name__)

DEFAULT_EVENTS_COLLECTION = "events"
DEFAULT_MEMORIES_COLLECTION = "memories"


class FirestoreMemoryService(BaseMemoryService):  # type: ignore[misc]
  """Memory service that uses Google Cloud Firestore as the backend.

  It uses the existing session data to create memories in a top-level memory collection.
  """

  def __init__(
      self,
      client: Optional[firestore.AsyncClient] = None,
      events_collection: Optional[str] = None,
      stop_words: Optional[set[str]] = None,
      memories_collection: Optional[str] = None,
  ):
    """Initializes the Firestore memory service.

    Args:
      client: An optional Firestore AsyncClient. If not provided, a new one
        will be created.
      events_collection: The name of the events collection or collection group.
        Defaults to 'events'.
      stop_words: A set of words to ignore when extracting keywords. Defaults to
        a standard English stop words list.
      memories_collection: The name of the memories collection. Defaults to
        'memories'.
    """
    if client is None:
      from google.cloud import firestore

      self.client = firestore.AsyncClient()
    else:
      self.client = client
    self.events_collection = events_collection or DEFAULT_EVENTS_COLLECTION
    self.memories_collection = (
        memories_collection or DEFAULT_MEMORIES_COLLECTION
    )
    self.stop_words = (
        stop_words if stop_words is not None else DEFAULT_STOP_WORDS
    )

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    """Extracts keywords from session events and stores them in the memories collection."""
    batch = self.client.batch()
    count = 0

    for event in session.events:
      if not event.content or not event.content.parts:
        continue

      text = " ".join([part.text for part in event.content.parts if part.text])
      if not text:
        continue

      keywords = self._extract_keywords(text)
      if not keywords:
        continue

      doc_ref = self.client.collection(self.memories_collection).document()
      batch.set(
          doc_ref,
          {
              "appName": session.app_name,
              "userId": session.user_id,
              "keywords": list(keywords),
              "author": event.author,
              "content": event.content.model_dump(
                  exclude_none=True, mode="json"
              ),
              "timestamp": event.timestamp,
          },
      )
      count += 1
      if count >= 500:
        await batch.commit()
        batch = self.client.batch()
        count = 0

    if count > 0:
      await batch.commit()

  def _extract_keywords(self, text: str) -> set[str]:
    """Extracts keywords from text, ignoring stop words."""
    words = re.findall(r"[A-Za-z]+", text.lower())
    return {word for word in words if word not in self.stop_words}

  async def _search_by_keyword(
      self, app_name: str, user_id: str, keyword: str
  ) -> list[MemoryEntry]:
    """Searches for events matching a single keyword."""
    query = (
        self.client.collection(self.memories_collection)
        .where(filter=FieldFilter("appName", "==", app_name))
        .where(filter=FieldFilter("userId", "==", user_id))
        .where(filter=FieldFilter("keywords", "array_contains", keyword))
    )

    docs = await query.get()
    entries = []
    for doc in docs:
      data = doc.to_dict()
      if data and "content" in data:
        try:
          from google.genai import types

          content = types.Content.model_validate(data["content"])
          entries.append(
              MemoryEntry(
                  content=content,
                  author=data.get("author", ""),
                  timestamp=_utils.format_timestamp(data.get("timestamp", 0.0)),
              )
          )
        except Exception as e:
          logger.warning(f"Failed to parse memory entry: {e}")

    return entries

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Searches memory for events matching the query."""
    keywords = self._extract_keywords(query)
    if not keywords:
      return SearchMemoryResponse()

    tasks = [
        self._search_by_keyword(app_name, user_id, keyword)
        for keyword in keywords
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    seen = set()
    memories = []
    for result_list in results:
      if isinstance(result_list, BaseException):
        logger.warning(f"Memory keyword search partial failure: {result_list}")
        continue
      for entry in result_list:
        content_text = ""
        if entry.content and entry.content.parts:
          content_text = " ".join(
              [part.text for part in entry.content.parts if part.text]
          )
        key = (entry.author, content_text, entry.timestamp)
        if key not in seen:
          seen.add(key)
          memories.append(entry)

    return SearchMemoryResponse(memories=memories)
