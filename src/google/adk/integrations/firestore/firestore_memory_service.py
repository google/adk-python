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
from collections.abc import Mapping
from collections.abc import Sequence
import hashlib
import logging
import re
from typing import Optional
from typing import TYPE_CHECKING

from google.cloud.firestore_v1.base_query import FieldFilter
from typing_extensions import override

from ...memory import _utils
from ...memory.base_memory_service import BaseMemoryService
from ...memory.base_memory_service import SearchMemoryResponse
from ...memory.memory_entry import MemoryEntry
from ._stop_words import DEFAULT_STOP_WORDS

if TYPE_CHECKING:
  from google.cloud import firestore

  from ...events.event import Event
  from ...sessions.session import Session

logger = logging.getLogger("google_adk." + __name__)

DEFAULT_EVENTS_COLLECTION = "events"
DEFAULT_MEMORIES_COLLECTION = "memories"


def _memory_doc_id(
    *, app_name: str, user_id: str, session_id: Optional[str], event_id: str
) -> str:
  """Returns a stable memory document ID for an event.

  Hashed because app names and user IDs may contain characters that are not
  allowed in document IDs, such as "/".
  """
  key = "\x00".join((app_name, user_id, session_id or "", event_id))
  return hashlib.sha256(key.encode("utf-8")).hexdigest()


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
    await self._write_memories(
        app_name=session.app_name,
        user_id=session.user_id,
        session_id=session.id,
        events=session.events,
    )

  @override
  async def add_events_to_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      events: Sequence[Event],
      session_id: Optional[str] = None,
      custom_metadata: Optional[Mapping[str, object]] = None,
  ) -> None:
    """Adds events, such as the latest turn, to the memories collection.

    Re-adding an event with the same session ID overwrites its entry. The
    session ID is part of that entry's ID, so an event added without one and
    later added with one is stored twice; `search_memory` returns one copy.
    """
    _ = custom_metadata
    await self._write_memories(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        events=events,
    )

  async def _write_memories(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      events: Sequence[Event],
  ) -> None:
    """Writes one memory document per event that has text keywords."""
    batch = self.client.batch()
    count = 0

    for event in events:
      if not event.content or not event.content.parts:
        continue

      text = " ".join([part.text for part in event.content.parts if part.text])
      if not text:
        continue

      keywords = self._extract_keywords(text)
      if not keywords:
        continue

      doc_ref = self.client.collection(self.memories_collection).document(
          _memory_doc_id(
              app_name=app_name,
              user_id=user_id,
              session_id=session_id,
              event_id=event.id,
          )
      )
      batch.set(
          doc_ref,
          {
              "appName": app_name,
              "userId": user_id,
              "sessionId": session_id,
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
