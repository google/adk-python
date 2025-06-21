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

import re
from typing import TYPE_CHECKING, Optional, Any

from rank_bm25 import BM25Okapi
from typing_extensions import override

from . import _utils
from .base_memory_service import BaseMemoryService, SearchMemoryResponse
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..events.event import Event
  from ..sessions.session import Session


def _user_key(app_name: str, user_id: str) -> str:
  return f"{app_name}/{user_id}"


def _tokenize(text: str) -> list[str]:
  if not text:
    return []
  return [word for word in re.split(r"[^a-zA-Z0-9]+", text.lower()) if word]


def _part_has_text(part) -> bool:
  return (
      hasattr(part, "text") and isinstance(part.text, str) and part.text.strip()
  )


class InMemoryBM25RetrievalMemoryService(BaseMemoryService):
  """A non-persistent memory service with BM25 ranking and retrieval."""

  def __init__(self) -> None:
    self._session_events: dict[str, dict[str, list[Event]]] = {}
    self._bm25_cache: dict[str, tuple[Any, list[Event]]] = {}

  @override
  async def add_session_to_memory(self, session: Session):
    user_key = _user_key(session.app_name, session.user_id)
    self._session_events.setdefault(user_key, {})

    valid_events: list[Event] = []
    for event in session.events or []:
      if (
          event.content
          and event.content.parts
          and any(_part_has_text(p) for p in event.content.parts)
      ):
        valid_events.append(event)

    self._session_events[user_key][session.id] = valid_events
    self._bm25_cache.pop(user_key, None)  # invalidate BM25 cache

  def _build_bm25_for_user(
      self, user_key: str
  ) -> tuple[Optional[BM25Okapi], list[Event]]:
    if user_key not in self._session_events:
      return None, []

    tokenized_corpus: list[list[str]] = []
    events_for_bm25: list[Event] = []

    for session_events_list in self._session_events[user_key].values():
      for event in session_events_list:
        current_event_texts: list[str] = []
        if event.content and event.content.parts:
          for part in event.content.parts:
            if _part_has_text(part):
              current_event_texts.append(part.text.strip())

        if not current_event_texts:
          continue

        full_text_content = " ".join(current_event_texts)
        tokenized_doc = _tokenize(full_text_content)

        if tokenized_doc:
          tokenized_corpus.append(tokenized_doc)
          events_for_bm25.append(event)

    if not tokenized_corpus:
      return None, []

    return BM25Okapi(tokenized_corpus), events_for_bm25

  @override
  async def search_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      query: str,
      max_results: Optional[int] = 3,
  ) -> SearchMemoryResponse:
    user_key = _user_key(app_name, user_id)
    if user_key not in self._session_events:
      return SearchMemoryResponse()

    if user_key not in self._bm25_cache:
      bm25, events = self._build_bm25_for_user(user_key)
      if bm25 is None:
        return SearchMemoryResponse()
      self._bm25_cache[user_key] = (bm25, events)
    else:
      bm25, events = self._bm25_cache[user_key]

    tokenized_query = _tokenize(query)
    if not tokenized_query:
      return SearchMemoryResponse()

    scores = bm25.get_scores(tokenized_query)
    query_word_set = set(tokenized_query)

    scored_events: list[tuple[Event, float]] = []
    for event, score in zip(events, scores):
      # keep any event that shares ≥1 token with the query
      current_event_texts = [
          p.text.strip() for p in event.content.parts if _part_has_text(p)
      ]
      event_word_set = set(_tokenize(" ".join(current_event_texts)))
      if query_word_set & event_word_set:
        scored_events.append((event, score))

    if not scored_events:
      return SearchMemoryResponse()

    scored_events.sort(key=lambda t: t[1], reverse=True)
    scored_events = scored_events[: (max_results or 3)]

    response = SearchMemoryResponse()
    response.memories.extend(
        MemoryEntry(
            content=ev.content,
            author=ev.author,
            timestamp=_utils.format_timestamp(ev.timestamp),
        )
        for ev, _ in scored_events
    )
    return response
