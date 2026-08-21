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
from collections.abc import Sequence
import re
import threading
from typing import TYPE_CHECKING

from typing_extensions import override

from . import _utils
from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..events.event import Event
  from ..sessions.session import Session

_UNKNOWN_SESSION_ID = '__unknown_session_id__'
_MAX_SEARCH_RESULTS = 10


def _user_key(app_name: str, user_id: str) -> tuple[str, str]:
  return (app_name, user_id)


def _extract_words_lower(text: str) -> set[str]:
  """Extracts Unicode-aware tokens from a string in lowercase."""
  return set(word.lower() for word in re.findall(r'\w+', text))


class InMemoryMemoryService(BaseMemoryService):
  """An in-memory memory service for prototyping purpose only.

  Uses keyword matching instead of semantic search. A search returns at most
  ten memories, the ones sharing the most words with the query.

  This class is thread-safe, however, it should be used for testing and
  development only.
  """

  def __init__(self) -> None:
    self._lock = threading.Lock()

    self._session_events: dict[tuple[str, str], dict[str, list[Event]]] = {}
    """Keys are (app_name, user_id). Values are dicts of session_id to
    session event lists.
    """

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    user_key = _user_key(session.app_name, session.user_id)

    with self._lock:
      self._session_events[user_key] = self._session_events.get(user_key, {})
      self._session_events[user_key][session.id] = [
          event
          for event in session.events
          if event.content and event.content.parts
      ]

  @override
  async def add_events_to_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      events: Sequence[Event],
      session_id: str | None = None,
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    _ = custom_metadata
    user_key = _user_key(app_name, user_id)
    scoped_session_id = session_id or _UNKNOWN_SESSION_ID
    events_to_add = [
        event for event in events if event.content and event.content.parts
    ]

    with self._lock:
      self._session_events[user_key] = self._session_events.get(user_key, {})
      existing_events = self._session_events[user_key].get(
          scoped_session_id, []
      )
      existing_ids = {event.id for event in existing_events}
      for event in events_to_add:
        if event.id not in existing_ids:
          existing_events.append(event)
          existing_ids.add(event.id)
      self._session_events[user_key][scoped_session_id] = existing_events

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    user_key = _user_key(app_name, user_id)

    with self._lock:
      # Copy the events into a stable snapshot while holding the lock. Iterating
      # a live reference outside the lock would race with concurrent writers
      # (add_session_to_memory / add_events_to_memory) mutating the same dict
      # and lists, raising "dictionary changed size during iteration".
      session_event_lists = [
          list(events)
          for events in self._session_events.get(user_key, {}).values()
      ]

    words_in_query = _extract_words_lower(query)
    scored_memories: list[tuple[int, MemoryEntry]] = []

    for session_events in session_event_lists:
      for event in session_events:
        if not event.content or not event.content.parts:
          continue
        event_text = ' '.join(
            [part.text for part in event.content.parts if part.text]
        )
        words_in_event = _extract_words_lower(event_text)
        if not words_in_event:
          continue

        event_text_lower = event_text.lower()
        matched_words = sum(
            1
            for query_word in words_in_query
            if query_word in words_in_event
            or (not query_word.isascii() and query_word in event_text_lower)
        )
        if matched_words:
          scored_memories.append((
              matched_words,
              MemoryEntry(
                  content=event.content,
                  author=event.author,
                  timestamp=_utils.format_timestamp(event.timestamp),
              ),
          ))

    # Almost any two sentences share a word, so returning every event that
    # matches at least one query word returns most of the store, and callers
    # such as the preload_memory tool put all of it in the prompt. Keep the
    # events matching the most query words. The sort key reads only the count,
    # so it is stable and events matching equally stay in insertion order.
    scored_memories.sort(key=lambda scored_memory: -scored_memory[0])
    return SearchMemoryResponse(
        memories=[memory for _, memory in scored_memories[:_MAX_SEARCH_RESULTS]]
    )
