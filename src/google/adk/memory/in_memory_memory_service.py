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
from typing import TYPE_CHECKING

from typing_extensions import override

from . import _utils
from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..events.event import Event
  from ..sessions.session import Session


def _user_key(app_name: str, user_id: str):
  return f'{app_name}/{user_id}'


def _extract_words_lower(text: str) -> set[str]:
  """Extracts words from a string and converts them to lowercase."""
  return set([word.lower() for word in re.findall(r'[A-Za-z]+', text)])


class InMemoryMemoryService(BaseMemoryService):
  """An in-memory memory service for prototyping purpose only.

  Uses keyword matching instead of semantic search.
  """

  def __init__(self):
    self._session_events: dict[str, dict[str, list[Event]]] = {}
    """Keys are app_name/user_id, session_id. Values are session event lists."""

  @override
  async def add_session_to_memory(self, session: Session):
    user_key = _user_key(session.app_name, session.user_id)
    self._session_events[user_key] = self._session_events.get(
        _user_key(session.app_name, session.user_id), {}
    )
    self._session_events[user_key][session.id] = [
        event
        for event in session.events
        if event.content and event.content.parts
    ]

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    user_key = _user_key(app_name, user_id)
    if user_key not in self._session_events:
      return SearchMemoryResponse()

    words_in_query = set(query.lower().split())
    response = SearchMemoryResponse()

    for session_events in self._session_events[user_key].values():
      for event in session_events:
        if not event.content or not event.content.parts:
          continue
        words_in_event = _extract_words_lower(
            ' '.join([part.text for part in event.content.parts if part.text])
        )
        if not words_in_event:
          continue

        if any(query_word in words_in_event for query_word in words_in_query):
          response.memories.append(
              MemoryEntry(
                  content=event.content,
                  author=event.author,
                  timestamp=_utils.format_timestamp(event.timestamp),
              )
          )

    return response
