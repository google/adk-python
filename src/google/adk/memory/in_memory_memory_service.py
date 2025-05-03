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

from ..events.event import Event
from ..sessions.session import Session
from .base_memory_service import BaseMemoryService
from .base_memory_service import MemoryResult
from .base_memory_service import SearchMemoryResponse


class InMemoryMemoryService(BaseMemoryService):
  """An in-memory memory service for prototyping purpose only.

  Uses keyword matching instead of semantic search.
  """

  def __init__(self):
    self.session_events: dict[str, list[Event]] = {}
    """keys are app_name/user_id/session_id"""

  async def add_session_to_memory(self, session: Session):
    key = f'{session.app_name}/{session.user_id}/{session.id}'
    self.session_events[key] = [
        event for event in session.events if event.content
    ]

  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Prototyping purpose only."""
    keywords = set(query.lower().split())
    response = SearchMemoryResponse()
    for key, events in self.session_events.items():
      if not key.startswith(f'{app_name}/{user_id}/'):
        continue
      matched_events = []
      for event in events:
        if not event.content or not event.content.parts:
          continue
        parts = event.content.parts
        text = '\n'.join([part.text for part in parts if part.text]).lower()
        for keyword in keywords:
          if keyword in text:
            matched_events.append(event)
            break
      if matched_events:
        session_id = key.split('/')[-1]
        response.memories.append(
            MemoryResult(session_id=session_id, events=matched_events)
        )
    return response
