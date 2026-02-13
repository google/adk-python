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

from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import Field

from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..events.event import Event
  from ..sessions.session import Session


class SearchMemoryResponse(BaseModel):
  """Represents the response from a memory search.

  Attributes:
      memories: A list of memory entries that relate to the search query.
  """

  memories: list[MemoryEntry] = Field(default_factory=list)


class BaseMemoryService(ABC):
  """Base class for memory services.

  The service provides functionality to ingest conversation history into memory
  so that it can be used for user queries.
  """

  @abstractmethod
  async def add_session_to_memory(
      self,
      session: Session,
  ) -> None:
    """Adds a session to the memory service.

    A session may be added multiple times during its lifetime.

    Args:
        session: The session to add.
    """

  async def add_events_to_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      events: Sequence[Event],
      session_id: str | None = None,
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    """Adds an explicit list of events to the memory service.

    This is intended for cases where callers want to persist only a subset of
    events (e.g., the latest turn), rather than re-ingesting the full session.

    Implementations should treat `events` as an incremental update (delta) and
    must not assume it represents the full session.
    Implementations may ignore `session_id` if it is not applicable.

    Args:
      app_name: The application name for memory scope.
      user_id: The user ID for memory scope.
      events: The events to add to memory.
      session_id: Optional session ID for memory scope/partitioning.
      custom_metadata: Optional, portable metadata for memory generation. Prefer
        this for service-specific fields (e.g., TTL) that may later become
        first-class API parameters.
    """
    raise NotImplementedError(
        "This memory service does not support adding event deltas. "
        "Call add_session_to_memory(session) to ingest the full session."
    )

  @abstractmethod
  async def search_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      query: str,
  ) -> SearchMemoryResponse:
    """Searches for sessions that match the query.

    Args:
        app_name: The name of the application.
        user_id: The id of the user.
        query: The query to search for.

    Returns:
        A SearchMemoryResponse containing the matching memories.
    """
