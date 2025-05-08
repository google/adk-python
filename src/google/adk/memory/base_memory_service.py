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

import abc

from pydantic import BaseModel
from pydantic import Field

from ..events.event import Event
from ..sessions.session import Session


class MemoryResult(BaseModel):
  """Represents a single memory retrieval result.

  Attributes:
      session_id: The session id associated with the memory.
      events: A list of events in the session.
  """

  session_id: str
  events: list[Event]


class SearchMemoryResponse(BaseModel):
  """Represents the response from a memory search.

  Attributes:
      memories: A list of memory results matching the search query.
  """

  memories: list[MemoryResult] = Field(default_factory=list)


class BaseMemoryService(abc.ABC):
  """Base class for memory services.

  The service provides functionalities to ingest sessions into memory so that
  the memory can be used for user queries.
  """

  @abc.abstractmethod
  async def add_session_to_memory(self, session: Session):
    """Adds a session to the memory service.

    A session may be added multiple times during its lifetime.

    Args:
        session: The session to add.
    """

  @abc.abstractmethod
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Searches for sessions that match the query.

    Args:
        app_name: The name of the application.
        user_id: The id of the user.
        query: The query to search for.

    Returns:
        A SearchMemoryResponse containing the matching memories.
    """
