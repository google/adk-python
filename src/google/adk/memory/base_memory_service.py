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

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import Field

from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..sessions.session import Session


class SearchMemoryResponse(BaseModel):
  """Represents the response from a memory search.

  Attributes:
      memories: A list of memory entries that relate to the search query.
  """

  memories: list[MemoryEntry] = Field(default_factory=list)


class BaseMemoryService(ABC):
  """Base class for memory services.

  The service provides functionalities to ingest sessions into memory so that
  the memory can be used for user queries.
  """

  @abstractmethod
  async def add_session_to_memory(
      self,
      session: Session,
  ):
    """Adds a session to the memory service.

    A session may be added multiple times during its lifetime.

    Args:
        session: The session to add.
    """

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
