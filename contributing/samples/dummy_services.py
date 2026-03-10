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
"""Dummy service implementations for testing."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types
from typing_extensions import override

if TYPE_CHECKING:
  from google.adk.sessions.session import Session


class FooMemoryService(BaseMemoryService):
  """A dummy memory service that returns a fixed response."""

  def __init__(self, uri: str | None = None, **kwargs):
    """Initializes the foo memory service.

    Args:
      uri: The service URI.
      **kwargs: Additional keyword arguments.
    """
    del uri, kwargs  # Unused in this dummy implementation.

  @override
  async def add_session_to_memory(self, session: Session):
    print('FooMemoryService.add_session_to_memory')

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    print('FooMemoryService.search_memory')
    return SearchMemoryResponse(
        memories=[
            MemoryEntry(
                content=types.Content(
                    parts=[types.Part(text='I love ADK from Foo')]
                ),
                author='bot',
                timestamp=datetime.now().isoformat(),
            )
        ]
    )


class BarMemoryService(BaseMemoryService):
  """A dummy memory service that returns a fixed response."""

  def __init__(self, uri: str | None = None, **kwargs):
    """Initializes the bar memory service.

    Args:
      uri: The service URI.
      **kwargs: Additional keyword arguments.
    """
    del uri, kwargs  # Unused in this dummy implementation.

  @override
  async def add_session_to_memory(self, session: Session):
    print('BarMemoryService.add_session_to_memory')

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    print('BarMemoryService.search_memory')
    return SearchMemoryResponse(
        memories=[
            MemoryEntry(
                content=types.Content(
                    parts=[types.Part(text='I love ADK from Bar')]
                ),
                author='bot',
                timestamp=datetime.now().isoformat(),
            )
        ]
    )
