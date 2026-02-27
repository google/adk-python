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

from typing import Any

from typing_extensions import override

from .base_session_service import BaseSessionService
from .in_memory_session_service import InMemorySessionService
from .session import Session
from .state import State
from .vertex_ai_session_service import VertexAiSessionService

try:
  from .database_session_service import DatabaseSessionService
except ImportError:
  # This handles the case where optional dependencies (like sqlalchemy)
  # are not installed. A placeholder class ensures the symbol is always
  # available for documentation tools and static analysis.
  # We use type: ignore[no-redef, misc] to satisfy strict mypy checks.
  class DatabaseSessionService(BaseSessionService):  # type: ignore[no-redef, misc]
    """Placeholder for DatabaseSessionService when dependencies are not installed."""

    _ERROR_MESSAGE = (
        'DatabaseSessionService requires sqlalchemy>=2.0, please ensure it is'
        ' installed correctly.'
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
      raise ImportError(self._ERROR_MESSAGE)

    @override
    async def create_session(self, *args: Any, **kwargs: Any) -> Any:
      raise ImportError(self._ERROR_MESSAGE)

    @override
    async def get_session(self, *args: Any, **kwargs: Any) -> Any:
      raise ImportError(self._ERROR_MESSAGE)

    @override
    async def list_sessions(self, *args: Any, **kwargs: Any) -> Any:
      raise ImportError(self._ERROR_MESSAGE)

    @override
    async def delete_session(self, *args: Any, **kwargs: Any) -> Any:
      raise ImportError(self._ERROR_MESSAGE)

    @override
    async def append_event(self, *args: Any, **kwargs: Any) -> Any:
      raise ImportError(self._ERROR_MESSAGE)


__all__ = [
    'BaseSessionService',
    'DatabaseSessionService',
    'InMemorySessionService',
    'Session',
    'State',
    'VertexAiSessionService',
]
