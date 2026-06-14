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

from typing_extensions import override
import importlib
from typing import TYPE_CHECKING

from ..utils._dependency import missing_extra
from .base_session_service import BaseSessionService
from .session import Session
from .state import State
from .state import StateSchemaError

if TYPE_CHECKING:
  from .database_session_service import DatabaseSessionService
  from .in_memory_session_service import InMemorySessionService
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
    'StateSchemaError',
    'VertexAiSessionService',
]

_LAZY_MEMBERS: dict[str, str] = {
    'InMemorySessionService': 'in_memory_session_service',
    'VertexAiSessionService': 'vertex_ai_session_service',
}


def __getattr__(name: str):
  if name in _LAZY_MEMBERS:
    module = importlib.import_module(f'{__name__}.{_LAZY_MEMBERS[name]}')
    return vars(module)[name]
  if name == 'DatabaseSessionService':
    try:
      module = importlib.import_module(f'{__name__}.database_session_service')
    except ImportError as e:
      raise missing_extra('sqlalchemy', 'db') from e
    return vars(module)['DatabaseSessionService']
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

