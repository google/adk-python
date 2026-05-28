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
