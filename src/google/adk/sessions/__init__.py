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
  # available for documentation tools, providing a clear error message
  # at runtime if used without the necessary dependencies.
  class DatabaseSessionService:

    def __init__(self, *args, **kwargs):
      raise ImportError(
          'DatabaseSessionService requires sqlalchemy>=2.0, please ensure it is'
          ' installed correctly.'
      )


__all__ = [
    'BaseSessionService',
    'DatabaseSessionService',
    'InMemorySessionService',
    'Session',
    'State',
    'VertexAiSessionService',
]
