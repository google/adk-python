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
from .base_memory_service import BaseMemoryService

if TYPE_CHECKING:
  from .in_memory_memory_service import InMemoryMemoryService
  from .vertex_ai_memory_bank_service import VertexAiMemoryBankService
  from .vertex_ai_rag_memory_service import VertexAiRagMemoryService

__all__ = [
    'BaseMemoryService',
    'InMemoryMemoryService',
    'VertexAiMemoryBankService',
    'VertexAiRagMemoryService',
]

_LAZY_MEMBERS: dict[str, str] = {
    'InMemoryMemoryService': 'in_memory_memory_service',
    'VertexAiMemoryBankService': 'vertex_ai_memory_bank_service',
    'VertexAiRagMemoryService': 'vertex_ai_rag_memory_service',
}


def __getattr__(name: str):
  if name in _LAZY_MEMBERS:
    module = importlib.import_module(f'{__name__}.{_LAZY_MEMBERS[name]}')
    return vars(module)[name]
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
