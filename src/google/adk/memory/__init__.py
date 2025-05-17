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
import logging

from .base_memory_service import BaseMemoryService
from .in_memory_memory_service import InMemoryMemoryService

logger = logging.getLogger('google_adk.' + __name__)

__all__ = [
    'BaseMemoryService',
    'InMemoryMemoryService',
]

try:
  from .vertex_ai_rag_memory_service import VertexAiRagMemoryService

  __all__.append('VertexAiRagMemoryService')
except ImportError:
  logger.debug(
      'The Vertex sdk is not installed. If you want to use the'
      ' VertexAiRagMemoryService please install it. If not, you can ignore this'
      ' warning.'
  )
