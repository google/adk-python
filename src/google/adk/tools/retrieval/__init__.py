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

from .base_retrieval_tool import BaseRetrievalTool
from .files_retrieval import FilesRetrieval
from .llama_index_retrieval import LlamaIndexRetrieval

__all__ = [
    'BaseRetrievalTool',
    'FilesRetrieval',
    'LlamaIndexRetrieval',
    'VertexAiRagRetrieval',
]


def __getattr__(name: str):
  if name == 'VertexAiRagRetrieval':
    from .vertex_ai_rag_retrieval import VertexAiRagRetrieval

    return VertexAiRagRetrieval
  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
