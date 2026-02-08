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

"""Milvus similarity search tool function for agent use."""

from __future__ import annotations

import json
import logging
from typing import Optional

from .milvus_vector_store import MilvusVectorStore

logger = logging.getLogger("google_adk." + __name__)


def similarity_search(
    query: str,
    vector_store: MilvusVectorStore,
    filter_expr: Optional[str] = None,
) -> str:
  """Search for similar content in Milvus vector store.

  This function is intended to be wrapped by MilvusToolset and exposed
  to agents as a tool.

  Args:
    query: The search query text.
    vector_store: The MilvusVectorStore instance (injected by MilvusToolset).
    filter_expr: Optional Milvus filter expression for pre-filtering
      results.

  Returns:
    Search results formatted as a JSON string.
  """
  results = vector_store.search(query=query, filter_expr=filter_expr)

  if not results:
    return "No matching results found."

  return json.dumps(results, ensure_ascii=False, default=str)
