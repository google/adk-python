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
  # fmt: off
  """Search the knowledge base for information relevant to the user's query.

  Use this tool when you need to look up facts, find relevant documents,
  or answer questions that require knowledge from the vector database.
  The tool performs semantic similarity search — it finds content whose
  meaning is closest to the query, not just keyword matches.

  Args:
    query (str): A natural-language search query describing what
      information you are looking for. Be specific and descriptive
      for better results. For example, use "How does Milvus handle
      vector indexing?" rather than just "Milvus".
    filter_expr (str): An optional filter expression to narrow down
      search results before ranking by similarity. Uses Milvus
      boolean expression syntax, for example:
      ``category == "tech"`` or ``year > 2023``. Leave empty if no
      filtering is needed.

  Returns:
    str: A JSON-formatted string containing the search results. Each
      result includes the matched content and a distance score
      indicating similarity (lower distance means higher similarity
      for L2/EUCLIDEAN, higher score means higher similarity for
      COSINE/IP). Returns "No matching results found." if no
      relevant content exists.
  """
  # fmt: on
  results = vector_store.search(query=query, filter_expr=filter_expr)

  if not results:
    return "No matching results found."

  return json.dumps(results, ensure_ascii=False, default=str)
