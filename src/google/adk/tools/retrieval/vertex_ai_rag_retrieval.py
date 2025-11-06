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

"""A retrieval tool that uses Vertex AI RAG to retrieve data."""

from __future__ import annotations

import logging
from typing import Any
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ...utils.model_name_utils import is_gemini_2_or_above
from ..tool_context import ToolContext
from .base_retrieval_tool import BaseRetrievalTool

if TYPE_CHECKING:
  from ...dependencies.vertexai import rag
  from ...models import LlmRequest

logger = logging.getLogger('google_adk.' + __name__)


class VertexAiRagRetrieval(BaseRetrievalTool):
  """A retrieval tool that uses Vertex AI RAG (Retrieval-Augmented Generation) to retrieve data."""

  def __init__(
      self,
      *,
      name: str,
      description: str,
      rag_corpora: list[str] = None,
      rag_resources: list[rag.RagResource] = None,
      similarity_top_k: int = None,
      vector_distance_threshold: float = None,
  ):
    super().__init__(name=name, description=description)
    self.vertex_rag_store = types.VertexRagStore(
        rag_corpora=rag_corpora,
        rag_resources=rag_resources,
        similarity_top_k=similarity_top_k,
        vector_distance_threshold=vector_distance_threshold,
    )

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    # Use Gemini built-in Vertex AI RAG tool for Gemini 2 models.
    if is_gemini_2_or_above(llm_request.model):
      llm_request.config = (
          types.GenerateContentConfig()
          if not llm_request.config
          else llm_request.config
      )
      llm_request.config.tools = (
          [] if not llm_request.config.tools else llm_request.config.tools
      )
      llm_request.config.tools.append(
          types.Tool(
              retrieval=types.Retrieval(vertex_rag_store=self.vertex_rag_store)
          )
      )
    else:
      # Add the function declaration to the tools
      await super().process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

  @override
  async def run_async(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:
    from ...dependencies.vertexai import rag

    response = rag.retrieval_query(
        text=args['query'],
        rag_resources=self.vertex_rag_store.rag_resources,
        rag_corpora=self.vertex_rag_store.rag_corpora,
        similarity_top_k=self.vertex_rag_store.similarity_top_k,
        vector_distance_threshold=self.vertex_rag_store.vector_distance_threshold,
    )

    logging.debug('RAG raw response: %s', response)

    return (
        f'No matching result found with the config: {self.vertex_rag_store}'
        if not response.contexts.contexts
        else [context.text for context in response.contexts.contexts]
    )
