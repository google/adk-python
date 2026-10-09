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

"""A retrieval tool that uses Vertex AI RAG to retrieve data."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import logging
import os
from typing import Any
from typing import Protocol

from google.genai import types
from typing_extensions import override

from ...models.llm_request import LlmRequest
from ...utils.model_name_utils import is_gemini_model
from ...utils.model_name_utils import is_gemini_model_id_check_disabled
from ..tool_context import ToolContext
from .base_retrieval_tool import BaseRetrievalTool

logger = logging.getLogger("google_adk." + __name__)


class RagResourceLike(Protocol):
  """Anything shaped like a RAG resource.

  Structural rather than nominal so a `vertexai.rag.RagResource` from
  google-cloud-aiplatform keeps working without ADK depending on that package.
  """

  rag_corpus: str | None
  rag_file_ids: list[str] | None


RagResourceOrDict = (
    types.VertexRagStoreRagResource
    | types.VertexRagStoreRagResourceDict
    | RagResourceLike
)


def _to_rag_resource(
    resource: RagResourceOrDict,
) -> types.VertexRagStoreRagResource:
  """Coerces a RAG-resource-shaped value into the google-genai type."""
  if isinstance(resource, types.VertexRagStoreRagResource):
    return resource
  if isinstance(resource, Mapping):
    return types.VertexRagStoreRagResource(**resource)
  if not hasattr(resource, "rag_corpus") and not hasattr(
      resource, "rag_file_ids"
  ):
    raise TypeError(
        "rag_resources entries must be a types.VertexRagStoreRagResource, a"
        " mapping, or an object carrying rag_corpus / rag_file_ids (such as"
        f" vertexai.rag.RagResource); got {type(resource).__name__}."
    )
  return types.VertexRagStoreRagResource(
      rag_corpus=getattr(resource, "rag_corpus", None),
      rag_file_ids=getattr(resource, "rag_file_ids", None),
  )


class VertexAiRagRetrieval(BaseRetrievalTool):
  """A retrieval tool that uses Vertex AI RAG (Retrieval-Augmented Generation) to retrieve data."""

  def __init__(
      self,
      *,
      name: str,
      description: str,
      rag_corpora: list[str] | None = None,
      rag_resources: list[RagResourceOrDict] | None = None,
      similarity_top_k: int | None = None,
      vector_distance_threshold: float | None = None,
  ):
    super().__init__(name=name, description=description)
    self.vertex_rag_store = types.VertexRagStore(
        rag_corpora=rag_corpora,
        rag_resources=(
            [_to_rag_resource(resource) for resource in rag_resources]
            if rag_resources
            else None
        ),
        similarity_top_k=similarity_top_k,
        vector_distance_threshold=vector_distance_threshold,
    )

  def _retrieval_store(self) -> types.VertexRagStore:
    """Returns the store trimmed to what RetrieveContexts accepts.

    RetrieveContextsRequest.VertexRagStore carries only rag_corpora,
    rag_resources and vector_distance_threshold. similarity_top_k moved to
    RagQuery and its field number is reserved, so leaving it on the store makes
    the backend reject the request.
    """
    return types.VertexRagStore(
        rag_corpora=self.vertex_rag_store.rag_corpora,
        rag_resources=self.vertex_rag_store.rag_resources,
        vector_distance_threshold=(
            self.vertex_rag_store.vector_distance_threshold
        ),
    )

  def _project_and_location(self) -> tuple[str | None, str | None]:
    """Resolves the project and location, preferring the corpus name.

    A fully-qualified corpus name names exactly one endpoint, so it wins; the
    environment only fills in what a bare corpus ID leaves unspecified.
    """
    store_project = None
    store_location = None
    corpus_name = None
    if self.vertex_rag_store.rag_corpora:
      corpus_name = self.vertex_rag_store.rag_corpora[0]
    elif self.vertex_rag_store.rag_resources:
      corpus_name = self.vertex_rag_store.rag_resources[0].rag_corpus

    if corpus_name and corpus_name.startswith("projects/"):
      parts = corpus_name.split("/")
      if len(parts) >= 4 and parts[0] == "projects" and parts[2] == "locations":
        store_project = parts[1]
        store_location = parts[3]

    return (
        store_project or os.environ.get("GOOGLE_CLOUD_PROJECT"),
        store_location or os.environ.get("GOOGLE_CLOUD_LOCATION"),
    )

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    # Use Gemini built-in Vertex AI RAG tool for Gemini models.
    model_check_disabled = is_gemini_model_id_check_disabled()
    if is_gemini_model(llm_request.model) or model_check_disabled:
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
    try:
      from ...dependencies._agentplatform import agentplatform
    except ImportError as e:
      from ...utils._dependency import missing_extra

      raise missing_extra("google-cloud-agentplatform", "gcp") from e

    agentplatform_types = agentplatform.types

    query = args.get("query")
    if not isinstance(query, str):
      raise ValueError("Vertex AI RAG retrieval requires a string 'query'.")

    project, location = self._project_and_location()

    client = agentplatform.Client(project=project, location=location).aio
    try:
      response = await client.rag.retrieve_contexts(
          vertex_rag_store=self._retrieval_store(),
          query=agentplatform_types.RagQuery(
              text=query,
              similarity_top_k=self.vertex_rag_store.similarity_top_k,
          ),
      )
    finally:
      # Shielded so a cancellation racing the close cannot leak the underlying
      # HTTP session.
      await asyncio.shield(client.aclose())

    logger.debug("RAG raw response: %s", response)

    contexts = response.contexts.contexts if response.contexts else None
    return (
        f"No matching result found with the config: {self.vertex_rag_store}"
        if not contexts
        else [context.text for context in contexts if context.text is not None]
    )
