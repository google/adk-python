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

"""Milvus Toolset for exposing Milvus vector search to agents."""

from __future__ import annotations

from typing import Callable
from typing import List
from typing import Optional
from typing import Union

from typing_extensions import override

from . import search_tool
from ...agents.readonly_context import ReadonlyContext
from ...features import experimental
from ...features import FeatureName
from ...tools.base_tool import BaseTool
from ...tools.base_toolset import BaseToolset
from ...tools.base_toolset import ToolPredicate
from .milvus_tool import MilvusTool
from .milvus_vector_store import MilvusVectorStore
from .settings import MilvusToolSettings

DEFAULT_MILVUS_TOOL_NAME_PREFIX = "milvus"


@experimental(FeatureName.MILVUS_TOOLSET)
class MilvusToolset(BaseToolset):
  """Milvus Toolset provides tools for vector similarity search in Milvus.

  The tool names are:
    - milvus_similarity_search
  """

  def __init__(
      self,
      *,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
      milvus_tool_settings: Optional[MilvusToolSettings] = None,
      embedding_fn: Callable[[list[str]], list[list[float]]],
  ):
    """Initialize the Milvus Toolset.

    Args:
      tool_filter: Optional filter to select a subset of tools.
      milvus_tool_settings: Milvus tool settings containing vector store
        configuration.
      embedding_fn: A function that takes a list of texts and returns a
        list of embedding vectors.
    """
    super().__init__(
        tool_filter=tool_filter,
        tool_name_prefix=DEFAULT_MILVUS_TOOL_NAME_PREFIX,
    )
    self._tool_settings = milvus_tool_settings or MilvusToolSettings()
    self._embedding_fn = embedding_fn
    self._vector_store: Optional[MilvusVectorStore] = None

  def _get_vector_store(self) -> MilvusVectorStore:
    """Lazily creates the MilvusVectorStore instance."""
    if self._vector_store is None:
      self._vector_store = MilvusVectorStore(
          settings=self._tool_settings,
          embedding_fn=self._embedding_fn,
      )
    return self._vector_store

  @override
  async def get_tools(
      self, readonly_context: Optional[ReadonlyContext] = None
  ) -> List[BaseTool]:
    """Get tools from the toolset."""
    all_tools: list[BaseTool] = []

    if self._tool_settings.vector_store_settings:
      vector_store = self._get_vector_store()
      all_tools.append(
          MilvusTool(
              func=search_tool.similarity_search,
              vector_store=vector_store,
          )
      )

    return [
        tool
        for tool in all_tools
        if self._is_tool_selected(tool, readonly_context)
    ]

  @override
  async def close(self) -> None:
    """Closes the Milvus client connection."""
    if self._vector_store is not None:
      self._vector_store.close()
      self._vector_store = None
