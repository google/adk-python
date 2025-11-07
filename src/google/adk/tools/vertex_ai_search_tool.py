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

from __future__ import annotations

from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ..utils.model_name_utils import is_gemini_1_model
from ..utils.model_name_utils import is_gemini_model
from .base_tool import BaseTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models import LlmRequest


class VertexAiSearchTool(BaseTool):
  """A built-in tool using Vertex AI Search.

  Attributes:
    data_store_id: The Vertex AI search data store resource ID.
    search_engine_id: The Vertex AI search engine resource ID.
  """

  def __init__(
      self,
      *,
      data_store_id: Optional[str] = None,
      data_store_specs: Optional[
          list[types.VertexAISearchDataStoreSpec]
      ] = None,
      search_engine_id: Optional[str] = None,
      filter: Optional[str] = None,
      max_results: Optional[int] = None,
      bypass_multi_tools_limit: bool = False,
  ):
    """Initializes the Vertex AI Search tool.

    Args:
      data_store_id: The Vertex AI search data store resource ID in the format
        of
        "projects/{project}/locations/{location}/collections/{collection}/dataStores/{dataStore}".
      data_store_specs: Specifications that define the specific DataStores to be
        searched. It should only be set if engine is used.
      search_engine_id: The Vertex AI search engine resource ID in the format of
        "projects/{project}/locations/{location}/collections/{collection}/engines/{engine}".
      filter: The filter to apply to the search results.
      max_results: The maximum number of results to return.
      bypass_multi_tools_limit: Whether to bypass the multi tools limitation,
        so that the tool can be used with other tools in the same agent.

    Raises:
      ValueError: If both data_store_id and search_engine_id are not specified
      or both are specified.
    """
    # Name and description are not used because this is a model built-in tool.
    super().__init__(name='vertex_ai_search', description='vertex_ai_search')
    if (data_store_id is None and search_engine_id is None) or (
        data_store_id is not None and search_engine_id is not None
    ):
      raise ValueError(
          'Either data_store_id or search_engine_id must be specified.'
      )
    if data_store_specs is not None and search_engine_id is None:
      raise ValueError(
          'search_engine_id must be specified if data_store_specs is specified.'
      )
    self.data_store_id = data_store_id
    self.data_store_specs = data_store_specs
    self.search_engine_id = search_engine_id
    self.filter = filter
    self.max_results = max_results
    self.bypass_multi_tools_limit = bypass_multi_tools_limit

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    if is_gemini_model(llm_request.model):
      if is_gemini_1_model(llm_request.model) and llm_request.config.tools:
        raise ValueError(
            'Vertex AI search tool cannot be used with other tools in Gemini'
            ' 1.x.'
        )
      llm_request.config = llm_request.config or types.GenerateContentConfig()
      llm_request.config.tools = llm_request.config.tools or []
      llm_request.config.tools.append(
          types.Tool(
              retrieval=types.Retrieval(
                  vertex_ai_search=types.VertexAISearch(
                      datastore=self.data_store_id,
                      data_store_specs=self.data_store_specs,
                      engine=self.search_engine_id,
                      filter=self.filter,
                      max_results=self.max_results,
                  )
              )
          )
      )
    else:
      raise ValueError(
          'Vertex AI search tool is not supported for model'
          f' {llm_request.model}'
      )
