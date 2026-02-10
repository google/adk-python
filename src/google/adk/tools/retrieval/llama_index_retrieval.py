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

"""Provides data for the agent."""

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from typing_extensions import override

from ..tool_context import ToolContext
from .base_retrieval_tool import BaseRetrievalTool

if TYPE_CHECKING:
  from llama_index.core.base.base_retriever import BaseRetriever


class LlamaIndexRetrieval(BaseRetrievalTool):

  def __init__(self, *, name: str, description: str, retriever: BaseRetriever):
    super().__init__(name=name, description=description)
    self.retriever = retriever

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    return self.retriever.retrieve(args['query'])[0].text
