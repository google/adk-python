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

from typing import TYPE_CHECKING

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

from ..memory.memory_entry import MemoryEntry
from .function_tool import FunctionTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models import LlmRequest


class LoadMemoryResponse(BaseModel):
  memories: list[MemoryEntry] = Field(default_factory=list)


async def load_memory(
    query: str, tool_context: ToolContext
) -> LoadMemoryResponse:
  """Loads the memory for the current user.

  Args:
    query: The query to load the memory for.

  Returns:
    A list of memory results.
  """
  search_memory_response = await tool_context.search_memory(query)
  return LoadMemoryResponse(memories=search_memory_response.memories)


class LoadMemoryTool(FunctionTool):
  """A tool that loads the memory for the current user.

  NOTE: Currently this tool only uses text part from the memory.
  """

  def __init__(self):
    super().__init__(load_memory)

  @override
  def _get_declaration(self) -> types.FunctionDeclaration | None:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                'query': types.Schema(
                    type=types.Type.STRING,
                )
            },
            required=['query'],
        ),
    )

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    await super().process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )
    # Tell the model about the memory.
    llm_request.append_instructions(["""
You have memory. You can use it to answer questions. If any questions need
you to look up the memory, you should call load_memory function with a query.
"""])


load_memory_tool = LoadMemoryTool()
