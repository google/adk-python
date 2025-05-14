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

from typing_extensions import override

from . import _memory_entry_utils
from .base_tool import BaseTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models import LlmRequest


class PreloadMemoryTool(BaseTool):
  """A tool that preloads the memory for the current user.

  NOTE: Currently this tool only uses text part from the memory.
  """

  def __init__(self):
    # Name and description are not used because this tool only
    # changes llm_request.
    super().__init__(name='preload_memory', description='preload_memory')

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    user_content = tool_context.user_content
    if (
        not user_content
        or not user_content.parts
        or not user_content.parts[0].text
    ):
      return

    user_query: str = user_content.parts[0].text
    response = await tool_context.search_memory(user_query)
    if not response.memories:
      return

    memory_text_lines = []
    for memory in response.memories:
      if time_str := (f'Time: {memory.timestamp}' if memory.timestamp else ''):
        memory_text_lines.append(time_str)
      if memory_text := _memory_entry_utils.extract_text(memory):
        memory_text_lines.append(
            f'{memory.author}: {memory_text}' if memory.author else memory_text
        )
    if not memory_text_lines:
      return

    full_memory_text = '\n'.join(memory_text_lines)
    si = f"""The following content is from your previous conversations with the user.
They may be useful for answering the user's current query.
<PAST_CONVERSATIONS>
{full_memory_text}
</PAST_CONVERSATIONS>
"""
    llm_request.append_instructions([si])


preload_memory_tool = PreloadMemoryTool()
