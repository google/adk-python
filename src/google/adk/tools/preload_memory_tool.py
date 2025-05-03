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

from datetime import datetime
from typing import TYPE_CHECKING

from typing_extensions import override

from .base_tool import BaseTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models import LlmRequest


class PreloadMemoryTool(BaseTool):
  """A tool that preloads the memory for the current user."""

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
    parts = tool_context.user_content.parts
    if not parts or not parts[0].text:
      return
    query = parts[0].text
    response = await tool_context.search_memory(query)
    if not response.memories:
      return
    memory_text = ''
    for memory in response.memories:
      time_str = datetime.fromtimestamp(memory.events[0].timestamp).isoformat()
      memory_text += f'Time: {time_str}\n'
      for event in memory.events:
        # TODO: support multi-part content.
        if (
            event.content
            and event.content.parts
            and event.content.parts[0].text
        ):
          memory_text += f'{event.author}: {event.content.parts[0].text}\n'
    si = f"""The following content is from your previous conversations with the user.
They may be useful for answering the user's current query.
<PAST_CONVERSATIONS>
{memory_text}
</PAST_CONVERSATIONS>
"""
    llm_request.append_instructions([si])


preload_memory_tool = PreloadMemoryTool()
