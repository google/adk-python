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

from google.genai import types
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
    response = tool_context.search_memory(query)
    if not response.memories:
      return
    memory_text = ''
    for memory in response.memories:
      time_str = datetime.fromtimestamp(memory.events[0].timestamp).isoformat()
      memory_text += f'Time: {time_str}\n'
      for event in memory.events:
        if not event.content or not event.content.parts:
          continue
          
        parts_text = []
        for part in event.content.parts:
          if part.text:
            parts_text.append(part.text)
          elif part.function_call:
            func_name = part.function_call.name or "unknown_function"
            args = str(part.function_call.args or {})
            parts_text.append(f"[Function Call: {func_name}({args})]")
          elif part.function_response:
            func_name = part.function_response.name or "unknown_function"
            response = str(part.function_response.response or {})
            parts_text.append(f"[Function Response: {func_name} -> {response}]")
          elif part.inline_data:
            mime_type = part.inline_data.mime_type or "unknown type"
            parts_text.append(f"[Data: {mime_type}]")
            
        if parts_text:
          memory_text += f'{event.author}: {" ".join(parts_text)}\n'
          
    
    si = f"""The following content is from your previous conversations with the user.
They may be useful for answering the user's current query.
<PAST_CONVERSATIONS>
{memory_text}
</PAST_CONVERSATIONS>
"""
    llm_request.append_instructions([si])


preload_memory_tool = PreloadMemoryTool()
