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
from typing import TYPE_CHECKING, List, Any

from typing_extensions import override

from .base_tool import BaseTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models import LlmRequest


class PreloadMemoryTool(BaseTool):
  """A tool that preloads the memory from past conversations for the current user."""

  def __init__(self):
    # Name and description are primarily for identifying the tool if needed,
    # but this tool acts implicitly by modifying the request.
    super().__init__(name='preload_memory', description='Preloads relevant past conversation memory into the LLM request.')

  def _get_text_content(self, part: Any) -> str:
    """Extract text content from a part."""
    if hasattr(part, 'text'):
      if isinstance(part.text, str):
        return part.text.strip()
    return ""

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    """
    Searches conversation memory based on the current user query and prepends
    relevant snippets to the LLM request instructions.
    """
    # Extract the primary query text from the user's latest message
    parts = tool_context.user_content.parts
    if not parts:
      # If there's no parts, we can't search memory effectively
      return
    
    if not hasattr(parts[0], 'text') or not parts[0].text:
      # If there's no text query part, we can't search memory effectively
      return
    
    query = parts[0].text.strip()
    if not query:
      # Handle cases where the text part might be empty or just whitespace
      return

    # Search the memory store using the extracted query
    response = tool_context.search_memory(query)

    # If no relevant memories were found, exit
    if not response or not response.memories:
      return

    # Format the found memories into a text block
    memory_text = ''
    for memory in response.memories:
      # Ensure the memory entry actually contains events
      if not hasattr(memory, 'events') or not memory.events:
        continue
        
      # Add a timestamp from the first event in the memory snippet
      try:
        time_str = datetime.fromtimestamp(memory.events[0].timestamp).isoformat()
        memory_text += f'Time: {time_str}\n'
      except (AttributeError, IndexError, TypeError):
        # Handle cases where timestamp might be missing or invalid
        memory_text += 'Time: [Unknown]\n'
      
      # Process each event in the memory
      for event in memory.events:
        # Skip events without content
        if not hasattr(event, 'content') or not event.content:
          continue
          
        # Skip events without parts
        if not hasattr(event.content, 'parts') or not event.content.parts:
          continue
        
        # Collect all text parts from this event
        text_parts = []
        for part in event.content.parts:
          text = self._get_text_content(part)
          if text:
            text_parts.append(text)
        
        # Skip if no text parts were found
        if not text_parts:
          continue
        
        # Join the text parts with spaces and add them to memory text
        joined_text = " ".join(text_parts)
        memory_text += f"{event.author}: {joined_text}\n"

    # If memory_text remains empty after processing, don't add instructions
    if not memory_text:
      return

    # Construct the system instruction prepending the memory context
    si = f"""The following content is from your previous conversations with the user.
They may be useful for answering the user's current query.
<PAST_CONVERSATIONS>
{memory_text.strip()}
</PAST_CONVERSATIONS>
"""

    # Append the constructed instruction to the LLM request
    llm_request.append_instructions([si])


# Instantiate the tool for use in the application.
preload_memory_tool = PreloadMemoryTool()
