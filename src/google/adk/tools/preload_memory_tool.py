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

from __future__ import annotations

import logging
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from . import _memory_entry_utils
from ..utils import content_utils
from .base_tool import BaseTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models.llm_request import LlmRequest

logger = logging.getLogger('google_adk.' + __name__)


class PreloadMemoryTool(BaseTool):
  """A tool that preloads the memory for the current user.

  This tool will be automatically executed for each llm_request, and it won't be
  called by the model.

  NOTE: Currently this tool only uses text part from the memory.

  By default a failing memory search is logged and the request proceeds without
  memories (fail-open). Use `on_error` to observe such failures, or
  `raise_on_error` to fail closed.
  """

  def __init__(
      self,
      *,
      raise_on_error: bool = False,
      on_error: Optional[Callable[[Exception, ToolContext], None]] = None,
  ) -> None:
    """Initializes the PreloadMemoryTool.

    Args:
      raise_on_error: Whether to re-raise exceptions from the memory search
        instead of continuing without memories.
      on_error: Optional callback invoked with the exception and the tool
        context when the memory search raises. Exceptions raised by the
        callback itself are logged and do not affect the request.
    """
    # Name and description are not used because this tool only
    # changes llm_request.
    super().__init__(name='preload_memory', description='preload_memory')
    self._raise_on_error = raise_on_error
    self._on_error = on_error

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    user_query = content_utils.extract_text_from_content(
        tool_context.user_content, splitter=' '
    )
    if not user_query:
      return

    try:
      response = await tool_context.search_memory(user_query)
    except Exception as e:
      logger.warning(
          'Failed to preload memory (query length: %d)',
          len(user_query),
          exc_info=True,
      )
      if self._on_error is not None:
        try:
          self._on_error(e, tool_context)
        except Exception:
          logger.warning(
              'PreloadMemoryTool on_error callback failed', exc_info=True
          )
      if self._raise_on_error:
        raise
      return

    if not response.memories:
      return

    memory_text_lines = []
    for memory in response.memories:
      if memory_text := _memory_entry_utils.extract_text(memory):
        if memory.timestamp:
          memory_text_lines.append(f'Time: {memory.timestamp}')
        memory_text_lines.append(
            f'{memory.author}: {memory_text}' if memory.author else memory_text
        )
    if not memory_text_lines:
      return

    full_memory_text = '\n'.join(memory_text_lines)
    memory_context = f"""The following content is from your previous conversations with the user.
They may be useful for answering the user's current query.
<PAST_CONVERSATIONS>
{full_memory_text}
</PAST_CONVERSATIONS>
"""
    llm_request._insert_transient_user_content([  # pylint: disable=protected-access
        types.Content(
            role='user', parts=[types.Part.from_text(text=memory_context)]
        )
    ])


preload_memory_tool = PreloadMemoryTool()
