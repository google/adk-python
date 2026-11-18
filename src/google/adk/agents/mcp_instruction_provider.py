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

"""Provides instructions to an agent by fetching prompts from an MCP server."""

from __future__ import annotations

import sys
from typing import Any
from typing import Callable
from typing import Mapping
from typing import TextIO

from ..dependencies._mcp import types
from ..tools.mcp_tool.mcp_session_manager import MCPSessionManager
from .llm_agent import InstructionProvider
from .readonly_context import ReadonlyContext


class McpInstructionProvider(InstructionProvider):  # type: ignore[misc]
  """Fetches agent instructions from an MCP server."""

  def __init__(
      self,
      connection_params: Any,
      prompt_name: str,
      errlog: TextIO = sys.stderr,
      *,
      arguments: (
          Mapping[str, str]
          | Callable[[ReadonlyContext], Mapping[str, str]]
          | None
      ) = None,
  ):
    """Initializes the McpInstructionProvider.

    Args:
        connection_params: Parameters for connecting to the MCP server.
        prompt_name: The name of the MCP Prompt to fetch.
        errlog: TextIO stream for error logging.
        arguments: The arguments to send with the prompt request, either a fixed
          mapping or a callable that builds one from the context. No arguments
          are sent when None.
    """
    self._connection_params = connection_params
    self._errlog = errlog
    self._mcp_session_manager = MCPSessionManager(
        connection_params=self._connection_params,
        errlog=self._errlog,
    )
    self.prompt_name = prompt_name
    self._arguments = arguments

  async def __call__(self, context: ReadonlyContext) -> str:
    """Fetches the instruction from the MCP server.

    Args:
        context: The read-only context of the agent.

    Returns:
        The instruction string.
    """
    # The caller decides what is sent; the server's declared argument names are
    # not consulted, so it cannot pick which session values it receives.
    arguments = (
        self._arguments(context)
        if callable(self._arguments)
        else self._arguments
    )
    arguments = None if arguments is None else dict(arguments)

    session = await self._mcp_session_manager.create_session()
    # Hold the session out of the pool's idle sweep while it is in use, so
    # another caller's sweep cannot close the transport mid-call.
    self._mcp_session_manager._begin_session_use()  # pylint: disable=protected-access
    try:
      prompt_result: types.GetPromptResult = await session.get_prompt(
          self.prompt_name, arguments=arguments
      )
    finally:
      self._mcp_session_manager._end_session_use()  # pylint: disable=protected-access

    if prompt_result and prompt_result.messages:
      # Concatenate content of all messages to form the instruction.
      instruction = "".join(
          message.content.text
          for message in prompt_result.messages
          if message.content.type == "text"
      )
      return instruction
    else:
      raise ValueError(f"Failed to load MCP prompt '{self.prompt_name}'.")
