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

"""Provides instructions to an agent by fetching prompts from an MCP server."""

from __future__ import annotations

import logging
import sys
from typing import Any
from typing import Dict
from typing import TextIO

from .llm_agent import InstructionProvider
from .readonly_context import ReadonlyContext

# Attempt to import MCP Session Manager from the MCP library, and hints user to
# upgrade their Python version to 3.10 if it fails.
try:
  from mcp import types

  from ..tools.mcp_tool.mcp_session_manager import MCPSessionManager
except ImportError as e:
  if sys.version_info < (3, 10):
    raise ImportError(
        "MCP Session Manager requires Python 3.10 or above. Please upgrade"
        " your Python version."
    ) from e
  else:
    raise e


class McpInstructionProvider(InstructionProvider):
  """Fetches agent instructions from an MCP server."""

  def __init__(
      self,
      connection_params: Any,
      prompt_name: str,
      errlog: TextIO = sys.stderr,
  ):
    """Initializes the McpInstructionProvider.

    Args:
        connection_params: Parameters for connecting to the MCP server.
        prompt_name: The name of the MCP Prompt to fetch.
        errlog: TextIO stream for error logging.
    """
    self._connection_params = connection_params
    self._errlog = errlog or logging.getLogger(__name__)
    self._mcp_session_manager = MCPSessionManager(
        connection_params=self._connection_params,
        errlog=self._errlog,
    )
    self.prompt_name = prompt_name

  async def __call__(self, context: ReadonlyContext) -> str:
    """Fetches the instruction from the MCP server.

    Args:
        context: The read-only context of the agent.

    Returns:
        The instruction string.
    """
    session = await self._mcp_session_manager.create_session()
    # Fetch prompt definition to get the required argument names
    prompt_definitions = await session.list_prompts()
    prompt_definition = next(
        (p for p in prompt_definitions.prompts if p.name == self.prompt_name),
        None,
    )

    # Fetch arguments from context state if the prompt requires them
    prompt_args: Dict[str, Any] = {}
    if prompt_definition and prompt_definition.arguments:
      arg_names = {arg.name for arg in prompt_definition.arguments}
      prompt_args = {
          k: v for k, v in (context.state or {}).items() if k in arg_names
      }

    # Fetch the specific prompt by name with arguments from context state
    prompt_result: types.GetPromptResult = await session.get_prompt(
        self.prompt_name, arguments=prompt_args
    )

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
