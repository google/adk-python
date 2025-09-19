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

import logging
from typing import Optional
from typing import Union

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..agents.context_cache_config import ContextCacheConfig
from ..tools.base_tool import BaseTool
from .cache_metadata import CacheMetadata


def _find_tool_with_function_declarations(
    llm_request: LlmRequest,
) -> Optional[types.Tool]:
  """Find an existing Tool with function_declarations in the LlmRequest."""
  # TODO: add individual tool with declaration and merge in google_llm.py
  if not llm_request.config or not llm_request.config.tools:
    return None

  return next(
      (
          tool
          for tool in llm_request.config.tools
          if isinstance(tool, types.Tool) and tool.function_declarations
      ),
      None,
  )


class LlmRequest(BaseModel):
  """LLM request class that allows passing in tools, output schema and system

  instructions to the model.

  Attributes:
    model: The model name.
    contents: The contents to send to the model.
    config: Additional config for the generate content request.
    tools_dict: The tools dictionary.
    cache_config: Context cache configuration for this request.
    cache_metadata: Cache metadata from previous requests, used for cache management.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)
  """The pydantic model config."""

  model: Optional[str] = None
  """The model name."""

  contents: list[types.Content] = Field(default_factory=list)
  """The contents to send to the model."""

  config: types.GenerateContentConfig = Field(
      default_factory=types.GenerateContentConfig
  )
  live_connect_config: types.LiveConnectConfig = Field(
      default_factory=types.LiveConnectConfig
  )
  """Additional config for the generate content request.

  tools in generate_content_config should not be set.
  """
  tools_dict: dict[str, BaseTool] = Field(default_factory=dict, exclude=True)
  """The tools dictionary."""

  cache_config: Optional[ContextCacheConfig] = None
  """Context cache configuration for this request."""

  cache_metadata: Optional[CacheMetadata] = None
  """Cache metadata from previous requests, used for cache management."""

  def append_instructions(
      self, instructions: Union[list[str], types.Content]
  ) -> None:
    """Appends instructions to the system instruction.

    Args:
      instructions: The instructions to append. Can be:
        - list[str]: Strings to append/concatenate to system instruction
        - types.Content: Content object to append to system instruction

    Note: Only text content is supported. Model API requires system_instruction
    to be a string. Non-text parts in Content will be handled differently.

    Behavior:
      - list[str]: concatenates with existing system_instruction using \\n\\n
      - types.Content: extracts text from parts and concatenates
    """

    # Handle Content object - extract only text parts
    if isinstance(instructions, types.Content):
      # TODO: Handle non-text contents in instruction by putting non-text parts
      # into llm_request.contents and adding a reference in the system instruction
      # that references the contents.

      # Extract text from all text parts
      text_parts = [part.text for part in instructions.parts if part.text]

      if not text_parts:
        return  # No text content to append

      new_text = "\n\n".join(text_parts)
      if not self.config.system_instruction:
        self.config.system_instruction = new_text
      elif isinstance(self.config.system_instruction, str):
        self.config.system_instruction += "\n\n" + new_text
      else:
        # Log warning for unsupported system_instruction types
        logging.warning(
            "Cannot append to system_instruction of unsupported type: %s. "
            "Only string system_instruction is supported.",
            type(self.config.system_instruction),
        )
      return

    # Handle list of strings
    if isinstance(instructions, list) and all(
        isinstance(inst, str) for inst in instructions
    ):
      if not instructions:  # Handle empty list
        return

      new_text = "\n\n".join(instructions)
      if not self.config.system_instruction:
        self.config.system_instruction = new_text
      elif isinstance(self.config.system_instruction, str):
        self.config.system_instruction += "\n\n" + new_text
      else:
        # Log warning for unsupported system_instruction types
        logging.warning(
            "Cannot append to system_instruction of unsupported type: %s. "
            "Only string system_instruction is supported.",
            type(self.config.system_instruction),
        )
      return

    # Invalid input
    raise TypeError("instructions must be list[str] or types.Content")

  def append_tools(self, tools: list[BaseTool]) -> None:
    """Appends tools to the request.

    Args:
      tools: The tools to append.
    """

    if not tools:
      return
    declarations = []
    for tool in tools:
      declaration = tool._get_declaration()
      if declaration:
        declarations.append(declaration)
        self.tools_dict[tool.name] = tool
    if declarations:
      if self.config.tools is None:
        self.config.tools = []

      # Find existing tool with function_declarations and append to it
      if tool_with_function_declarations := _find_tool_with_function_declarations(
          self
      ):
        if tool_with_function_declarations.function_declarations is None:
          tool_with_function_declarations.function_declarations = []
        tool_with_function_declarations.function_declarations.extend(
            declarations
        )
      else:
        # No existing tool with function_declarations, create new one
        self.config.tools.append(types.Tool(function_declarations=declarations))

  def set_output_schema(self, base_model: type[BaseModel]) -> None:
    """Sets the output schema for the request.

    Args:
      base_model: The pydantic base model to set the output schema to.
    """

    self.config.response_schema = base_model
    self.config.response_mime_type = "application/json"
