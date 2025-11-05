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

import inspect
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.utils import instructions_utils

if TYPE_CHECKING:
  from google.adk.agents.llm_agent import InstructionProvider
  from google.adk.agents.llm_agent import LlmAgent


class GlobalInstructionPlugin(BasePlugin):
  """Plugin that provides global instructions functionality at the App level.

  This plugin replaces the deprecated global_instruction field on LlmAgent.
  Global instructions are applied to all agents in the application, providing
  a consistent way to set application-wide instructions, identity, or
  personality.

  The plugin operates through the before_model_callback, allowing it to modify
  LLM requests before they are sent to the model.
  """

  def __init__(
      self,
      global_instruction: Union[str, InstructionProvider] = "",
      name: str = "global_instruction",
  ) -> None:
    """Initialize the GlobalInstructionPlugin.

    Args:
      global_instruction: The instruction to apply globally. Can be a string or
        an InstructionProvider function that takes ReadonlyContext and returns a
        string (sync or async).
      name: The name of the plugin (defaults to "global_instruction").
    """
    super().__init__(name=name)
    self.global_instruction = global_instruction

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Apply global instructions to the LLM request.

    This callback is executed before each request is sent to the model,
    allowing the plugin to inject global instructions into the request.

    Args:
      callback_context: The context for the current agent call.
      llm_request: The prepared request object to be sent to the model.

    Returns:
      None to allow the LLM request to proceed normally.
    """
    # Only process if we have a global instruction configured
    if not self.global_instruction:
      return None

    # Resolve the global instruction (handle both string and InstructionProvider)
    final_global_instruction = await self._resolve_global_instruction(
        callback_context
    )

    if not final_global_instruction:
      return None

    # Make the global instruction the leading system instruction.
    existing_instruction = llm_request.config.system_instruction

    if not existing_instruction:
      llm_request.config.system_instruction = final_global_instruction
      return None

    if isinstance(existing_instruction, str):
      llm_request.config.system_instruction = (
          f"{final_global_instruction}\n\n{existing_instruction}"
      )
    else:  # It's an Iterable
      # Convert to list to allow prepending
      new_instruction_list = [final_global_instruction]
      new_instruction_list.extend(list(existing_instruction))
      llm_request.config.system_instruction = new_instruction_list

    return None

  async def _resolve_global_instruction(
      self, readonly_context: ReadonlyContext
  ) -> str:
    """Resolve the global instruction, handling both string and InstructionProvider.

    Args:
      readonly_context: The readonly context for resolving instructions.

    Returns:
      The fully resolved and processed global instruction string, ready to use.
    """
    if isinstance(self.global_instruction, str):
      # For string instructions, apply state injection
      return await instructions_utils.inject_session_state(
          self.global_instruction, readonly_context
      )
    else:
      # Handle InstructionProvider (callable)
      # InstructionProvider already handles state internally, no injection needed
      instruction = self.global_instruction(readonly_context)
      if inspect.isawaitable(instruction):
        instruction = await instruction
      return instruction
