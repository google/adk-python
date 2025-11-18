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

"""Handles instructions and global instructions for LLM flow."""

from __future__ import annotations

from typing import AsyncGenerator
from typing import TYPE_CHECKING

from typing_extensions import override

from ...agents.readonly_context import ReadonlyContext
from ...events.event import Event
from ...utils import instructions_utils
from ._base_llm_processor import BaseLlmRequestProcessor

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext
  from ...models.llm_request import LlmRequest


class _InstructionsLlmRequestProcessor(BaseLlmRequestProcessor):
  """Handles instructions and global instructions for LLM flow."""

  async def _process_agent_instruction(
      self, agent, invocation_context: InvocationContext
  ) -> str:
    """Process agent instruction with state injection.

    Args:
      agent: The agent with instruction to process
      invocation_context: The invocation context

    Returns:
      The processed instruction text
    """
    raw_si, bypass_state_injection = await agent.canonical_instruction(
        ReadonlyContext(invocation_context)
    )
    si = raw_si
    if not bypass_state_injection:
      si = await instructions_utils.inject_session_state(
          raw_si, ReadonlyContext(invocation_context)
      )
    return si

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.base_agent import BaseAgent
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent

    root_agent: BaseAgent = agent.root_agent

    # Handle global instructions (DEPRECATED - use GlobalInstructionPlugin instead)
    # TODO: Remove this code block when global_instruction field is removed
    if isinstance(root_agent, LlmAgent) and root_agent.global_instruction:
      raw_si, bypass_state_injection = (
          await root_agent.canonical_global_instruction(
              ReadonlyContext(invocation_context)
          )
      )
      si = raw_si
      if not bypass_state_injection:
        si = await instructions_utils.inject_session_state(
            raw_si, ReadonlyContext(invocation_context)
        )
      llm_request.append_instructions([si])

    # Handle static_instruction - add via append_instructions
    if agent.static_instruction:
      from google.genai import _transformers

      # Convert ContentUnion to Content using genai transformer
      static_content = _transformers.t_content(agent.static_instruction)
      llm_request.append_instructions(static_content)

    # Handle instruction based on whether static_instruction exists
    if agent.instruction and not agent.static_instruction:
      # Only add to system instructions if no static instruction exists
      si = await self._process_agent_instruction(agent, invocation_context)
      llm_request.append_instructions([si])
    elif agent.instruction and agent.static_instruction:
      # Static instruction exists, so add dynamic instruction to content
      from google.genai import types

      si = await self._process_agent_instruction(agent, invocation_context)
      # Create user content for dynamic instruction
      dynamic_content = types.Content(role='user', parts=[types.Part(text=si)])
      llm_request.contents.append(dynamic_content)

    # Maintain async generator behavior
    return
    yield  # This line ensures it behaves as a generator but is never reached


request_processor = _InstructionsLlmRequestProcessor()
