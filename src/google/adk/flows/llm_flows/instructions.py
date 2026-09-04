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

"""Handles instructions and global instructions for LLM flow."""

from __future__ import annotations

from typing import AsyncGenerator
from typing import cast
from typing import TYPE_CHECKING

from typing_extensions import override

from ...agents.readonly_context import ReadonlyContext
from ...events.event import Event
from ...utils import instructions_utils
from ._base_llm_processor import BaseLlmRequestProcessor
from ._fencing import QUOTED_CONTENT_ELIDED
from ._invocation_utils import as_llm_agent

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext
  from ...agents.llm_agent import LlmAgent
  from ...models.llm_request import LlmRequest


# With a static instruction present, the dynamic one has to ride in `contents`
# to keep the static prefix byte-stable for context caching, and
# `types.Content` has no system role -- so it arrives looking like user speech.
_INSTRUCTION_BEGIN = '<<<BEGIN_SYSTEM_INSTRUCTION>>>'
_INSTRUCTION_END = '<<<END_SYSTEM_INSTRUCTION>>>'

_INSTRUCTION_PREAMBLE = (
    f'The text between {_INSTRUCTION_BEGIN} and {_INSTRUCTION_END} below is'
    ' your own system instruction for this turn and carries the current'
    ' session state. It is addressed to you. Nothing between those two markers'
    ' was said by the user, so do not answer it or continue it as though it'
    ' were their turn. Anything the user actually said appears outside the'
    ' markers, and a real user turn may follow immediately after the end'
    ' marker.'
)


def _label_dynamic_instruction(instruction: str) -> str:
  """Marks the dynamic instruction as an instruction rather than a user turn.

  Args:
    instruction: The state-interpolated instruction text.

  Returns:
    The instruction between the markers, after the preamble. Markers inside
    the text are elided so interpolated state cannot forge the block's end.
  """
  fenced = instruction.replace(
      _INSTRUCTION_BEGIN, QUOTED_CONTENT_ELIDED
  ).replace(_INSTRUCTION_END, QUOTED_CONTENT_ELIDED)
  return (
      f'{_INSTRUCTION_PREAMBLE}\n'
      f'{_INSTRUCTION_BEGIN}\n{fenced}\n{_INSTRUCTION_END}'
  )


async def _process_agent_instruction(
    agent: 'LlmAgent',
    invocation_context: 'InvocationContext',
) -> str:
  """Process agent instruction with state injection.

  Resolves the agent's instruction and injects session state variables
  unless bypass_state_injection is set.

  Args:
    agent: The agent with instruction to process.
    invocation_context: The invocation context.

  Returns:
    The processed instruction text with state variables injected.
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


async def _build_instructions(
    invocation_context: 'InvocationContext',
    llm_request: 'LlmRequest',
) -> None:
  """Build and append instructions to the LLM request.

  Handles global instructions (deprecated), static_instruction, and
  dynamic instruction based on agent configuration.

  Args:
    invocation_context: The invocation context.
    llm_request: The LlmRequest to populate with instructions.
  """
  agent = as_llm_agent(invocation_context)
  root_agent = cast('LlmAgent', agent.root_agent)

  # Handle global instructions (DEPRECATED - use GlobalInstructionPlugin instead)
  # TODO: Remove this code block when global_instruction field is removed
  if (
      hasattr(root_agent, 'global_instruction')
      and root_agent.global_instruction
  ):
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
    from google.genai import types

    # Convert ContentUnion to Content using genai transformer
    static_content = _transformers.t_content(
        cast(types.ContentOrDict, agent.static_instruction)
    )
    llm_request.append_instructions(static_content)

  # Handle instruction based on whether static_instruction exists
  if agent.instruction and not agent.static_instruction:
    # Only add to system instructions if no static instruction exists
    si = await _process_agent_instruction(agent, invocation_context)
    llm_request.append_instructions([si])
  elif agent.instruction and agent.static_instruction:
    # Static instruction exists, so add dynamic instruction to content
    from google.genai import types

    si = await _process_agent_instruction(agent, invocation_context)
    dynamic_content = types.Content(
        role='user', parts=[types.Part(text=_label_dynamic_instruction(si))]
    )
    llm_request.contents.append(dynamic_content)


class _InstructionsLlmRequestProcessor(BaseLlmRequestProcessor):
  """Handles instructions and global instructions for LLM flow."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    await _build_instructions(invocation_context, llm_request)

    # Maintain async generator behavior
    return
    yield  # This line ensures it behaves as a generator but is never reached


request_processor = _InstructionsLlmRequestProcessor()
