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

import re
from typing import AsyncGenerator
from typing import Generator
from typing import TYPE_CHECKING

from typing_extensions import override

from ...agents.readonly_context import ReadonlyContext
from ...events.event import Event
from ...sessions.state import State
from ...utils import instructions_utils
from ._base_llm_processor import BaseLlmRequestProcessor

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext
  from ...models.llm_request import LlmRequest


class _InstructionsLlmRequestProcessor(BaseLlmRequestProcessor):
  """Handles instructions and global instructions for LLM flow."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.base_agent import BaseAgent
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    root_agent: BaseAgent = agent.root_agent

    # Appends global instructions if set.
    if (
        isinstance(root_agent, LlmAgent) and root_agent.global_instruction
    ):  # not empty str
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

    # Appends agent instructions if set.
    if agent.instruction:  # not empty str
      raw_si, bypass_state_injection = await agent.canonical_instruction(
          ReadonlyContext(invocation_context)
      )
      si = raw_si
      if not bypass_state_injection:
        si = await instructions_utils.inject_session_state(
            raw_si, ReadonlyContext(invocation_context)
        )
      llm_request.append_instructions([si])

    # Maintain async generator behavior
    if False:  # Ensures it behaves as a generator
      yield  # This is a no-op but maintains generator structure


request_processor = _InstructionsLlmRequestProcessor()
