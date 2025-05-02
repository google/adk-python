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
      raw_si = root_agent.canonical_global_instruction(
          ReadonlyContext(invocation_context)
      )
      si = _populate_values(raw_si, invocation_context)
      llm_request.append_instructions([si])

    # Appends agent instructions if set.
    if agent.instruction:  # not empty str
      raw_si = agent.canonical_instruction(ReadonlyContext(invocation_context))
      si = _populate_values(raw_si, invocation_context)
      llm_request.append_instructions([si])

    # Maintain async generator behavior
    if False:  # Ensures it behaves as a generator
      yield  # This is a no-op but maintains generator structure


request_processor = _InstructionsLlmRequestProcessor()


def _populate_values(
    instruction_template: str,
    context: InvocationContext,
) -> str:
  """Populates values in the instruction template, e.g. state, artifact, etc."""

  def _replace_match(match) -> str:
    var_name = match.group().lstrip('{').rstrip('}').strip()
    optional = False
    if var_name.endswith('?'):
      optional = True
      var_name = var_name.removesuffix('?')
    if var_name.startswith('artifact.'):
      var_name = var_name.removeprefix('artifact.')
      if context.artifact_service is None:
        raise ValueError('Artifact service is not initialized.')
      artifact = context.artifact_service.load_artifact(
          app_name=context.session.app_name,
          user_id=context.session.user_id,
          session_id=context.session.id,
          filename=var_name,
      )
      if not var_name:
        raise KeyError(f'Artifact {var_name} not found.')
      return str(artifact)
    else:
      if not _is_valid_state_name(var_name):
        return match.group()
      if var_name in context.session.state:
        return str(context.session.state[var_name])
      else:
        if optional:
          return ''
        else:
          raise KeyError(f'Context variable not found: `{var_name}`.')

  return re.sub(r'{+[^{}]*}+', _replace_match, instruction_template)


def _is_valid_state_name(var_name):
  """Checks if the variable name is a valid state name.

  Valid state is either:
    - Valid identifier
    - <Valid prefix>:<Valid identifier>
  All the others will just return as it is.

  Args:
    var_name: The variable name to check.

  Returns:
    True if the variable name is a valid state name, False otherwise.
  """
  parts = var_name.split(':')
  if len(parts) == 1:
    return var_name.isidentifier()

  if len(parts) == 2:
    prefixes = [State.APP_PREFIX, State.USER_PREFIX, State.TEMP_PREFIX]
    if (parts[0] + ':') in prefixes:
      return parts[1].isidentifier()
  return False
