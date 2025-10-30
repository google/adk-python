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

"""Handles agent transfer for LLM flow."""

from __future__ import annotations

import typing
from typing import AsyncGenerator

from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ...tools.function_tool import FunctionTool
from ...tools.tool_context import ToolContext
from ...tools.transfer_to_agent_tool import transfer_to_agent
from ._base_llm_processor import BaseLlmRequestProcessor

if typing.TYPE_CHECKING:
  from ...agents import BaseAgent
  from ...agents import LlmAgent


class _AgentTransferLlmRequestProcessor(BaseLlmRequestProcessor):
  """Agent transfer request processor."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.llm_agent import LlmAgent

    if not isinstance(invocation_context.agent, LlmAgent):
      return

    transfer_targets = _get_transfer_targets(invocation_context.agent)
    if not transfer_targets:
      return

    llm_request.append_instructions([
        _build_target_agents_instructions(
            invocation_context.agent, transfer_targets
        )
    ])

    transfer_to_agent_tool = FunctionTool(func=transfer_to_agent)
    tool_context = ToolContext(invocation_context)
    await transfer_to_agent_tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    return
    yield  # AsyncGenerator requires yield statement in function body.


request_processor = _AgentTransferLlmRequestProcessor()


def _build_target_agents_info(target_agent: BaseAgent) -> str:
  return f"""
Agent name: {target_agent.name}
Agent description: {target_agent.description}
"""


line_break = '\n'


def _build_target_agents_instructions(
    agent: LlmAgent, target_agents: list[BaseAgent]
) -> str:
  # Build list of available agent names for the NOTE
  # target_agents already includes parent agent if applicable, so no need to add it again
  available_agent_names = [target_agent.name for target_agent in target_agents]

  # Sort for consistency
  available_agent_names.sort()

  # Format agent names with backticks for clarity
  formatted_agent_names = ', '.join(
      f'`{name}`' for name in available_agent_names
  )

  si = f"""
You have a list of other agents to transfer to:

{line_break.join([
    _build_target_agents_info(target_agent) for target_agent in target_agents
])}

If you are the best to answer the question according to your description, you
can answer it.

If another agent is better for answering the question according to its
description, call `{_TRANSFER_TO_AGENT_FUNCTION_NAME}` function to transfer the
question to that agent. When transferring, do not generate any text other than
the function call.

**NOTE**: the only available agents for `{_TRANSFER_TO_AGENT_FUNCTION_NAME}` function are {formatted_agent_names}.
"""

  if agent.parent_agent and not agent.disallow_transfer_to_parent:
    si += f"""
If neither you nor the other agents are best for the question, transfer to your parent agent {agent.parent_agent.name}.
"""
  return si


_TRANSFER_TO_AGENT_FUNCTION_NAME = transfer_to_agent.__name__


def _get_transfer_targets(agent: LlmAgent) -> list[BaseAgent]:
  from ...agents.llm_agent import LlmAgent

  result = []
  result.extend(agent.sub_agents)

  if not agent.parent_agent or not isinstance(agent.parent_agent, LlmAgent):
    return result

  if not agent.disallow_transfer_to_parent:
    result.append(agent.parent_agent)

  if not agent.disallow_transfer_to_peers:
    result.extend([
        peer_agent
        for peer_agent in agent.parent_agent.sub_agents
        if peer_agent.name != agent.name
    ])

  return result
