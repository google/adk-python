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

import asyncio
import logging
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
  from a2a.types import AgentCard

  from ...agents import BaseAgent
  from ...agents import LlmAgent
  from ...agents.remote_a2a_agent import RemoteA2aAgent

logger = logging.getLogger('google_adk.' + __name__)


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

    # Build instructions asynchronously to support A2A agent card resolution
    instructions = await _build_target_agents_instructions(
        invocation_context.agent, transfer_targets
    )
    llm_request.append_instructions([instructions])

    transfer_to_agent_tool = FunctionTool(func=transfer_to_agent)
    tool_context = ToolContext(invocation_context)
    await transfer_to_agent_tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    return
    yield  # AsyncGenerator requires yield statement in function body.


request_processor = _AgentTransferLlmRequestProcessor()


def _build_target_agent_info_from_card(
    target_agent: RemoteA2aAgent, agent_card: AgentCard
) -> str:
  """Build rich agent info from A2A Agent Card.

  Args:
    target_agent: The RemoteA2aAgent instance
    agent_card: The resolved A2A Agent Card

  Returns:
    Formatted string with detailed agent information from the card,
    optimized for LLM consumption when selecting subagents.
  """
  info_parts = []

  # Start with a clear header for the agent
  info_parts.append(f'### Agent: {target_agent.name}')

  # Include both RemoteA2aAgent description and agent card description
  # This provides both the locally-configured context and the remote agent's self-description
  descriptions = []
  if target_agent.description:
    descriptions.append(f'Description: {target_agent.description}')
  if agent_card.description and agent_card.description != target_agent.description:
    descriptions.append(f'Agent card description: {agent_card.description}')

  if descriptions:
    info_parts.append('\n'.join(descriptions))

  # Add skills in a structured, LLM-friendly format
  if agent_card.skills:
    info_parts.append('\nSkills:')
    for skill in agent_card.skills:
      # Format: "- skill_name: description (tags: tag1, tag2)"
      skill_parts = [f'  - **{skill.name}**']
      if skill.description:
        skill_parts.append(f': {skill.description}')
      if skill.tags:
        skill_parts.append(f' [Tags: {", ".join(skill.tags)}]')
      info_parts.append(''.join(skill_parts))

  return '\n'.join(info_parts)


async def _build_target_agents_info_async(target_agent: BaseAgent) -> str:
  """Build agent info, using A2A Agent Card if available.

  Args:
    target_agent: The agent to build info for

  Returns:
    Formatted string with agent information
  """
  from ...agents.remote_a2a_agent import RemoteA2aAgent

  # Check if this is a RemoteA2aAgent and ensure it's resolved
  if isinstance(target_agent, RemoteA2aAgent):
    try:
      # Ensure the agent card is resolved
      await target_agent._ensure_resolved()

      # If we have an agent card, use it to build rich info
      if target_agent._agent_card:
        return _build_target_agent_info_from_card(
            target_agent, target_agent._agent_card
        )
    except Exception as e:
      # If resolution fails, fall through to default behavior
      logger.warning(
          'Failed to resolve A2A agent card for agent "%s", falling back to' ' basic info. Error: %s',
          target_agent.name,
          e,
      )
      pass
  # Fallback to original behavior for non-A2A agents or if card unavailable
  return _build_target_agents_info(target_agent)


def _build_target_agents_info(target_agent: BaseAgent) -> str:
  """Build basic agent info (fallback for non-A2A agents).

  Args:
    target_agent: The agent to build info for

  Returns:
    Formatted string with basic agent information, matching the enhanced format
    for consistency with A2A agent cards.
  """
  info_parts = [f'### Agent: {target_agent.name}']

  if target_agent.description:
    info_parts.append(f'Description: {target_agent.description}')

  return '\n'.join(info_parts)


line_break = '\n'


async def _build_target_agents_instructions(
    agent: LlmAgent, target_agents: list[BaseAgent]
) -> str:
  """Build instructions for agent transfer with detailed agent information.

  Args:
    agent: The current agent
    target_agents: List of agents that can be transferred to

  Returns:
    Formatted instructions string with agent transfer information,
    optimized for LLM decision-making about which subagent to use.
  """
  # Build list of available agent names for the NOTE
  # target_agents already includes parent agent if applicable, so no need to add it again
  available_agent_names = [target_agent.name for target_agent in target_agents]

  # Sort for consistency
  available_agent_names.sort()

  # Format agent names with backticks for clarity
  formatted_agent_names = ', '.join(
      f'`{name}`' for name in available_agent_names
  )

  # Build agent info asynchronously and concurrently to support A2A agent card resolution
  tasks = [
      _build_target_agents_info_async(target_agent)
      for target_agent in target_agents
  ]
  agent_info_list = await asyncio.gather(*tasks)

  # Create a separator for visual clarity
  agents_section = '\n\n'.join(agent_info_list)

  si = f"""
## Available Agents for Transfer

You can delegate tasks to the following specialized agents. Carefully review each agent's description and skills to determine the best match for the user's request.

{agents_section}

## Decision Criteria

1. **Assess your own capability**: If you are the best agent to handle this request based on your own description and capabilities, answer it directly.

2. **Consider specialized agents**: If another agent has more relevant skills or expertise for this request, call the `{_TRANSFER_TO_AGENT_FUNCTION_NAME}` function to transfer to that agent. Match the user's needs with the agent's skills and descriptions above.

3. **When transferring**: Only call the function - do not generate any additional text.

**IMPORTANT**: The only valid agent names for `{_TRANSFER_TO_AGENT_FUNCTION_NAME}` are: {formatted_agent_names}
"""

  if agent.parent_agent and not agent.disallow_transfer_to_parent:
    si += f"""
4. **Escalate to parent**: If neither you nor the specialized agents are suitable for this request, transfer to your parent agent `{agent.parent_agent.name}` for broader assistance.
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
