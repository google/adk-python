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

"""Handles agent transfer for LLM flow."""

from __future__ import annotations

import asyncio
import logging
import typing
from typing import AsyncGenerator
from typing import Sequence

from typing_extensions import override

from ....agents.invocation_context import InvocationContext
from ....events.event import Event
from ....models.llm_request import LlmRequest
from ....tools.tool_context import ToolContext
from ....tools.transfer_to_agent_tool import TransferToAgentTool
from .._base_llm_processor import BaseLlmRequestProcessor
from ..core._utils import as_llm_agent

if typing.TYPE_CHECKING:
  from ....agents.base_agent import BaseAgent
  from ....agents.llm_agent import LlmAgent


logger = logging.getLogger('google_adk.' + __name__)


class _AgentTransferLlmRequestProcessor(BaseLlmRequestProcessor):
  """Agent transfer request processor."""

  name = 'agent_transfer'

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    agent = as_llm_agent(invocation_context)
    transfer_targets = _get_transfer_targets(agent)
    if not transfer_targets:
      return

    model = (
        getattr(getattr(agent, 'canonical_live_model', None), 'model', None)
        or llm_request.model
        if invocation_context.live_request_queue is not None
        else llm_request.model
    )
    if err_msg := _get_incompatible_builtin_tool_error(agent, model):
      sub_agents = getattr(agent, 'sub_agents', None) or []
      sub_agent_targets = [t for t in transfer_targets if t in sub_agents]
      if sub_agent_targets:
        raise ValueError(err_msg)
      return

    transfer_to_agent_tool = _build_transfer_tool(transfer_targets)

    if agent.mode not in ('task', 'single_turn'):
      transfer_target_infos = await asyncio.gather(*[
          _build_transfer_target_info(target, invocation_context)
          for target in transfer_targets
      ])

      llm_request.append_instructions([
          _build_transfer_instructions(
              transfer_to_agent_tool.name,
              agent,
              transfer_target_infos,
          )
      ])

    tool_context = ToolContext(invocation_context)
    await transfer_to_agent_tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    return
    yield  # AsyncGenerator requires yield statement in function body.


request_processor = _AgentTransferLlmRequestProcessor()


class _AgentLike(typing.Protocol):
  name: str
  description: str


class _TransferTargetInfo:
  """Invocation-scoped metadata used to build transfer instructions."""

  def __init__(self, *, name: str, description: str) -> None:
    self.name = name
    self.description = description


async def _build_transfer_target_info(
    target_agent: BaseAgent,
    ctx: InvocationContext,
) -> _TransferTargetInfo:
  """Builds transfer metadata without mutating invocation-scoped values."""
  cache_key = f'_transfer_target_info_{target_agent.name}'
  metadata = getattr(ctx, '_private_metadata', None)
  if isinstance(metadata, dict) and cache_key in metadata:
    cached = metadata[cache_key]
    if isinstance(cached, _TransferTargetInfo):
      return cached

  get_transfer_description = getattr(
      target_agent, '_get_transfer_description', None
  )
  if callable(get_transfer_description):
    try:
      description = await asyncio.wait_for(
          get_transfer_description(ctx), timeout=5.0
      )
    except Exception as e:
      logger.warning(
          'Failed to load transfer description for agent %s: %s',
          target_agent.name,
          e,
      )
      description = target_agent.description
  else:
    description = target_agent.description

  info = _TransferTargetInfo(
      name=target_agent.name,
      description=description,
  )
  if isinstance(metadata, dict):
    metadata[cache_key] = info
  return info


def _build_target_agents_info(target_agent: _AgentLike) -> str:
  return f"""
Agent name: {target_agent.name}
Agent description: {target_agent.description}
"""


line_break = '\n'


def _build_transfer_instruction_body(
    tool_name: str,
    target_agents: Sequence[_AgentLike],
) -> str:
  """Build the core transfer instruction text.

  This is the agent-tree-agnostic portion of transfer instructions. It
  works with any objects exposing agent names and descriptions.

  Args:
    tool_name: The name of the transfer tool (e.g. 'transfer_to_agent').
    target_agents: Agents available as transfer targets.

  Returns:
    Instruction text for the LLM about agent transfers.
  """
  available_agent_names = [t.name for t in target_agents]
  available_agent_names.sort()
  formatted_agent_names = ', '.join(
      f'`{name}`' for name in available_agent_names
  )

  return f"""
You have a list of other agents to transfer to:

{line_break.join([
    _build_target_agents_info(target_agent) for target_agent in target_agents
])}

If you are the best to answer the question according to your description,
you can answer it.

If another agent is better for answering the question according to its
description, call `{tool_name}` function to transfer the question to that
agent. When transferring, do not generate any text other than the function
call.

**NOTE**: the only available agents for `{tool_name}` function are
{formatted_agent_names}.
"""


def _build_transfer_instructions(
    tool_name: str,
    agent: LlmAgent,
    target_agents: Sequence[_AgentLike],
) -> str:
  """Build instructions for agent transfer (agent-tree variant).

  Delegates to ``_build_transfer_instruction_body`` for the core text,
  then appends parent-agent-specific instructions if applicable.

  Args:
    tool_name: The name of the transfer tool (e.g. 'transfer_to_agent').
    agent: The current agent that may initiate transfers.
    target_agents: List of agents that can be transferred to.

  Returns:
    Instruction text for the LLM about agent transfers.
  """
  if agent.mode in ('task', 'single_turn'):
    return ''

  si = _build_transfer_instruction_body(tool_name, target_agents)

  if agent.parent_agent and not agent.disallow_transfer_to_parent:
    si += f"""
If neither you nor the other agents are best for the question, transfer to your parent agent {agent.parent_agent.name}.
"""
  return si


def _get_transfer_targets(agent: BaseAgent) -> list[BaseAgent]:
  """Gets the list of agents that the current agent can transfer to.

  The transfer targets include:
  1.  Sub-agents of the current agent, excluding those in 'single_turn' mode.
  2.  The parent agent, if it exists and the current agent does not disallow
      transfer to the parent.
  3.  Peer agents (other sub-agents of the parent), if the current agent does
      not disallow transfer to peers.

  Args:
    agent: The BaseAgent for which to find transfer targets.

  Returns:
    A list of BaseAgent instances that are valid transfer targets.
  """
  if not hasattr(agent, 'disallow_transfer_to_parent'):
    return []

  result = []
  if hasattr(agent, 'sub_agents') and agent.sub_agents:
    result.extend([
        sub_agent
        for sub_agent in agent.sub_agents
        if not hasattr(sub_agent, 'mode')
        or sub_agent.mode not in ('single_turn', 'task')
    ])

  parent = getattr(agent, 'parent_agent', None)
  if not parent or not hasattr(parent, 'disallow_transfer_to_parent'):
    return result

  if not getattr(agent, 'disallow_transfer_to_parent', False):
    result.append(parent)

  if not getattr(agent, 'disallow_transfer_to_peers', False):
    result.extend([
        peer_agent
        for peer_agent in getattr(parent, 'sub_agents', []) or []
        if getattr(peer_agent, 'name', None) != getattr(agent, 'name', None)
        and (
            not hasattr(peer_agent, 'mode')
            or peer_agent.mode not in ('single_turn', 'task')
        )
    ])

  return result


def _build_transfer_tool(
    transfer_targets: Sequence[BaseAgent],
) -> TransferToAgentTool:
  """Builds the transfer tool offering the given agents as targets.

  Args:
    transfer_targets: The agents that can be transferred to.

  Returns:
    A TransferToAgentTool for the given targets.
  """
  return TransferToAgentTool(
      agent_names=[target.name for target in transfer_targets]
  )


def _get_incompatible_builtin_tool_error(
    agent: BaseAgent, model: str | None = None
) -> str | None:
  """Returns an error message if the agent uses built-in tools incompatible with function calling.

  Args:
    agent: The agent whose ``tools`` are inspected.
    model: The resolved model name for the request. Gemini 3 and later accept
      built-in tools alongside function declarations, so no error is returned
      for them. A ``model`` override on the tool itself takes precedence.

  Returns:
    An error message, or None if the configuration is compatible.
  """
  tools = getattr(agent, 'tools', None)
  if not tools:
    return None

  from ....tools.enterprise_search_tool import EnterpriseWebSearchTool
  from ....tools.google_search_tool import GoogleSearchTool
  from ....tools.vertex_ai_search_tool import VertexAiSearchTool
  from ....utils.model_name_utils import _supports_builtin_tools_with_function_calling

  agent_name = getattr(agent, 'name', '')
  for tool in tools:
    if not isinstance(
        tool, (GoogleSearchTool, VertexAiSearchTool, EnterpriseWebSearchTool)
    ):
      continue
    if _supports_builtin_tools_with_function_calling(
        getattr(tool, 'model', None) or model
    ):
      continue
    if (
        isinstance(tool, (GoogleSearchTool, VertexAiSearchTool))
        and not tool.bypass_multi_tools_limit
    ):
      return (
          f"Agent '{agent_name}' has sub-agent transfer targets but is"
          f' configured with {tool.__class__.__name__} without'
          ' bypass_multi_tools_limit=True. Gemini models before 3.0 do not'
          ' allow built-in search tools to be combined with function calling'
          ' (agent delegation). To enable both search and sub-agent'
          ' delegation, use a Gemini 3+ model or set'
          ' bypass_multi_tools_limit=True on GoogleSearchTool or'
          ' VertexAiSearchTool.'
      )
    if isinstance(tool, EnterpriseWebSearchTool):
      return (
          f"Agent '{agent_name}' has sub-agent transfer targets but is"
          ' configured with EnterpriseWebSearchTool. Gemini models before 3.0'
          ' do not allow EnterpriseWebSearchTool to be combined with function'
          ' calling (agent delegation). To enable both, use a Gemini 3+ model'
          ' or move EnterpriseWebSearchTool into a mode="single_turn"'
          ' sub-agent.'
      )
  return None


__all__ = [
    '_AgentLike',
    '_AgentTransferLlmRequestProcessor',
    '_build_target_agents_info',
    '_build_transfer_instruction_body',
    '_build_transfer_instructions',
    '_build_transfer_tool',
    '_get_incompatible_builtin_tool_error',
    '_get_transfer_targets',
    'line_break',
    'request_processor',
]
