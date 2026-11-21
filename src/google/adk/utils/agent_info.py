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

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Any

from google.genai import types
import pydantic

from ..agents.base_agent import BaseAgent
from ..agents.llm_agent import _wrap_base_node_as_tool
from ..agents.llm_agent import LlmAgent
from ..agents.llm_agent import ToolUnion
from ..tools.base_tool import BaseTool
from ..tools.base_toolset import BaseToolset
from ..tools.function_tool import FunctionTool
from ..workflow._base_node import _walk_static_nodes
from ..workflow._base_node import BaseNode

logger = logging.getLogger('google_adk.' + __name__)


class AgentInfo(pydantic.BaseModel):
  name: str
  description: str
  instruction: str
  tools: list[types.Tool]
  sub_agents: list[str]

  @pydantic.field_validator('instruction', mode='before')
  @classmethod
  def _coerce_callable_instruction(cls, value: object) -> object:
    # LlmAgent.instruction is str | InstructionProvider. Providers may be
    # async and may need session state, so app-info must not resolve them.
    if callable(value):
      name = getattr(value, '__name__', None) or type(value).__name__
      return f'<InstructionProvider: {name}>'
    return value


async def get_tools_info(
    tools: Sequence[ToolUnion | BaseNode],
) -> list[Any]:
  """Returns the info for a given list of tools."""
  final_tools = []
  for tool in tools:
    if isinstance(tool, BaseTool):
      final_tools.append(tool)
    elif isinstance(tool, BaseNode):
      final_tools.append(_wrap_base_node_as_tool(tool))
    elif isinstance(tool, BaseToolset):
      # Await the async coroutine call natively!
      tools_res = await tool.get_tools()
      final_tools.extend(tools_res)
    else:
      final_tools.append(FunctionTool(tool))
  declarations = (tool._get_declaration() for tool in final_tools)
  return [
      types.Tool(function_declarations=[declaration])
      for declaration in declarations
      if declaration
  ]


def _nearest_llm_sub_agent_names(agent: BaseAgent) -> list[str]:
  """Returns the unique names of the nearest LlmAgent sub-agents of ``agent``.

  Non-LLM ``BaseAgent`` containers (e.g. ``SequentialAgent``) are flattened so
  their LlmAgent children are reported directly.
  """
  names: dict[str, None] = {}
  visited: set[int] = {id(agent)}

  def _collect(current: BaseAgent) -> None:
    for sub_agent in current.sub_agents:
      if id(sub_agent) in visited:
        continue
      visited.add(id(sub_agent))
      if isinstance(sub_agent, LlmAgent):
        names[sub_agent.name] = None
      else:
        _collect(sub_agent)

  _collect(agent)
  return list(names)


# `LlmAgent | BaseNode` is redundant at runtime but keeps the breaking-change
# detector from flagging the widening from the original `LlmAgent` annotation.
async def get_agents_dict(agent: LlmAgent | BaseNode) -> dict[str, AgentInfo]:
  """Returns a dict with info for every LlmAgent reachable from ``agent``."""
  agents_dict: dict[str, AgentInfo] = {}

  for node in _walk_static_nodes(agent):
    if not isinstance(node, LlmAgent):
      continue
    if node.name in agents_dict:
      logger.warning(
          'Duplicate LlmAgent name %r encountered while collecting app info;'
          ' keeping first agent.',
          node.name,
      )
      continue

    agents_dict[node.name] = AgentInfo(
        name=node.name,
        description=node.description,
        instruction=node.instruction,
        tools=await get_tools_info(node.tools),
        sub_agents=_nearest_llm_sub_agent_names(node),
    )

  return agents_dict
