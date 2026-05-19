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

from typing import Any

from google.genai import types
import pydantic

from ..agents.llm_agent import LlmAgent
from ..agents.llm_agent import ToolUnion
from ..tools.base_tool import BaseTool
from ..tools.base_toolset import BaseToolset
from ..tools.function_tool import FunctionTool


class AgentInfo(pydantic.BaseModel):
  name: str
  description: str
  instruction: str
  tools: list[types.Tool]
  sub_agents: list[str]


def get_tools_info(tools: list[ToolUnion]) -> list[Any]:
  """Returns the info for a given list of tools."""
  final_tools = []
  for tool in tools:
    if isinstance(tool, BaseTool):
      final_tools.append(tool)
    elif isinstance(tool, BaseToolset):
      final_tools.extend(tool.get_tools())
    else:
      final_tools.append(FunctionTool(tool))
  return [
      types.Tool(function_declarations=[tool._get_declaration()])
      for tool in final_tools
      if tool._get_declaration()
  ]


def get_agents_dict(agent: LlmAgent) -> dict[str, AgentInfo]:
  """Returns a dict with info for the agent and its sub-agents."""
  agents_dict = {}

  def _traverse(current_agent: LlmAgent):
    if current_agent.name in agents_dict:
      return

    sub_agent_names = []
    for sub_agent in current_agent.sub_agents:
      if isinstance(sub_agent, LlmAgent):
        _traverse(sub_agent)
        sub_agent_names.append(sub_agent.name)

    agents_dict[current_agent.name] = AgentInfo(
        name=current_agent.name,
        description=current_agent.description,
        instruction=current_agent.instruction,
        tools=get_tools_info(current_agent.tools),
        sub_agents=sub_agent_names,
    )

  _traverse(agent)
  return agents_dict
