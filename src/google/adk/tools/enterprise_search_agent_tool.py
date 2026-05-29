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

from typing import Union

from ..agents.llm_agent import LlmAgent
from ..models.base_llm import BaseLlm
from ._search_agent_tool import _SearchAgentTool
from .enterprise_search_tool import enterprise_web_search_tool


def create_enterprise_search_agent(model: Union[str, BaseLlm]) -> LlmAgent:
  """Create a sub-agent that only uses enterprise_web_search tool."""
  return LlmAgent(
      name='enterprise_search_agent',
      model=model,
      description=(
          'An agent for performing Enterprise search using the'
          ' `enterprise_web_search` tool'
      ),
      instruction="""
        You are a specialized Enterprise search agent.

        When given a search query, use the `enterprise_web_search` tool to find the related information.
      """,
      tools=[enterprise_web_search_tool],
  )


class EnterpriseSearchAgentTool(_SearchAgentTool):
  """A tool that wraps a sub-agent that only uses enterprise_web_search tool.

  This is a workaround to support using enterprise_web_search tool with other tools.
  TODO(b/448114567): Remove once the workaround is no longer needed.

  Attributes:
    agent: The sub-agent that this tool wraps.
  """

  def __init__(self, agent: LlmAgent):
    self.agent = agent
    super().__init__(agent=self.agent)
