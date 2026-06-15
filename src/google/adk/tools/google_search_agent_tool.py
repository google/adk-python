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

from typing import Union

from ..agents.llm_agent import LlmAgent
from ..models.base_llm import BaseLlm
from .agent_tool import AgentTool
from .google_search_tool import google_search


def create_google_search_agent(model: Union[str, BaseLlm]) -> LlmAgent:
  """Create a sub-agent that only uses google_search tool."""
  return LlmAgent(
      name='google_search_agent',
      model=model,
      description=(
          'An agent for performing Google search using the `google_search` tool'
      ),
      instruction="""
        You are a specialized Google search agent.

        When given a search query, use the `google_search` tool to find the related information.
      """,
      tools=[google_search],
  )


class GoogleSearchAgentTool(AgentTool):
  """A tool that wraps a sub-agent that only uses google_search tool.

  This is a workaround to support using google_search tool with other tools.
  TODO(b/448114567): Remove once the workaround is no longer needed.

  Attributes:
    model: The model to use for the sub-agent.
  """

  def __init__(self, agent: LlmAgent):
    self.agent = agent
    super().__init__(agent=self.agent, propagate_grounding_metadata=True)
