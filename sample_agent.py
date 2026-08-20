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

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent


def search_web(query: str) -> str:
  """Searches the web for given query."""
  return f"Search results for: {query}"


def calculate(expression: str) -> str:
  """Calculates math expression."""
  return f"Calculated: {expression}"


# 1. Researcher Agent with web search tool
researcher = LlmAgent(
    name="ResearcherAgent",
    instruction="Search the web and gather background information.",
    tools=[search_web],
)

# 2. Analyst Agent with calculator tool
analyst = LlmAgent(
    name="AnalystAgent",
    instruction="Analyze research data and calculate statistics.",
    tools=[calculate],
)

# 3. Writer Agent
writer = LlmAgent(
    name="WriterAgent",
    instruction="Synthesize findings and draft final executive report.",
)

# Root Sequential Pipeline
root_agent = SequentialAgent(
    name="ResearchAndReportingPipeline",
    description="Multi-agent workflow that researches, analyzes, and drafts reports.",
    sub_agents=[researcher, analyst, writer],
)

# New Sequential Pipeline
ResearchAndReportingPipeline = SequentialAgent(
    name="ResearchAndReportingPipeline",
    description="Multi-agent workflow that researches, analyzes, and drafts reports.",
    sub_agents=[],
)
