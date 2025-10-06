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

"""Sub-agent for ADK Knowledge."""

from google.adk.agents.llm_agent import Agent
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent


def create_adk_knowledge_agent() -> Agent:
  """Create a sub-agent that only uses google_search tool."""
  return RemoteA2aAgent(
      name="adk_knowledge_agent",
      description=(
          "Agent for performing Vertex AI Search to find ADK knowledge and"
          " documentation"
      ),
      agent_card=(
          f"https://adk-agent-builder-knowledge-service-654646711756.us-central1.run.app/a2a/adk_knowledge_agent{AGENT_CARD_WELL_KNOWN_PATH}"
      ),
  )
