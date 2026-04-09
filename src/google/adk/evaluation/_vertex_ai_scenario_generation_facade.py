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

"""Vertex AI Scenario Generation Facade."""

from __future__ import annotations

import logging
import os

from . import conversation_scenarios
from ..agents import base_agent
from ..dependencies.vertexai import vertexai

types = vertexai.types


logger = logging.getLogger("google_adk." + __name__)

_ERROR_MESSAGE_SUFFIX = """
You should specify both project id and location. This metric uses Vertex Gen AI
Eval SDK, and it requires google cloud credentials.

If using an .env file add the values there, or explicitly set in the code using
the template below:

os.environ['GOOGLE_CLOUD_LOCATION'] = <LOCATION>
os.environ['GOOGLE_CLOUD_PROJECT'] = <PROJECT ID>
"""


class ScenarioGenerator:
  """Facade for generating eval scenarios using Vertex Gen AI Eval SDK.

  Using this class requires a GCP project. Please set GOOGLE_CLOUD_PROJECT and
  GOOGLE_CLOUD_LOCATION in your .env file.
  """

  def __init__(self):
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    api_key = os.environ.get("GOOGLE_API_KEY")

    if api_key:
      self._client = vertexai.Client(api_key=api_key)
    elif project_id or location:
      if not project_id:
        raise ValueError("Missing project id." + _ERROR_MESSAGE_SUFFIX)
      if not location:
        raise ValueError("Missing location." + _ERROR_MESSAGE_SUFFIX)
      self._client = vertexai.Client(project=project_id, location=location)
    else:
      raise ValueError(
          "Either API Key or Google cloud Project id and location should be"
          " specified."
      )

  def generate_scenarios(
      self,
      agent: base_agent.BaseAgent,
      config: conversation_scenarios.ConversationGenerationConfig,
  ) -> list[conversation_scenarios.ConversationScenario]:
    """Generates conversation scenarios for the specified agent.

    Args:
      agent: The root agent representing the system under test.
      config: The configuration for ConversationGenerationConfig.

    Returns:
      A list of ADK ConversationScenario objects.
    """
    agent_info = types.evals.AgentInfo.load_from_agent(agent=agent)

    vertex_config = types.evals.UserScenarioGenerationConfig(
        count=config.count,
        generation_instruction=config.generation_instruction,
        environment_context=config.environment_context,
        model_name=config.model_name,
    )

    eval_dataset = self._client.evals.generate_conversation_scenarios(
        agent_info=agent_info,
        config=vertex_config,
    )

    scenarios = []
    for eval_case in eval_dataset.eval_cases:
      if not eval_case.user_scenario:
        continue
      scenarios.append(
          conversation_scenarios.ConversationScenario(
              starting_prompt=eval_case.user_scenario.starting_prompt,
              conversation_plan=eval_case.user_scenario.conversation_plan,
          )
      )

    return scenarios
