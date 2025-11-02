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

from google.adk.evaluation.app_details import AgentDetails
from google.adk.evaluation.app_details import AppDetails
from google.genai import types as genai_types
from pytest import raises


def test_get_developer_instructions_existing_agent():
  agent_details = {
      'agent1': AgentDetails(
          name='agent1', instructions='instruction for agent1'
      ),
      'agent2': AgentDetails(
          name='agent2', instructions='instruction for agent2'
      ),
  }
  app_details = AppDetails(
      agent_details=agent_details,
  )

  # Test for existing agent
  instructions = app_details.get_developer_instructions('agent1')
  assert instructions == 'instruction for agent1'


def test_get_developer_instructions_non_existing_Agent():
  agent_details = {
      'agent1': AgentDetails(
          name='agent1', instructions='instruction for agent1'
      ),
      'agent2': AgentDetails(
          name='agent2', instructions='instruction for agent2'
      ),
  }
  app_details = AppDetails(
      agent_details=agent_details,
  )

  # Test for existing agent
  with raises(ValueError, match='`agent3` not found in the agentic system.'):
    app_details.get_developer_instructions('agent3')


def test_get_tools_by_agent_name():
  tool1 = genai_types.Tool(
      function_declarations=[genai_types.FunctionDeclaration(name='tool1_func')]
  )
  agent_details = {
      'agent1': AgentDetails(name='agent1', tool_declarations=[tool1]),
      'agent2': AgentDetails(name='agent2', tool_declarations=[]),
  }
  app_details = AppDetails(
      agent_details=agent_details,
  )

  tools = app_details.get_tools_by_agent_name()
  expected_tools = {'agent1': [tool1], 'agent2': []}
  assert tools == expected_tools
