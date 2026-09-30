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

from google.adk.agents import LlmAgent


def test_single_turn_managed_agent_is_wrapped_as_tool():
  from google.adk.agents import ManagedAgent
  from google.adk.tools.agent_tool import _SingleTurnAgentTool

  managed = ManagedAgent(name='m', agent_id='a', mode='single_turn')
  coordinator = LlmAgent(name='c', sub_agents=[managed])

  assert any(
      isinstance(t, _SingleTurnAgentTool) and t.agent is managed
      for t in coordinator.tools
  )


def test_managed_agent_without_mode_is_not_wrapped():
  from google.adk.agents import ManagedAgent
  from google.adk.tools.agent_tool import _SingleTurnAgentTool

  managed = ManagedAgent(name='m', agent_id='a')  # mode defaults to None
  coordinator = LlmAgent(name='c', sub_agents=[managed])

  assert not any(isinstance(t, _SingleTurnAgentTool) for t in coordinator.tools)


def test_single_turn_managed_agent_excluded_from_transfer_targets():
  from google.adk.agents import ManagedAgent
  from google.adk.flows.llm_flows.agent_transfer import _get_transfer_targets

  managed = ManagedAgent(name='m', agent_id='a', mode='single_turn')
  coordinator = LlmAgent(name='c', sub_agents=[managed])

  targets = _get_transfer_targets(coordinator)
  assert managed not in targets
