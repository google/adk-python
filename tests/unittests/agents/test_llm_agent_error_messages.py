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

"""Tests for enhanced error messages in agent handling."""

from google.adk.agents.llm_agent import LlmAgent
from google.genai import types
import pytest


def test_agent_not_found_enhanced_error():
  """Verify enhanced error message for agent not found."""
  root_agent = LlmAgent(
      name='root',
      model='gemini-2.5-flash',
      sub_agents=[
          LlmAgent(name='agent_a', model='gemini-2.5-flash'),
          LlmAgent(name='agent_b', model='gemini-2.5-flash'),
      ],
  )

  with pytest.raises(ValueError) as exc_info:
    root_agent._LlmAgent__get_agent_to_run('nonexistent_agent')

  error_msg = str(exc_info.value)

  # Verify error message components
  assert 'nonexistent_agent' in error_msg
  assert 'Available agents:' in error_msg
  assert 'agent_a' in error_msg
  assert 'agent_b' in error_msg
  assert 'Possible causes:' in error_msg
  assert 'Suggested fixes:' in error_msg


def test_agent_tree_traversal():
  """Verify agent tree traversal helper works correctly."""
  root_agent = LlmAgent(
      name='orchestrator',
      model='gemini-2.5-flash',
      sub_agents=[
          LlmAgent(
              name='parent_agent',
              model='gemini-2.5-flash',
              sub_agents=[
                  LlmAgent(name='child_agent', model='gemini-2.5-flash'),
              ],
          ),
      ],
  )

  available_agents = root_agent._get_available_agent_names()

  # Verify all agents in tree are found
  assert 'orchestrator' in available_agents
  assert 'parent_agent' in available_agents
  assert 'child_agent' in available_agents
  assert len(available_agents) == 3


def test_agent_not_found_shows_all_agents():
  """Verify error message shows all agents (no truncation)."""
  # Create 100 sub-agents
  sub_agents = [
      LlmAgent(name=f'agent_{i}', model='gemini-2.5-flash') for i in range(100)
  ]

  root_agent = LlmAgent(
      name='root', model='gemini-2.5-flash', sub_agents=sub_agents
  )

  with pytest.raises(ValueError) as exc_info:
    root_agent._LlmAgent__get_agent_to_run('nonexistent')

  error_msg = str(exc_info.value)

  # Verify all agents are shown (no truncation)
  assert 'agent_0' in error_msg  # First agent shown
  assert 'agent_99' in error_msg  # Last agent also shown
  assert 'showing first 20 of' not in error_msg  # No truncation message


class TestSetDefaultModelErrors:
  """Tests for LlmAgent.set_default_model error messages."""

  def test_wrong_type_shows_actual_type(self):
    """TypeError should include the actual type that was passed."""
    with pytest.raises(TypeError, match=r'got int'):
      LlmAgent.set_default_model(123)

  def test_wrong_type_list_shows_actual_type(self):
    """TypeError should show 'list' when a list is passed."""
    with pytest.raises(TypeError, match=r'got list'):
      LlmAgent.set_default_model(['gemini-2.5-flash'])

  def test_empty_string_still_raises_value_error(self):
    """Empty string should still raise ValueError (not changed)."""
    with pytest.raises(ValueError, match=r'non-empty string'):
      LlmAgent.set_default_model('')


class TestSetDefaultLiveModelErrors:
  """Tests for LlmAgent.set_default_live_model error messages."""

  def test_wrong_type_shows_actual_type(self):
    """TypeError should include the actual type that was passed."""
    with pytest.raises(TypeError, match=r'got dict'):
      LlmAgent.set_default_live_model({})

  def test_empty_string_still_raises_value_error(self):
    """Empty string should still raise ValueError (not changed)."""
    with pytest.raises(ValueError, match=r'non-empty string'):
      LlmAgent.set_default_live_model('')


class TestValidateGenerateContentConfigErrors:
  """Tests for LlmAgent.validate_generate_content_config error messages."""

  def test_tools_error_includes_move_guidance(self):
    """Error should tell users to move tools to LlmAgent(tools=[...])."""
    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[])]
    )
    with pytest.raises(ValueError, match=r'Move your tools'):
      LlmAgent.validate_generate_content_config(config)

  def test_system_instruction_error_includes_move_guidance(self):
    """Error should tell users to move instruction to LlmAgent(instruction=...)."""
    config = types.GenerateContentConfig(system_instruction='You are helpful.')
    with pytest.raises(ValueError, match=r'Move your instruction'):
      LlmAgent.validate_generate_content_config(config)

  def test_response_schema_error_includes_move_guidance(self):
    """Error should tell users to move schema to LlmAgent(output_schema=...)."""
    config = types.GenerateContentConfig(response_schema={'type': 'string'})
    with pytest.raises(ValueError, match=r'Move your schema'):
      LlmAgent.validate_generate_content_config(config)
