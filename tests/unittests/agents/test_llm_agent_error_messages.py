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


class TestGenerateContentKwargErrors:
  """Tests for misplaced GenerateContentConfig kwargs on LlmAgent."""

  def test_temperature_kwarg_names_generate_content_config(self):
    """temperature= should name the generate_content_config destination."""
    with pytest.raises(
        ValueError,
        match=(
            r'temperature is a GenerateContentConfig field\. Pass\n?'
            r' *generate_content_config=types\.GenerateContentConfig\(temperature=\.\.\.\)'
        ),
    ):
      LlmAgent(name='test_agent', temperature=0.2)

  def test_multiple_generation_kwargs_named_in_one_error(self):
    """temperature= and top_p= should both appear in a single error."""
    with pytest.raises(ValueError, match=r'temperature.*top_p') as exc_info:
      LlmAgent(name='test_agent', temperature=0.2, top_p=0.95)
    message = str(exc_info.value)
    assert 'generate_content_config=types.GenerateContentConfig(' in message
    assert 'temperature=...' in message
    assert 'top_p=...' in message

  def test_camel_case_alias_names_canonical_field(self):
    """JSON-style aliases should still name the snake_case config field."""
    with pytest.raises(
        ValueError,
        match=r'max_output_tokens is a GenerateContentConfig field',
    ):
      LlmAgent.model_validate({'name': 'test_agent', 'maxOutputTokens': 256})

  def test_system_instruction_kwarg_points_to_instruction(self):
    """system_instruction= should tell users to use instruction=."""
    with pytest.raises(ValueError, match=r'LlmAgent.instruction'):
      LlmAgent(name='test_agent', system_instruction='You are helpful.')

  def test_response_schema_kwarg_points_to_output_schema(self):
    """response_schema= should tell users to use output_schema=."""
    with pytest.raises(ValueError, match=r'LlmAgent.output_schema'):
      LlmAgent(name='test_agent', response_schema={'type': 'string'})

  def test_typo_still_extra_forbidden(self):
    """A real typo must stay extra_forbidden, not a config lecture."""
    misspelled = 'temperature'[:-1]
    with pytest.raises(ValueError, match='Extra inputs are not permitted'):
      LlmAgent(name='test_agent', **{misspelled: 0.2})

  def test_unknown_extra_kwarg_still_forbidden(self):
    """Unrecognized kwargs remain extra_forbidden."""
    with pytest.raises(ValueError, match='Extra inputs are not permitted'):
      LlmAgent(name='test_agent', not_a_real_field=True)

  def test_tools_and_generate_content_config_still_construct(self):
    """tools is on both models and must not be treated as a config field."""

    def _a_tool():
      pass

    agent = LlmAgent(
        name='test_agent',
        tools=[_a_tool],
        generate_content_config=types.GenerateContentConfig(temperature=0.2),
    )
    assert agent.tools
    assert agent.generate_content_config.temperature == pytest.approx(0.2)

  def test_subclass_inherits_the_validator(self):
    """Subclasses of LlmAgent should get the same redirect."""

    class ChildAgent(LlmAgent):
      pass

    with pytest.raises(
        ValueError, match=r'temperature is a GenerateContentConfig field'
    ):
      ChildAgent(name='test_agent', temperature=0.1)

  def test_typo_and_config_field_named_together(self):
    """A typo arriving with a config field should be named in the same error."""
    misspelled = 'temperature'[:-1]
    with pytest.raises(ValueError) as exc_info:
      LlmAgent(name='test_agent', **{misspelled: 0.2, 'top_k': 5})
    message = str(exc_info.value)
    assert 'top_k is a GenerateContentConfig field' in message
    assert 'generate_content_config=types.GenerateContentConfig(top_k=...)' in (
        message
    )
    assert 'Extra inputs are not permitted' in message
    assert misspelled in message

  def test_redirect_and_config_field_named_together(self):
    """Reserved-field redirects and config fields share one error."""
    with pytest.raises(ValueError) as exc_info:
      LlmAgent(
          name='test_agent',
          system_instruction='You are helpful.',
          temperature=0.1,
      )
    message = str(exc_info.value)
    assert 'LlmAgent.instruction' in message
    assert 'temperature is a GenerateContentConfig field' in message
    assert 'generate_content_config=types.GenerateContentConfig(' in message

  def test_http_options_kwarg_points_to_generate_content_config(self):
    """http_options= without base_url should point to generate_content_config."""
    with pytest.raises(
        ValueError, match=r'http_options is a GenerateContentConfig field'
    ):
      LlmAgent(name='test_agent', http_options={'timeout': 10})

  def test_http_options_with_base_url_points_to_model(self):
    """http_options= with base_url should tell users to use model=."""
    with pytest.raises(
        ValueError,
        match=(
            r'Base URL is a transport setting and must be set via'
            r' LlmAgent\.model'
        ),
    ):
      LlmAgent(
          name='test_agent',
          http_options={'base_url': 'https://example.com'},
      )

  def test_base_url_kwarg_points_to_model(self):
    """base_url= should tell users to use model=."""
    with pytest.raises(ValueError, match=r'LlmAgent.model'):
      LlmAgent(name='test_agent', base_url='https://example.com')

  def test_missing_name_reported_alongside_config_field(self):
    """Missing name should be reported alongside misplaced config kwargs."""
    with pytest.raises(ValueError) as exc_info:
      LlmAgent(temperature=0.2)
    message = str(exc_info.value)
    assert "Field 'name' is required" in message
    assert 'temperature is a GenerateContentConfig field' in message
