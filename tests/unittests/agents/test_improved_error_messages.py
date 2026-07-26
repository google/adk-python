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

"""Tests for improved error messages with actionable context.

These tests verify that error messages include:
- The actual type/value that was provided (not just what was expected)
- Actionable guidance on how to fix the issue
- Enough context for users to debug without additional investigation
"""

from unittest.mock import MagicMock

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest

# --- Runner._resolve_app tests ---


class TestRunnerResolveAppErrors:
  """Tests for Runner._resolve_app error messages."""

  def test_multiple_args_shows_which_were_provided(self):
    """Error message should list which arguments were actually provided."""
    from google.adk.runners import Runner

    mock_agent = LlmAgent(name='test', model='gemini-2.5-flash')
    mock_app = MagicMock()
    mock_app.__class__.__name__ = 'App'

    with pytest.raises(ValueError, match=r'but got:.*app=.*agent='):
      Runner._resolve_app(
          app=mock_app,
          app_name=None,
          agent=mock_agent,
          node=None,
          plugins=None,
      )

  def test_no_args_message_includes_guidance(self):
    """Error message should tell the user to pass exactly one argument."""
    from google.adk.runners import Runner

    with pytest.raises(ValueError, match=r'Got none.*Pass exactly one'):
      Runner._resolve_app(
          app=None,
          app_name=None,
          agent=None,
          node=None,
          plugins=None,
      )


# --- LlmAgent.set_default_model tests ---


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


# --- LlmAgent.set_default_live_model tests ---


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


# --- LlmAgent.validate_generate_content_config tests ---


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


# --- LlmRequest.append_instructions tests ---


class TestAppendInstructionsErrors:
  """Tests for LlmRequest.append_instructions error messages."""

  def test_wrong_type_shows_actual_type(self):
    """TypeError should include the actual type that was passed."""
    request = LlmRequest()
    with pytest.raises(TypeError, match=r'got int'):
      request.append_instructions(42)

  def test_wrong_type_dict_shows_actual_type(self):
    """TypeError should show 'dict' when a dict is passed."""
    request = LlmRequest()
    with pytest.raises(TypeError, match=r'got dict'):
      request.append_instructions({'instruction': 'test'})


# --- LlmAgentConfig tests ---


class TestLlmAgentConfigErrors:
  """Tests for LlmAgentConfig validation error messages."""

  def test_both_model_sources_shows_actual_values(self):
    """Error should include the actual model and model_code values."""
    import warnings

    from google.adk.agents.common_configs import CodeConfig
    from google.adk.agents.llm_agent_config import LlmAgentConfig

    with warnings.catch_warnings():
      warnings.simplefilter('ignore', DeprecationWarning)
      with pytest.raises(ValueError, match=r'both were provided.*Got model='):
        LlmAgentConfig(
            name='test_agent',
            instruction='test',
            model='gemini-2.5-flash',
            model_code=CodeConfig(name='my_module.MyModel'),
        )
