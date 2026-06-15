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

import textwrap

from google.adk.evaluation.simulation.llm_backed_user_simulator_prompts import _DEFAULT_USER_SIMULATOR_INSTRUCTIONS_TEMPLATE
from google.adk.evaluation.simulation.llm_backed_user_simulator_prompts import _get_user_simulator_instructions_template
from google.adk.evaluation.simulation.llm_backed_user_simulator_prompts import _USER_SIMULATOR_INSTRUCTIONS_WITH_PERSONA_TEMPLATE
from google.adk.evaluation.simulation.llm_backed_user_simulator_prompts import get_llm_backed_user_simulator_prompt
from google.adk.evaluation.simulation.llm_backed_user_simulator_prompts import is_valid_user_simulator_template
from google.adk.evaluation.simulation.user_simulator_personas import UserBehavior
from google.adk.evaluation.simulation.user_simulator_personas import UserPersona
import pytest

_MOCK_DEFAULT_TEMPLATE = textwrap.dedent("""\
  Default template

  # Conversation Plan
  {{conversation_plan}}

  # Conversation History
  {{conversation_history}}

  # Stop signal
  {{stop_signal}}
""").strip()

_MOCK_PERSONA_TEMPLATE = textwrap.dedent("""\
  Persona template

  # Persona Description
  {{persona.description}}
  {% for b in persona.behaviors %}
  ## {{ b.name }}
  {{ b.description }}

  Instructions:
  {{ b.get_behavior_instructions_str() }}
  {% endfor %}
  # Conversation Plan
  {{conversation_plan}}

  # Conversation History
  {{conversation_history}}

  # Stop signal
  {{stop_signal}}
""").strip()


class TestGetUserSimulatorInstructionsTemplate:
  """Test cases for _get_user_simulator_instructions_template."""

  def test_get_user_simulator_instructions_template_default(self):
    assert (
        _get_user_simulator_instructions_template()
        == _DEFAULT_USER_SIMULATOR_INSTRUCTIONS_TEMPLATE
    )

  def test_get_user_simulator_instructions_template_with_custom_instructions(
      self,
  ):
    custom_instructions = "custom instructions"
    assert (
        _get_user_simulator_instructions_template(
            custom_instructions=custom_instructions
        )
        == custom_instructions
    )

  def test_get_user_simulator_instructions_template_with_persona(self):
    user_persona = UserPersona(
        id="test_persona", description="Test persona", behaviors=[]
    )
    assert (
        _get_user_simulator_instructions_template(user_persona=user_persona)
        == _USER_SIMULATOR_INSTRUCTIONS_WITH_PERSONA_TEMPLATE
    )

  def test_get_user_simulator_instructions_template_with_bad_custom_instructions_raises_error(
      self,
  ):
    custom_instructions = "custom instructions"
    user_persona = UserPersona(
        id="test_persona", description="Test persona", behaviors=[]
    )
    with pytest.raises(ValueError):
      _get_user_simulator_instructions_template(
          custom_instructions=custom_instructions, user_persona=user_persona
      )


sample_persona = UserPersona(
    id="test_persona",
    description="Test persona description",
    behaviors=[
        UserBehavior(
            name="Test behavior",
            description="Test behavior description",
            behavior_instructions=["instruction 1", "instruction 2"],
            violation_rubrics=["rubric 1"],
        )
    ],
)


class TestGetLlmBackedUserSimulatorPrompt:
  """Test cases for get_llm_backed_user_simulator_prompt."""

  def test_get_llm_backed_user_simulator_prompt_default(self, mocker):
    mocker.patch(
        "google.adk.evaluation.simulation.llm_backed_user_simulator_prompts._DEFAULT_USER_SIMULATOR_INSTRUCTIONS_TEMPLATE",
        _MOCK_DEFAULT_TEMPLATE,
    )
    prompt = get_llm_backed_user_simulator_prompt(
        conversation_plan="test plan",
        conversation_history="test history",
        stop_signal="test stop",
    )
    expected_prompt = textwrap.dedent("""\
      Default template

      # Conversation Plan
      test plan

      # Conversation History
      test history

      # Stop signal
      test stop""").strip()

    assert prompt == expected_prompt

  def test_get_llm_backed_user_simulator_prompt_with_custom_instructions(self):
    custom_instructions = textwrap.dedent("""\
      Custom instructions:

      # Past history
      {{conversation_plan}}

      # Plan
      {{conversation_plan}}

      # Finished!
      {{stop_signal}}""").strip()
    prompt = get_llm_backed_user_simulator_prompt(
        conversation_plan="test plan",
        conversation_history="test history",
        stop_signal="test stop",
        custom_instructions=custom_instructions,
    )

    expected_prompt = textwrap.dedent("""\
      Custom instructions:

      # Past history
      test plan

      # Plan
      test plan

      # Finished!
      test stop""").strip()
    assert prompt == expected_prompt

  def test_get_llm_backed_user_simulator_prompt_with_persona(self, mocker):
    mocker.patch(
        "google.adk.evaluation.simulation.llm_backed_user_simulator_prompts._USER_SIMULATOR_INSTRUCTIONS_WITH_PERSONA_TEMPLATE",
        _MOCK_PERSONA_TEMPLATE,
    )
    prompt = get_llm_backed_user_simulator_prompt(
        conversation_plan="test plan",
        conversation_history="test history",
        stop_signal="test stop",
        user_persona=sample_persona,
    )
    expected_prompt = textwrap.dedent("""\
      Persona template

      # Persona Description
      Test persona description

      ## Test behavior
      Test behavior description

      Instructions:
        * instruction 1
        * instruction 2

      # Conversation Plan
      test plan

      # Conversation History
      test history

      # Stop signal
      test stop""").strip()
    assert prompt == expected_prompt


class TestIsValidUserSimulatorTemplate:
  """Test cases for is_valid_user_simulator_template."""

  def test_valid_template(self):
    template = "Hello {{ name }}"
    params = ["name"]
    assert is_valid_user_simulator_template(template, params) is True

  def test_invalid_syntax(self):
    template = "Hello {{ name"
    params = ["name"]
    assert is_valid_user_simulator_template(template, params) is False

  def test_missing_parameter(self):
    template = "Hello"
    params = ["name"]
    assert is_valid_user_simulator_template(template, params) is False
