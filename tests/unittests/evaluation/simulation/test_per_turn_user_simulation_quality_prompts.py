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

from google.adk.evaluation.simulation.per_turn_user_simulator_quality_prompts import _get_latest_turn_user_simulator_quality_prompt_template
from google.adk.evaluation.simulation.per_turn_user_simulator_quality_prompts import _LATEST_TURN_USER_SIMULATOR_EVALUATOR_PROMPT_TEMPLATE
from google.adk.evaluation.simulation.per_turn_user_simulator_quality_prompts import _LATEST_TURN_USER_SIMULATOR_WITH_PERSONA_EVALUATOR_PROMPT_TEMPLATE
from google.adk.evaluation.simulation.per_turn_user_simulator_quality_prompts import get_per_turn_user_simulator_quality_prompt
from google.adk.evaluation.simulation.user_simulator_personas import UserBehavior
from google.adk.evaluation.simulation.user_simulator_personas import UserPersona

_MOCK_DEFAULT_TEMPLATE = textwrap.dedent("""\
  Default template

  # Conversation Plan
  {{conversation_plan}}

  # Conversation History
  {{conversation_history}}

  # Generated User Response
  {{generated_user_response}}

  # Stop signal
  {{stop_signal}}
""").strip()

_MOCK_PERSONA_TEMPLATE = textwrap.dedent("""\
  Persona template

  # Persona Description
  {{persona.description}}
  {% for b in persona.behaviors %}
  ## Criteria: {{ b.name | render_string_filter}}
  {{ b.description | render_string_filter}}

  Mark as FAIL if any of the following Violations occur:
  {{ b.get_violation_rubrics_str() | render_string_filter}}
  {% endfor %}
  # Conversation Plan
  {{conversation_plan}}

  # Conversation History
  {{conversation_history}}

  # Generated User Response
  {{generated_user_response}}

  # Stop signal
  {{stop_signal}}
""").strip()


class TestGetLatestTurnUserSimulatorQualityPrompt:
  """Test cases for get_latest_turn_user_simulator_quality_prompt."""

  def test_get_get_latest_turn_user_simulator_quality_prompt_template_default(
      self,
  ):
    prompt = _get_latest_turn_user_simulator_quality_prompt_template(
        user_persona=None
    )
    assert prompt == _LATEST_TURN_USER_SIMULATOR_EVALUATOR_PROMPT_TEMPLATE

  def test_get_latest_turn_user_simulator_quality_prompt_template_with_persona(
      self,
  ):
    """Tests that the correct prompt is returned when a persona is provided."""
    persona = UserPersona(
        id="test_persona",
        description="Test persona description.",
        behaviors=[
            UserBehavior(
                name="test_behavior",
                description="Test behavior description.",
                behavior_instructions=["instruction1"],
                violation_rubrics=["violation1"],
            )
        ],
    )
    prompt = _get_latest_turn_user_simulator_quality_prompt_template(
        user_persona=persona
    )
    assert (
        prompt
        == _LATEST_TURN_USER_SIMULATOR_WITH_PERSONA_EVALUATOR_PROMPT_TEMPLATE
    )


class TestGetPerTurnUserSimulatorQualityPrompt:
  """Test cases for get_per_turn_user_simulator_quality_prompt."""

  def test_get_per_turn_user_simulator_quality_prompt_default(self, mocker):
    """Tests that the correct prompt is returned when no persona is provided."""
    mocker.patch(
        "google.adk.evaluation.simulation.per_turn_user_simulator_quality_prompts._LATEST_TURN_USER_SIMULATOR_EVALUATOR_PROMPT_TEMPLATE",
        _MOCK_DEFAULT_TEMPLATE,
    )
    prompt = get_per_turn_user_simulator_quality_prompt(
        conversation_plan="plan",
        conversation_history="history",
        generated_user_response="response",
        stop_signal="stop",
        user_persona=None,
    )
    expected_prompt = textwrap.dedent("""\
      Default template

      # Conversation Plan
      plan

      # Conversation History
      history

      # Generated User Response
      response

      # Stop signal
      stop""").strip()
    assert prompt == expected_prompt

  def test_get_per_turn_user_simulator_quality_prompt_with_persona(
      self, mocker
  ):
    """Tests that the correct prompt is returned when a persona is provided."""
    mocker.patch(
        "google.adk.evaluation.simulation.per_turn_user_simulator_quality_prompts._LATEST_TURN_USER_SIMULATOR_WITH_PERSONA_EVALUATOR_PROMPT_TEMPLATE",
        _MOCK_PERSONA_TEMPLATE,
    )
    persona = UserPersona(
        id="test_persona",
        description="Test persona description.",
        behaviors=[
            UserBehavior(
                name="test_behavior",
                description="Test behavior description.",
                behavior_instructions=["instruction1"],
                violation_rubrics=["violation1"],
            )
        ],
    )
    prompt = get_per_turn_user_simulator_quality_prompt(
        conversation_plan="plan",
        conversation_history="history",
        generated_user_response="response",
        stop_signal="stop",
        user_persona=persona,
    )
    expected_prompt = textwrap.dedent("""\
      Persona template

      # Persona Description
      Test persona description.

      ## Criteria: test_behavior
      Test behavior description.

      Mark as FAIL if any of the following Violations occur:
        * violation1

      # Conversation Plan
      plan

      # Conversation History
      history

      # Generated User Response
      response

      # Stop signal
      stop""").strip()
    assert prompt == expected_prompt
