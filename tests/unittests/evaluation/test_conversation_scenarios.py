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

"""Tests for ConversationScenario / ConversationScenarios."""

from __future__ import annotations

from google.adk.errors.not_found_error import NotFoundError
from google.adk.evaluation.conversation_scenarios import ConversationScenario
from google.adk.evaluation.conversation_scenarios import ConversationScenarios
from google.adk.evaluation.simulation.pre_built_personas import get_default_persona_registry
from google.adk.evaluation.simulation.user_simulator_personas import UserBehavior
from google.adk.evaluation.simulation.user_simulator_personas import UserPersona
import pydantic
import pytest


def _custom_persona() -> UserPersona:
  return UserPersona(
      id="CUSTOM",
      description="A persona defined inline by the eval author.",
      behaviors=[
          UserBehavior(
              name="Be terse",
              description="Answers in as few words as possible.",
              behavior_instructions=["Reply with at most five words."],
              violation_rubrics=["The reply rambles."],
          )
      ],
  )


def test_user_persona_given_as_id_resolves_to_default_persona():
  """A bare string is looked up in the default persona registry."""
  scenario = ConversationScenario(
      starting_prompt="I need to book a flight.",
      conversation_plan="Book SFO to LAX.",
      user_persona="EXPERT",
  )

  expected = get_default_persona_registry().get_persona("EXPERT")
  assert isinstance(scenario.user_persona, UserPersona)
  assert scenario.user_persona.id == "EXPERT"
  assert scenario.user_persona == expected


def test_user_persona_given_as_unknown_id_raises_not_found():
  """An id absent from the default registry is an error, not a silent None."""
  with pytest.raises(NotFoundError, match="NO_SUCH_PERSONA not found"):
    ConversationScenario(
        starting_prompt="hi",
        conversation_plan="chat",
        user_persona="NO_SUCH_PERSONA",
    )


def test_user_persona_given_as_object_is_kept_verbatim():
  """An explicit UserPersona is not routed through the registry."""
  persona = _custom_persona()

  scenario = ConversationScenario(
      starting_prompt="hi",
      conversation_plan="chat",
      user_persona=persona,
  )

  assert scenario.user_persona == persona


def test_user_persona_defaults_to_none():
  """`user_persona` is optional and defaults to None."""
  scenario = ConversationScenario(
      starting_prompt="hi", conversation_plan="chat"
  )

  assert scenario.user_persona is None


def test_conversation_scenarios_defaults_to_empty_list():
  """The container is usable with no scenarios supplied."""
  assert ConversationScenarios().scenarios == []


def test_conversation_scenarios_round_trips_through_json():
  """Serializing then deserializing preserves every scenario field."""
  scenarios = ConversationScenarios(
      scenarios=[
          ConversationScenario(
              starting_prompt="I need to book a flight.",
              conversation_plan="Book SFO to LAX, then rent a car.",
              user_persona="NOVICE",
          ),
          ConversationScenario(
              starting_prompt="What can you do?",
              conversation_plan="Ask about capabilities and stop.",
          ),
      ]
  )

  restored = ConversationScenarios.model_validate_json(
      scenarios.model_dump_json()
  )

  assert restored == scenarios
  assert restored.scenarios[0].user_persona.id == "NOVICE"
  assert restored.scenarios[1].user_persona is None


def test_conversation_scenarios_parses_camel_case_json():
  """Authored JSON uses camelCase keys; snake_case attributes are populated."""
  scenarios = ConversationScenarios.model_validate({
      "scenarios": [{
          "startingPrompt": "I need to book a flight.",
          "conversationPlan": "Book SFO to LAX.",
          "userPersona": "EVALUATOR",
      }]
  })

  scenario = scenarios.scenarios[0]
  assert scenario.starting_prompt == "I need to book a flight."
  assert scenario.conversation_plan == "Book SFO to LAX."
  assert scenario.user_persona.id == "EVALUATOR"


def test_conversation_scenario_rejects_unknown_field():
  """A misspelled key is rejected rather than silently dropped."""
  with pytest.raises(pydantic.ValidationError) as exc_info:
    ConversationScenario.model_validate({
        "startingPrompt": "I need to book a flight.",
        "conversationPlan": "Book SFO to LAX.",
        "userPersonaa": "EXPERT",
    })

  assert [(e["type"], e["loc"]) for e in exc_info.value.errors()] == [
      ("extra_forbidden", ("userPersonaa",))
  ]
