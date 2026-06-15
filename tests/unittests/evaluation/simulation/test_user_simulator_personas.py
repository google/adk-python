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

from __future__ import annotations

from google.adk.errors.not_found_error import NotFoundError
from google.adk.evaluation.simulation.user_simulator_personas import UserBehavior
from google.adk.evaluation.simulation.user_simulator_personas import UserPersona
from google.adk.evaluation.simulation.user_simulator_personas import UserPersonaRegistry
import pytest


class TestUserBehavior:
  """Test cases for UserBehavior."""

  def test_create_user_behavior(self):
    """Tests UserBehavior creation."""
    behavior = UserBehavior(
        name="test_behavior",
        description="Test behavior description.",
        behavior_instructions=["instruction1", "instruction2"],
        violation_rubrics=["violation1", "violation2"],
    )
    assert behavior.name == "test_behavior"
    assert behavior.description == "Test behavior description."
    assert behavior.behavior_instructions == ["instruction1", "instruction2"]
    assert behavior.violation_rubrics == ["violation1", "violation2"]

  def test_get_behavior_instructions_str(self):
    """Tests get_behavior_instructions_str method."""
    behavior = UserBehavior(
        name="test_behavior",
        description="Test behavior description.",
        behavior_instructions=["instruction1", "instruction2"],
        violation_rubrics=[],
    )
    assert (
        behavior.get_behavior_instructions_str()
        == "  * instruction1\n  * instruction2"
    )

  def test_get_violation_rubrics_str(self):
    """Tests get_violation_rubrics_str method."""
    behavior = UserBehavior(
        name="test_behavior",
        description="Test behavior description.",
        behavior_instructions=[],
        violation_rubrics=["violation1", "violation2"],
    )
    assert (
        behavior.get_violation_rubrics_str() == "  * violation1\n  * violation2"
    )


class TestUserPersona:
  """Test cases for UserPersona."""

  def test_create_user_persona(self):
    """Tests UserPersona creation."""
    behavior = UserBehavior(
        name="test_behavior",
        description="Test behavior description.",
        behavior_instructions=["instruction1"],
        violation_rubrics=["violation1"],
    )
    persona = UserPersona(
        id="test_persona",
        description="Test persona description.",
        behaviors=[behavior],
    )
    assert persona.id == "test_persona"
    assert persona.description == "Test persona description."
    assert persona.behaviors == [behavior]


class TestUserPersonaRegistry:
  """Test cases for UserPersonaRegistry."""

  def test_register_and_get_persona(self):
    """Tests register_persona and get_persona methods."""
    registry = UserPersonaRegistry()
    persona = UserPersona(
        id="test_persona", description="Test persona", behaviors=[]
    )
    registry.register_persona("persona1", persona)
    assert registry.get_persona("persona1") == persona

  def test_get_persona_not_found(self):
    """Tests get_persona for a non-existent persona."""
    registry = UserPersonaRegistry()
    with pytest.raises(NotFoundError, match="persona2 not found in registry."):
      registry.get_persona("persona2")

  def test_update_persona(self):
    """Tests updating an existing persona in the registry."""
    registry = UserPersonaRegistry()
    persona1 = UserPersona(
        id="test_persona1", description="Test persona 1", behaviors=[]
    )
    persona2 = UserPersona(
        id="test_persona2", description="Test persona 2", behaviors=[]
    )
    registry.register_persona("persona1", persona1)
    assert registry.get_persona("persona1") == persona1
    registry.register_persona("persona1", persona2)
    assert registry.get_persona("persona1") == persona2

  def test_get_registered_personas(self):
    """Tests get_registered_personas method."""
    registry = UserPersonaRegistry()
    persona1 = UserPersona(
        id="test_persona1", description="Test persona 1", behaviors=[]
    )
    persona2 = UserPersona(
        id="test_persona2", description="Test persona 2", behaviors=[]
    )
    registry.register_persona("persona1", persona1)
    registry.register_persona("persona2", persona2)
    registered_personas = registry.get_registered_personas()
    assert len(registered_personas) == 2
    assert persona1 in registered_personas
    assert persona2 in registered_personas
