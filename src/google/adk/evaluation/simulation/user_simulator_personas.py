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

import logging
from typing import Sequence

from pydantic import BaseModel
from pydantic import Field

from ...errors.not_found_error import NotFoundError

logger = logging.getLogger("google_adk." + __name__)


class UserBehavior(BaseModel):
  """Container for the behavior of a persona."""

  name: str = Field(description="Name of the UserBehavior")

  description: str = Field(
      description=(
          "General description of the expected behavior. This will be used in"
          " bot the instructions for the user simulator and the user simulator"
          " evaluator."
      )
  )

  behavior_instructions: list[str] = Field(
      description=(
          "Instructions the user should follow. These will be included in the"
          " instructions for the user simulator."
      )
  )

  violation_rubrics: list[str] = Field(
      description=(
          "Rubrics to evaluate whether the user simulator presents the"
          " behavior. If the user response presents any of these violations,"
          " the evaluator will consider the user simulator response as invalid."
      )
  )

  def get_behavior_instructions_str(self):
    """Returns a string version of the violation rubrics."""
    return "\n".join(f"  * {i}" for i in self.behavior_instructions)

  def get_violation_rubrics_str(self):
    """Returns a string version of the violation rubrics."""
    return "\n".join(f"  * {v}" for v in self.violation_rubrics)


class UserPersona(BaseModel):
  """Container for a persona."""

  id: str = Field(
      description=(
          "Human readable identifier for the UserPersona. Persona registries"
          " will refer to this identifier."
      )
  )

  description: str = Field(
      description=(
          "Description for the UserPersona. This will be included in the"
          " instructions for the user simulator and its verifier."
      )
  )

  behaviors: Sequence[UserBehavior] = Field(
      description=(
          "Sequence of UserBehaviors for the persona. These will be included in"
          " the instructions for the user simulator and its verifier."
      )
  )


class UserPersonaRegistry:
  """A registry for UserPersona instances."""

  def __init__(self):
    self._registry: dict[str, UserPersona] = {}

  def get_persona(self, persona_id: str) -> UserPersona:
    """Returns the User Persona associated with the given id."""
    if persona_id not in self._registry:
      raise NotFoundError(f"{persona_id} not found in registry.")

    return self._registry[persona_id]

  def register_persona(
      self,
      persona_id: str,
      user_persona: UserPersona,
  ):
    """Registers a user persona given the persona id.

    If a mapping already exist, then it is updated.
    """
    if persona_id in self._registry:
      logger.info(
          "Updating User Persona for %s from %s to %s",
          persona_id,
          self._registry[persona_id],
          user_persona,
      )

    self._registry[persona_id] = user_persona

  def get_registered_personas(
      self,
  ) -> list[UserPersona]:
    """Returns the list of User Personas registered so far."""
    return [persona for _, persona in self._registry.items()]
