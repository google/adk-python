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

from typing import ClassVar
from typing import Optional

from google.genai import types as genai_types
from pydantic import Field
from typing_extensions import override

from ..events.event import Event
from ..utils.feature_decorator import experimental
from .evaluator import Evaluator
from .user_simulator import BaseUserSimulatorConfig
from .user_simulator import NextUserMessage
from .user_simulator import UserSimulator


class LlmBackedUserSimulatorConfig(BaseUserSimulatorConfig):
  """Contains configurations required by an LLM backed user simulator."""

  model: str = Field(
      default="gemini-2.5-flash",
      description="The model to use for user simulation.",
  )

  model_config: Optional[genai_types.GenerateContentConfig] = Field(
      default=genai_types.GenerateContentConfig,
      description="The configuration for the model.",
  )

  max_allowed_invocations: int = Field(
      default=20,
      description="""Maximum number of invocations allowed by the simulated
interaction.  This property allows us to stop a run-off conversation, where the
agent and the user simulator get into an never ending loop.

(Not recommended)If you don't want a limit, you can set the value to -1.
      """,
  )


@experimental
class LlmBackedUserSimulator(UserSimulator):
  """A UserSimulator that uses a LLM to generate messages on behalf of the user."""

  config_type: ClassVar[type[LlmBackedUserSimulatorConfig]] = (
      LlmBackedUserSimulatorConfig
  )

  def __init__(self, *, config: BaseUserSimulatorConfig):
    super().__init__(config, config_type=LlmBackedUserSimulator.config_type)

  @override
  async def get_next_user_message(
      self,
      conversation_plan: str,
      events: list[Event],
  ) -> NextUserMessage:
    """Returns the next user message to send to the agent with help from a LLM.

    Args:
      conversation_plan: A plan that user simulation system needs to follow as
        it plays out the conversation.
      events: The unaltered conversation history between the user and the
        agent(s) under evaluation.
    """
    raise NotImplementedError()

  @override
  def get_simulation_evaluator(
      self,
  ) -> Evaluator:
    """Returns an Evaluator that evaluates if the simulation was successful or not."""
    raise NotImplementedError()
