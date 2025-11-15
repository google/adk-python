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

from typing import Optional

from ..utils.feature_decorator import experimental
from .eval_case import EvalCase
from .llm_backed_user_simulator import LlmBackedUserSimulator
from .static_user_simulator import StaticUserSimulator
from .user_simulator import BaseUserSimulatorConfig
from .user_simulator import UserSimulator


@experimental
class UserSimulatorProvider:
  """Provides a UserSimulator instance per EvalCase, mixing configuration data
  from the EvalConfig with per-EvalCase conversation data."""

  def __init__(
      self,
      user_simulator_config: Optional[BaseUserSimulatorConfig] = None,
  ):
    if user_simulator_config is None:
      user_simulator_config = BaseUserSimulatorConfig()
    elif not isinstance(user_simulator_config, BaseUserSimulatorConfig):
      # assume that the user simulator will fully validate the config it gets.
      raise ValueError(f"Expect config of type `{BaseUserSimulatorConfig}`.")
    self._user_simulator_config = user_simulator_config

  def provide(self, eval_case: EvalCase) -> UserSimulator:
    """Provide an appropriate user simulator based on the type of conversation data in the EvalCase

    Args:
      eval_case: An EvalCase containing a `conversation` xor a
        `conversation_scenario`.

    Returns:
      A StaticUserSimulator or an LlmBackedUserSimulator based on the type of
      the conversation data.

    Raises:
      ValueError: If no conversation data or multiple types of conversation data
      are provided.
    """
    if eval_case.conversation is None:
      if eval_case.conversation_scenario is None:
        raise ValueError(
            "Neither static invocations nor conversation scenario provided in"
            " EvalCase. Provide exactly one."
        )

      return LlmBackedUserSimulator(
          config=self._user_simulator_config,
          conversation_scenario=eval_case.conversation_scenario,
      )

    else:  # eval_case.conversation is not None
      if eval_case.conversation_scenario is not None:
        raise ValueError(
            "Both static invocations and conversation scenario provided in"
            " EvalCase. Provide exactly one."
        )

      return StaticUserSimulator(static_conversation=eval_case.conversation)
