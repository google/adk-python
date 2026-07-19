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

from typing import Optional

from ...utils.feature_decorator import experimental
from ..eval_case import EvalCase
from ._llm_audio_user_simulator import _LlmAudioUserSimulator
from ._llm_audio_user_simulator import LlmAudioUserSimulatorConfig
from .llm_backed_user_simulator import LlmBackedUserSimulator
from .llm_backed_user_simulator import LlmBackedUserSimulatorConfig
from .static_user_simulator import StaticUserSimulator
from .user_simulator import _SIMULATOR_BY_CONFIG_TYPE
from .user_simulator import BaseUserSimulatorConfig
from .user_simulator import register_user_simulator
from .user_simulator import UserSimulator

# --------------------------------------------------------------------------- #
# Built-in user-simulator registrations
# --------------------------------------------------------------------------- #
#
# The provider is the natural home for wiring up ADK's *built-in* simulators
# to the shared dispatch registry. Each new built-in simulator adds one line
# here, right alongside the existing ones.
register_user_simulator(LlmBackedUserSimulatorConfig, LlmBackedUserSimulator)
register_user_simulator(LlmAudioUserSimulatorConfig, _LlmAudioUserSimulator)

# The historical default when the caller supplies no config, or supplies a
# bare `BaseUserSimulatorConfig`. Preserves the pre-discriminator behavior of
# always instantiating `LlmBackedUserSimulator` in the "no config given" case.
_LEGACY_DEFAULT_CONFIG_TYPE: type[BaseUserSimulatorConfig] = (
    LlmBackedUserSimulatorConfig
)


@experimental
class UserSimulatorProvider:
  """Provides a UserSimulator instance per EvalCase, mixing configuration data

  from the EvalConfig with per-EvalCase conversation data.

  Dispatch is driven by the runtime type of `user_simulator_config`, looked
  up against the shared `_SIMULATOR_BY_CONFIG_TYPE` registry in
  `user_simulator`. Built-in simulators are registered at the top of this
  module; third-party simulators register themselves from their own module
  via `register_user_simulator(...)`. Either way, no changes to this class
  are needed.
  """

  def __init__(
      self,
      user_simulator_config: Optional[BaseUserSimulatorConfig] = None,
  ):
    if user_simulator_config is None:
      # No config supplied: fall back to the legacy default subclass so that
      # `provide()` still finds a registered simulator.
      user_simulator_config = _LEGACY_DEFAULT_CONFIG_TYPE()
    elif not isinstance(user_simulator_config, BaseUserSimulatorConfig):
      raise ValueError(f"Expect config of type `{BaseUserSimulatorConfig}`.")

    self._user_simulator_config = user_simulator_config

  def provide(self, eval_case: EvalCase) -> UserSimulator:
    """Provide an appropriate user simulator based on the EvalCase and config.

    Routing:
      * If the EvalCase carries a static `conversation`, return a
        `StaticUserSimulator` (config-agnostic).
      * Otherwise, look up the simulator implementation registered for
        `type(self._user_simulator_config)` and instantiate it.

    Args:
      eval_case: An EvalCase containing a `conversation` xor a
        `conversation_scenario`.

    Returns:
      A `StaticUserSimulator` when the EvalCase carries static invocations,
      otherwise the `UserSimulator` implementation registered for the
      caller's `user_simulator_config` type.

    Raises:
      ValueError: If no conversation data or multiple types of conversation
        data are provided, or if no `UserSimulator` is registered for the
        caller's config type.
    """
    if eval_case.conversation is None:
      if eval_case.conversation_scenario is None:
        raise ValueError(
            "Neither static invocations nor conversation scenario provided in"
            " EvalCase. Provide exactly one."
        )

      config_type = type(self._user_simulator_config)
      simulator_cls = _SIMULATOR_BY_CONFIG_TYPE.get(config_type)
      if simulator_cls is None:
        registered = sorted(
            t.__name__ for t in _SIMULATOR_BY_CONFIG_TYPE.keys()
        )
        raise ValueError(
            "No UserSimulator registered for config type"
            f" `{config_type.__name__}`. Register one via"
            " `register_user_simulator()`. Currently registered:"
            f" {registered}."
        )

      # When the config resolves to the audio decorator, build the inner
      # LlmBackedUserSimulator first and inject it as the text simulator.
      if simulator_cls is _LlmAudioUserSimulator:
        assert isinstance(
            self._user_simulator_config, LlmAudioUserSimulatorConfig
        )
        text_config = LlmBackedUserSimulatorConfig(
            model=self._user_simulator_config.model,
            model_configuration=self._user_simulator_config.model_configuration,
            max_allowed_invocations=self._user_simulator_config.max_allowed_invocations,
            custom_instructions=self._user_simulator_config.custom_instructions,
            include_function_calls=self._user_simulator_config.include_function_calls,
        )
        text_simulator = LlmBackedUserSimulator(
            config=text_config,
            conversation_scenario=eval_case.conversation_scenario,
        )
        return _LlmAudioUserSimulator(
            config=self._user_simulator_config,
            text_simulator=text_simulator,
        )

      return simulator_cls(
          config=self._user_simulator_config,
          conversation_scenario=eval_case.conversation_scenario,
      )

    else:  # eval_case.conversation is not None
      if eval_case.conversation_scenario is not None:
        raise ValueError(
            "Both static invocations and conversation scenario provided in"
            " EvalCase. Provide exactly one."
        )

      # Static conversations replay pre-authored turns.
      static_simulator = StaticUserSimulator(
          static_conversation=eval_case.conversation
      )

      # When an audio config is set, route the static turns through the audio
      # decorator so they are synthesized to audio, just like the scenario
      # case. The `StaticUserSimulator` is injected as the decorator's inner
      # text simulator.
      simulator_cls = _SIMULATOR_BY_CONFIG_TYPE.get(
          type(self._user_simulator_config)
      )
      if simulator_cls is _LlmAudioUserSimulator:
        return _LlmAudioUserSimulator(
            config=self._user_simulator_config,
            text_simulator=static_simulator,
        )

      return static_simulator
