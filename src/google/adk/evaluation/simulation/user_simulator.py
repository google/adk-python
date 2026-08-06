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

from abc import ABC
import enum
from typing import Optional

from google.genai import types as genai_types
from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator
from pydantic import ValidationError

from ...events.event import Event
from ...utils.feature_decorator import experimental
from ..common import EvalBaseModel
from ..evaluator import Evaluator


class BaseUserSimulatorConfig(BaseModel):
  """Base class for configurations pertaining to user simulator.

  Concrete subclasses MUST override `type` with a `Literal[...]` value
  unique to that subclass (e.g. `Literal["llm_backed"]`).
  """

  type: Optional[str] = Field(
      default=None,
      description=(
          "Discriminator for the concrete config subclass. Each concrete"
          " subclass overrides this with a `Literal[...]` value unique to"
          ' that subclass (e.g. `Literal["llm_backed"]`). The value is'
          " used by `EvalConfig` to route JSON deserialization to the"
          " correct subclass, and by `UserSimulatorProvider` to look up the"
          " matching simulator implementation. The default is `None` on the"
          " base -- it is *not* a valid discriminator value on its own; a"
          " bare `BaseUserSimulatorConfig` cannot be dispatched to any"
          " simulator and must be promoted to a concrete subclass first."
      ),
  )

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
      extra="allow",
  )


class Status(enum.Enum):
  """The resulting status of get_next_user_message()."""

  SUCCESS = "success"
  TURN_LIMIT_REACHED = "turn_limit_reached"
  STOP_SIGNAL_DETECTED = "stop_signal_detected"
  NO_MESSAGE_GENERATED = "no_message_generated"


class NextUserMessage(EvalBaseModel):
  status: Status = Field(
      description="""The resulting status of `get_next_user_message()`.

The caller of `get_next_user_message()` should inspect this field to determine
if the user simulator was able to successfully generate a message or why it was
not able to do so."""
  )

  user_message: Optional[genai_types.Content] = Field(
      description="""The next user message.""", default=None
  )

  @model_validator(mode="after")
  def ensure_user_message_iff_success(self) -> NextUserMessage:
    if (self.status == Status.SUCCESS) == (self.user_message is None):
      raise ValueError(
          "A user_message should be provided if and only if the status is"
          " SUCCESS"
      )
    return self


@experimental
class UserSimulator(ABC):
  """A user simulator for the purposes of automating interaction with an Agent.

  Typically, you must create one user simulator instance per eval case.
  """

  def __init__(
      self,
      config: BaseUserSimulatorConfig,
      config_type: type[BaseUserSimulatorConfig],
  ):
    # Unpack the config to a specific type needed by the class implementing this
    # interface.
    try:
      self._config = config_type.model_validate(config.model_dump())
    except ValidationError as e:
      raise ValueError(f"Expect config of type `{config_type}`.") from e

  async def get_next_user_message(
      self,
      events: list[Event],
  ) -> NextUserMessage:
    """Returns the next user message to send to the agent.

    Args:
      events: The unaltered conversation history between the user and the
        agent(s) under evaluation.

    Returns:
      A NextUserMessage object containing the next user message to send to the
      agent, or a status indicating why no message was generated.
    """
    raise NotImplementedError()

  def get_simulation_evaluator(
      self,
  ) -> Optional[Evaluator]:
    """Returns an instance of an Evaluator that evaluates if the user simulation was successful or not."""
    raise NotImplementedError()


# --------------------------------------------------------------------------- #
# Config-type -> Simulator-type registry
# --------------------------------------------------------------------------- #
#
# The registry maps a concrete `BaseUserSimulatorConfig` subclass to the
# `UserSimulator` implementation that consumes it. It lives here (on the
# base module) rather than on the provider so that new simulator subclasses
# can self-register from their own module at import time without creating a
# circular dependency with `user_simulator_provider`. `UserSimulatorProvider`
# reads from this registry to dispatch based on config type.
_SIMULATOR_BY_CONFIG_TYPE: dict[
    type[BaseUserSimulatorConfig], type[UserSimulator]
] = {}


def register_user_simulator(
    config_type: type[BaseUserSimulatorConfig],
    simulator_type: type[UserSimulator],
) -> None:
  """Register a `UserSimulator` implementation for a given config subclass.

  This is the extension point for new user-simulator types. A new subclass
  ships its own `BaseUserSimulatorConfig` subclass (with a unique
  `Literal[...]` value for its `type` discriminator) and its own
  `UserSimulator` subclass, then calls this function once at import time
  (typically as an epilogue at the bottom of the simulator's own module) to
  wire them together. `UserSimulatorProvider.provide` will then dispatch to
  the new simulator whenever an `EvalConfig` carries a config of that type.

  Args:
    config_type: The concrete `BaseUserSimulatorConfig` subclass.
    simulator_type: The `UserSimulator` subclass that consumes it.
  """
  _SIMULATOR_BY_CONFIG_TYPE[config_type] = simulator_type
