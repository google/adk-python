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

from ..events.event import Event
from ..utils.feature_decorator import experimental
from .common import EvalBaseModel
from .evaluator import Evaluator


class BaseUserSimulatorConfig(BaseModel):
  """Base class for configurations pertaining to user simulator."""

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
