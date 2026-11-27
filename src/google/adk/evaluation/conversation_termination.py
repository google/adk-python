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

"""Why a simulated conversation stopped generating further user turns."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .simulation.user_simulator import Status


class ConversationTerminationReason(enum.Enum):
  """Why a simulated conversation stopped generating further user turns.

  Persisted on `InferenceResult` / `EvalCaseResult` so eval results can tell
  a clean stop apart from a turn-limit cut-off. Values are stable string
  enums for JSON serialization.
  """

  STOP_SIGNAL = "stop_signal"
  TURN_LIMIT = "turn_limit"
  NO_MESSAGE = "no_message"


def conversation_termination_reason_from_status(
    status: Status,
) -> ConversationTerminationReason | None:
  """Maps a non-SUCCESS simulator `Status` to a persisted termination reason."""
  # Local import avoids a runtime cycle with `user_simulator` -> `evaluator`.
  from .simulation.user_simulator import Status as UserSimulatorStatus

  if status == UserSimulatorStatus.STOP_SIGNAL_DETECTED:
    return ConversationTerminationReason.STOP_SIGNAL
  if status == UserSimulatorStatus.TURN_LIMIT_REACHED:
    return ConversationTerminationReason.TURN_LIMIT
  if status == UserSimulatorStatus.NO_MESSAGE_GENERATED:
    return ConversationTerminationReason.NO_MESSAGE
  return None
