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

from google.adk.evaluation.conversation_termination import (
    ConversationTerminationReason,
)
from google.adk.evaluation.conversation_termination import (
    conversation_termination_reason_from_status,
)
from google.adk.evaluation.simulation.user_simulator import Status
import pytest


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (Status.STOP_SIGNAL_DETECTED, ConversationTerminationReason.STOP_SIGNAL),
        (Status.TURN_LIMIT_REACHED, ConversationTerminationReason.TURN_LIMIT),
        (Status.NO_MESSAGE_GENERATED, ConversationTerminationReason.NO_MESSAGE),
        (Status.SUCCESS, None),
    ],
)
def test_conversation_termination_reason_from_status(status, expected):
  assert conversation_termination_reason_from_status(status) == expected
