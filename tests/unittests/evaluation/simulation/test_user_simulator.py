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

from google.adk.evaluation.simulation.user_simulator import NextUserMessage
from google.adk.evaluation.simulation.user_simulator import Status
from google.genai.types import Content
import pytest


def test_next_user_message_validation():
  """Tests post-init validation of NextUserMessage."""
  with pytest.raises(
      ValueError,
      match=(
          "A user_message should be provided if and only if the status is"
          " SUCCESS"
      ),
  ):
    NextUserMessage(status=Status.SUCCESS)

  with pytest.raises(
      ValueError,
      match=(
          "A user_message should be provided if and only if the status is"
          " SUCCESS"
      ),
  ):
    NextUserMessage(status=Status.TURN_LIMIT_REACHED, user_message=Content())

  # these two should not cause exceptions
  NextUserMessage(status=Status.SUCCESS, user_message=Content())
  NextUserMessage(status=Status.TURN_LIMIT_REACHED)
