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

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.simulation import static_user_simulator
from google.adk.evaluation.simulation import user_simulator
from google.genai import types
import pytest


class TestStaticUserSimulator:
  """Test cases for StaticUserSimulator."""

  @pytest.mark.asyncio
  async def test_get_next_user_message(self):
    """Tests that the provided messages are returned in order followed by the stop signal."""
    conversation = [
        Invocation(
            invocation_id="inv1",
            user_content=types.Content(parts=[types.Part(text="message 1")]),
        ),
        Invocation(
            invocation_id="inv2",
            user_content=types.Content(parts=[types.Part(text="message 2")]),
        ),
    ]
    simulator = static_user_simulator.StaticUserSimulator(
        static_conversation=conversation
    )

    next_message_1 = await simulator.get_next_user_message(events=[])
    assert user_simulator.Status.SUCCESS == next_message_1.status
    assert "message 1" == next_message_1.user_message.parts[0].text

    next_message_2 = await simulator.get_next_user_message(events=[])
    assert user_simulator.Status.SUCCESS == next_message_2.status
    assert "message 2" == next_message_2.user_message.parts[0].text

    next_message_3 = await simulator.get_next_user_message(events=[])
    assert user_simulator.Status.STOP_SIGNAL_DETECTED == next_message_3.status
    assert next_message_3.user_message is None
