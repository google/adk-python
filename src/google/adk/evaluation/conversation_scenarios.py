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

from pydantic import Field

from .common import EvalBaseModel


class ConversationScenario(EvalBaseModel):
  """Scenario for a conversation between a simulated user and the Agent under test."""

  starting_prompt: str
  """Starting prompt for the conversation.

  This prompt acts as the fixed first user message that is given to the Agent.
  Any subsequent user messages are obtained by the system that is simulating the
  user.
  """

  conversation_plan: str
  """A plan that user simulation system needs to follow as it plays out the conversation.

  Example:
  For a Travel Agent that has tools that let it book a flight and car, a sample
  starting prompt could be:

  `I need to book a flight.`

  A conversation plan could look like:

  First, you want to book a one-way flight from SFO to LAX for next Tuesday.
  You prefer a morning flight and your budget is under $150. If the agent finds
  a valid flight, confirm the booking. Once confirmed, your next goal is to rent
  a standard-size car for three days from the airport. Once both tasks are done,
  your overall goal is complete.
  """


class ConversationScenarios(EvalBaseModel):
  """A simple container for the list of ConversationScenario.

  Mainly serves the purpose of helping with serialization and deserialization.
  """

  scenarios: list[ConversationScenario] = Field(
      default_factory=list, description="""A list of ConversationScenario."""
  )
