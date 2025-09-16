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

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse


class TestSpec(BaseModel):
  """Test specification for conformance test cases.

  This is the human-authored specification that defines what should be tested.
  Category and name are inferred from folder structure.
  """

  model_config = ConfigDict(
      extra="forbid",
  )

  description: str
  """Human-readable description of what this test validates."""

  agent: str
  """Name of the ADK agent to test against."""

  user_messages: list[str]
  """Sequence of user messages to send to the agent during test execution."""


class LlmRecording(BaseModel):
  """Paired LLM request and response."""

  model_config = ConfigDict(
      extra="forbid",
  )

  llm_request: LlmRequest
  """The LLM request."""

  llm_response: LlmResponse
  """The LLM response."""


class ToolRecording(BaseModel):
  """Paired tool call and response."""

  model_config = ConfigDict(
      extra="forbid",
  )

  tool_call: types.FunctionCall
  """The tool call."""

  tool_response: types.FunctionResponse
  """The tool response."""


class Recording(BaseModel):
  """Single interaction recording, ordered by request timestamp."""

  model_config = ConfigDict(
      extra="forbid",
  )

  user_message_index: int
  """Index of the user message this recording belongs to (0-based)."""

  agent_name: str
  """Name of the agent."""

  # oneof fields - start
  llm_recording: Optional[LlmRecording] = None
  """LLM request-response pair."""

  tool_recording: Optional[ToolRecording] = None
  """Tool call-response pair."""
  # oneof fields - end


class Recordings(BaseModel):
  """All recordings in chronological order."""

  model_config = ConfigDict(
      extra="forbid",
  )

  recordings: list[Recording] = Field(default_factory=list)
  """Chronological list of all recordings."""
