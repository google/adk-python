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

from datetime import datetime
from typing import Optional
import uuid

from google.genai import types
from pydantic import alias_generators
from pydantic import ConfigDict
from pydantic import Field

from ..models.llm_response import LlmResponse
from .event_actions import EventActions


class Event(LlmResponse):
  """Represents an event in a conversation between agents and users.

  It is used to store the content of the conversation, as well as the actions
  taken by the agents like function calls, etc.
  """

  model_config = ConfigDict(
      extra='forbid',
      ser_json_bytes='base64',
      val_json_bytes='base64',
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )
  """The pydantic model config."""

  invocation_id: str = ''
  """The invocation ID of the event. Should be non-empty before appending to a session."""
  author: str
  """'user' or the name of the agent, indicating who appended the event to the
  session."""
  actions: EventActions = Field(default_factory=EventActions)
  """The actions taken by the agent."""

  long_running_tool_ids: Optional[set[str]] = None
  """Set of ids of the long running function calls.
  Agent client will know from this field about which function call is long running.
  only valid for function call event
  """
  branch: Optional[str] = None
  """The branch of the event.

  The format is like agent_1.agent_2.agent_3, where agent_1 is the parent of
  agent_2, and agent_2 is the parent of agent_3.

  Branch is used when multiple sub-agent shouldn't see their peer agents'
  conversation history.
  """

  # The following are computed fields.
  # Do not assign the ID. It will be assigned by the session.
  id: str = ''
  """The unique identifier of the event."""
  timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
  """The timestamp of the event."""

  def model_post_init(self, __context):
    """Post initialization logic for the event."""
    # Generates a random ID for the event.
    if not self.id:
      self.id = Event.new_id()

  def is_final_response(self) -> bool:
    """Returns whether the event is the final response of an agent.

    NOTE: This method is ONLY for use by Agent Development Kit.

    Note that when multiple agents participage in one invocation, there could be
    one event has `is_final_response()` as True for each participating agent.
    """
    if self.actions.skip_summarization or self.long_running_tool_ids:
      return True
    return (
        not self.get_function_calls()
        and not self.get_function_responses()
        and not self.partial
        and not self.has_trailing_code_execution_result()
    )

  def get_function_calls(self) -> list[types.FunctionCall]:
    """Returns the function calls in the event."""
    func_calls = []
    if self.content and self.content.parts:
      for part in self.content.parts:
        if part.function_call:
          func_calls.append(part.function_call)
    return func_calls

  def get_function_responses(self) -> list[types.FunctionResponse]:
    """Returns the function responses in the event."""
    func_response = []
    if self.content and self.content.parts:
      for part in self.content.parts:
        if part.function_response:
          func_response.append(part.function_response)
    return func_response

  def has_trailing_code_execution_result(
      self,
  ) -> bool:
    """Returns whether the event has a trailing code execution result."""
    if self.content:
      if self.content.parts:
        return self.content.parts[-1].code_execution_result is not None
    return False

  @staticmethod
  def new_id():
    return str(uuid.uuid4())
