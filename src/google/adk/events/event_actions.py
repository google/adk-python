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

from typing import Any
from typing import Optional

from google.genai.types import Content
from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..auth.auth_tool import AuthConfig
from ..tools.tool_confirmation import ToolConfirmation


class EventCompaction(BaseModel):
  """The compaction of the events."""

  model_config = ConfigDict(
      extra='forbid',
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )
  """The pydantic model config."""

  start_timestamp: float
  """The start timestamp of the compacted events, in seconds."""

  end_timestamp: float
  """The end timestamp of the compacted events, in seconds."""

  compacted_content: Content
  """The compacted content of the events."""


class EventActions(BaseModel):
  """Represents the actions attached to an event."""

  model_config = ConfigDict(
      extra='forbid',
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )
  """The pydantic model config."""

  skip_summarization: Optional[bool] = None
  """If true, it won't call model to summarize function response.

  Only used for function_response event.
  """

  state_delta: dict[str, object] = Field(default_factory=dict)
  """Indicates that the event is updating the state with the given delta."""

  artifact_delta: dict[str, int] = Field(default_factory=dict)
  """Indicates that the event is updating an artifact. key is the filename,
  value is the version."""

  transfer_to_agent: Optional[str] = None
  """If set, the event transfers to the specified agent."""

  escalate: Optional[bool] = None
  """The agent is escalating to a higher level agent."""

  requested_auth_configs: dict[str, AuthConfig] = Field(default_factory=dict)
  """Authentication configurations requested by tool responses.

  This field will only be set by a tool response event indicating tool request
  auth credential.
  - Keys: The function call id. Since one function response event could contain
  multiple function responses that correspond to multiple function calls. Each
  function call could request different auth configs. This id is used to
  identify the function call.
  - Values: The requested auth config.
  """

  requested_tool_confirmations: dict[str, ToolConfirmation] = Field(
      default_factory=dict
  )
  """A dict of tool confirmation requested by this event, keyed by
  function call id."""

  compaction: Optional[EventCompaction] = None
  """The compaction of the events."""

  end_of_agent: Optional[bool] = None
  """If true, the current agent has finished its current run. Note that there
  can be multiple events with end_of_agent=True for the same agent within one
  invocation when there is a loop."""

  agent_state: Optional[dict[str, Any]] = None
  """The agent state at the current event."""

  rewind_before_invocation_id: Optional[str] = None
  """The invocation id to rewind to. This is only set for rewind event."""
