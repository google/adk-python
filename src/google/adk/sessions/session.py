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

from typing import Any

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr

from ..events.event import Event


class Session(BaseModel):
  """Represents a series of interactions between a user and agents."""

  model_config = ConfigDict(
      extra="forbid",
      arbitrary_types_allowed=True,
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )
  """The pydantic model config."""

  id: str = Field(
      description="Unique identifier of the session.",
      examples=["session-abc123"],
  )
  """The unique identifier of the session."""
  app_name: str = Field(
      description="Application name that owns the session.",
      examples=["hello_world"],
  )
  """The name of the app."""
  user_id: str = Field(
      description="User ID that owns the session.",
      examples=["user-123"],
  )
  """The id of the user."""
  state: dict[str, Any] = Field(
      default_factory=dict,
      description="Current persisted session state.",
      examples=[{"locale": "en-US"}],
  )
  """The state of the session."""
  events: list[Event] = Field(
      default_factory=list,
      description=(
          "Ordered event history for the session, including user, model, and"
          " tool events."
      ),
      examples=[[]],
  )
  """The events of the session, e.g. user input, model response, function
  call/response, etc."""
  last_update_time: float = Field(
      default=0.0,
      description=(
          "Unix timestamp in seconds for the most recent session update."
      ),
      examples=[1_742_000_000.0],
  )
  """The last update time of the session."""

  _storage_update_marker: str | None = PrivateAttr(default=None)
  """Internal storage revision marker used for stale-session detection."""
