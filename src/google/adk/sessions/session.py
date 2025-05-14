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

from typing import Any

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..events.event import Event


class Session(BaseModel):
  """Represents a series of interactions between a user and agents.

  Attributes:
    id: The unique identifier of the session.
    app_name: The name of the app.
    user_id: The id of the user.
    state: The state of the session.
    events: The events of the session, e.g. user input, model response, function
      call/response, etc.
    last_update_time: The last update time of the session.
  """

  model_config = ConfigDict(
      extra='forbid',
      arbitrary_types_allowed=True,
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )
  """The pydantic model config."""

  id: str
  """The unique identifier of the session."""
  app_name: str
  """The name of the app."""
  user_id: str
  """The id of the user."""
  state: dict[str, Any] = Field(default_factory=dict)
  """The state of the session."""
  events: list[Event] = Field(default_factory=list)
  """The events of the session, e.g. user input, model response, function
  call/response, etc."""
  last_update_time: float = 0.0
  """The last update time of the session."""
