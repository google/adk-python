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


class UiWidget(BaseModel):
  """Rendering metadata for a UI widget associated with an event.

  When present on an Event.actions, the UI renders the widget using the
  specified provider's renderer component.
  """

  model_config = ConfigDict(
      extra='forbid',
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  id: str
  """The unique identifier of the UI widget."""

  provider: str
  """Widget provider identifier. Determines which rendering strategy
  the UI uses.

  Known values:
    - 'mcp': MCP App iframe, rendered with the MCP Apps AppBridge.
  """

  payload: dict[str, Any] = Field(default_factory=dict)
  """Provider-specific data required for rendering.

  If provider is 'mcp', the payload is a dictionary with the following fields:
  {
    "resource_uri: "ui://...",
    "tool": {...},
    "tool_args": {...}
  }
  Future providers can have their set of payload fields.
  """
