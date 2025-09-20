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

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..agents.base_agent import BaseAgent
from ..apps.base_events_compactor import BaseEventsCompactor
from ..plugins.base_plugin import BasePlugin
from ..utils.feature_decorator import experimental


@experimental
class App(BaseModel):
  """Represents an LLM-backed agentic application.

  An `App` is the top-level container for an agentic system powered by LLMs.
  It manages a root agent (`root_agent`), which serves as the root of an agent
  tree, enabling coordination and communication across all agents in the
  hierarchy.
  The `plugins` are application-wide components that provide shared capabilities
  and services to the entire system.
  """

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra="forbid",
  )

  name: str
  """The name of the application."""

  root_agent: BaseAgent
  """The root agent in the application. One app can only have one root agent."""

  plugins: list[BasePlugin] = Field(default_factory=list)
  """The plugins in the application."""

  event_compactor: Optional[BaseEventsCompactor] = None
  """The event compactor strategy for the application."""
