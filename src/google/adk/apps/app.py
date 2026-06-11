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

import re
from typing import Any
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from ..agents.base_agent import BaseAgent
from ..agents.context_cache_config import ContextCacheConfig
from ..plugins.base_plugin import BasePlugin
from ._configs import EventsCompactionConfig
from ._configs import ResumabilityConfig

__all__ = [
    "App",
    "EventsCompactionConfig",
    "ResumabilityConfig",
    "validate_app_name",
]

_VALID_APP_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


def validate_app_name(name: str) -> None:
  """Ensures the provided application name is safe and intuitive."""
  if not _VALID_APP_NAME_RE.match(name):
    raise ValueError(
        f"Invalid app name '{name}': must start with a letter and can only"
        " consist of letters, digits, underscores, and hyphens."
    )
  if name == "user":
    raise ValueError("App name cannot be 'user'; reserved for end-user input.")


class App(BaseModel):
  """Represents an LLM-backed agentic application.

  An `App` is the top-level container for an agentic system powered by LLMs.
  It manages either a root agent (`root_agent`) or a root node (`root_node`),
  which serves as the entry point for execution.

  Exactly one of `root_agent` or `root_node` must be provided.

  The `plugins` are application-wide components that provide shared capabilities
  and services to the entire system.
  """

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra="forbid",
  )

  name: str
  """The name of the application."""

  # Change to Union[BaseAgent, BaseNode, None] after dependency is fixed.
  root_agent: Union[BaseAgent, Any, None] = None
  """The root agent or node in the application.

  Accepts either a BaseAgent or a BaseNode instance.
  """

  plugins: list[BasePlugin] = Field(default_factory=list)
  """The plugins in the application."""

  events_compaction_config: Optional[EventsCompactionConfig] = None
  """The config of event compaction for the application."""

  context_cache_config: Optional[ContextCacheConfig] = None
  """Context cache configuration that applies to all LLM agents in the app."""

  resumability_config: Optional[ResumabilityConfig] = None
  """
  The config of the resumability for the application.
  If configured, will be applied to all agents in the app.
  """

  @model_validator(mode="after")
  def _validate(self) -> App:
    validate_app_name(self.name)
    if self.root_agent is None:
      raise ValueError("root_agent must be provided.")

    from ..workflow._base_node import BaseNode

    if not isinstance(self.root_agent, (BaseAgent, BaseNode)):
      raise TypeError(
          "root_agent must be a BaseAgent or BaseNode instance, got"
          f" {type(self.root_agent).__name__}"
      )
    return self
