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

from typing import List

from pydantic import BaseModel


class StatefulParameter(BaseModel):
  """Represents a stateful parameter and its connections."""

  parameter_name: str
  """The name of the shared parameter (e.g., "ticket_id")."""

  creating_tools: List[str]
  """A list of tools that generate this parameter."""

  consuming_tools: List[str]
  """A list of tools that use this parameter as input."""


class ToolConnectionMap(BaseModel):
  """Represents the map of tool connections."""

  stateful_parameters: List[StatefulParameter]
  """A list of stateful parameters and their connections."""
