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

"""Per-node execution state."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ._node_status import NodeStatus


class NodeState(BaseModel):
  """State of a node in the workflow."""

  model_config = ConfigDict(extra='ignore', ser_json_bytes='base64')

  status: NodeStatus = NodeStatus.INACTIVE
  """The run status of the node."""

  input: Any = None
  """The input provided to the node."""

  attempt_count: int = Field(default=1, exclude_if=lambda v: v == 1)
  """The attempt count for this node run (1-based)."""

  interrupts: list[str] = Field(default_factory=list)
  """The interrupt ids that are pending to be resolved."""

  resume_inputs: dict[str, Any] = Field(default_factory=dict)
  """The responses for resuming the node, keyed by interrupt id."""

  run_counter: int = Field(default=0, exclude_if=lambda v: v == 0)
  """Sequential counter incremented each time the node gets a fresh run.

  Preserving this count independently of run_id prevents path collisions
  if a node switches between custom string IDs and auto-generated numeric IDs.
  """

  run_id: str | None = None
  """The run ID of this node run."""

  parent_run_id: str | None = None
  """The run ID of the parent node which dynamically
  scheduled this node run."""
