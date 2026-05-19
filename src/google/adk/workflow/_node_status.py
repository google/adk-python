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

"""Node execution status enum."""

from __future__ import annotations

from enum import Enum


class NodeStatus(Enum):
  """The status of a node in the workflow graph."""

  INACTIVE = 0
  """The node is not ready to be executed."""

  PENDING = 1
  """The node is ready to be executed."""

  RUNNING = 2
  """The node is being executed."""

  COMPLETED = 3
  """The node has been executed successfully."""

  WAITING = 4
  """The node is waiting (e.g. for a user response or re-trigger)."""

  FAILED = 5
  """The node has failed."""

  CANCELLED = 6
  """The node has been cancelled."""
