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

"""Configuration for durable session persistence."""

from __future__ import annotations

from typing import Any
from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..utils.feature_decorator import experimental


@experimental
class DurableSessionConfig(BaseModel):
  """Configuration for durable session persistence.

  Durable sessions provide checkpoint-based persistence that survives process
  restarts, enabling recovery from failures and migration across hosts. This
  goes beyond the basic resumability feature by persisting session state to
  external storage (BigQuery + GCS).

  Attributes:
    is_durable: Whether to enable durable checkpointing.
    checkpoint_policy: When to create checkpoints:
      - "async_boundary": Checkpoint when hitting async/long-running operations
      - "every_turn": Checkpoint after every agent turn
      - "manual": Only checkpoint when explicitly requested
    checkpoint_store: The store to use for persisting checkpoints.
    lease_timeout_seconds: How long a lease is valid before expiring.
    max_checkpoint_size_bytes: Maximum size for checkpoint state blobs.
  """

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra="forbid",
  )

  is_durable: bool = False
  """Whether to enable durable checkpointing."""

  checkpoint_policy: Literal["async_boundary", "every_turn", "manual"] = (
      "async_boundary"
  )
  """When to create checkpoints during execution."""

  checkpoint_store: Optional[Any] = Field(default=None)
  """The store to use for persisting checkpoints (DurableSessionStore)."""

  lease_timeout_seconds: int = Field(default=300, ge=60, le=3600)
  """How long a lease is valid before expiring (60-3600 seconds)."""

  max_checkpoint_size_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
  """Maximum size for checkpoint state blobs (default 10MB)."""
