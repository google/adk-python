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

"""Abstract base class for durable session checkpoint stores."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from typing import Dict
from typing import Optional


@dataclass
class Checkpoint:
  """Represents a checkpoint for a durable session.

  Attributes:
    session_id: The ID of the session this checkpoint belongs to.
    checkpoint_seq: The sequence number of this checkpoint (monotonically
      increasing).
    created_at: When this checkpoint was created.
    gcs_state_uri: The GCS URI where the full state blob is stored.
    sha256: SHA-256 hash of the state blob for integrity verification.
    size_bytes: Size of the state blob in bytes.
    agent_state: Small agent state stored inline in BigQuery (optional).
    trigger: What triggered this checkpoint (e.g., "async_boundary", "manual").
  """

  session_id: str
  checkpoint_seq: int
  created_at: datetime
  gcs_state_uri: str
  sha256: str
  size_bytes: int
  agent_state: Optional[Dict[str, Any]] = None
  trigger: str = "async_boundary"


@dataclass
class SessionMetadata:
  """Metadata about a durable session.

  Attributes:
    session_id: The unique session identifier.
    status: Current status ("active", "paused", "completed", "failed").
    agent_name: Name of the root agent for this session.
    created_at: When the session was created.
    updated_at: When the session was last updated.
    current_checkpoint_seq: The latest checkpoint sequence number.
    active_lease_id: ID of the current lease holder (if any).
    lease_expiry: When the current lease expires.
    ttl_expiry: When this session should be garbage collected.
    metadata: Additional custom metadata.
  """

  session_id: str
  status: str
  agent_name: str
  created_at: datetime
  updated_at: datetime
  current_checkpoint_seq: int
  active_lease_id: Optional[str] = None
  lease_expiry: Optional[datetime] = None
  ttl_expiry: Optional[datetime] = None
  metadata: Optional[Dict[str, Any]] = None


class DurableSessionStore(abc.ABC):
  """Abstract base class for checkpoint stores.

  A checkpoint store provides persistent storage for session checkpoints,
  enabling recovery from failures and migration across hosts.

  Implementations must provide:
  - Checkpoint write/read operations with two-phase commit
  - Lease management to prevent concurrent modifications
  - Session metadata management
  """

  @abc.abstractmethod
  async def create_session(
      self,
      *,
      session_id: str,
      agent_name: str,
      metadata: Optional[Dict[str, Any]] = None,
  ) -> SessionMetadata:
    """Create a new durable session.

    Args:
      session_id: Unique identifier for the session.
      agent_name: Name of the root agent.
      metadata: Optional custom metadata.

    Returns:
      The created session metadata.

    Raises:
      ValueError: If a session with this ID already exists.
    """

  @abc.abstractmethod
  async def get_session(self, *, session_id: str) -> Optional[SessionMetadata]:
    """Get session metadata.

    Args:
      session_id: The session to retrieve.

    Returns:
      The session metadata, or None if not found.
    """

  @abc.abstractmethod
  async def update_session_status(
      self, *, session_id: str, status: str
  ) -> None:
    """Update the status of a session.

    Args:
      session_id: The session to update.
      status: The new status.
    """

  @abc.abstractmethod
  async def write_checkpoint(
      self,
      *,
      session_id: str,
      checkpoint_seq: int,
      state_blob: bytes,
      agent_state: Optional[Dict[str, Any]] = None,
      trigger: str = "async_boundary",
  ) -> Checkpoint:
    """Write a checkpoint with two-phase commit.

    This operation should:
    1. Upload the state blob to GCS
    2. Record the checkpoint metadata in BigQuery
    3. Update the session's current_checkpoint_seq

    Args:
      session_id: The session to checkpoint.
      checkpoint_seq: The sequence number for this checkpoint.
      state_blob: The serialized state to persist.
      agent_state: Small agent state to store inline (optional).
      trigger: What triggered this checkpoint.

    Returns:
      The created checkpoint.

    Raises:
      ValueError: If the checkpoint_seq is not greater than the current.
    """

  @abc.abstractmethod
  async def read_latest_checkpoint(
      self, *, session_id: str
  ) -> Optional[tuple[Checkpoint, bytes]]:
    """Read the latest checkpoint for a session.

    Args:
      session_id: The session to read.

    Returns:
      A tuple of (checkpoint, state_blob), or None if no checkpoints exist.
    """

  @abc.abstractmethod
  async def read_checkpoint(
      self, *, session_id: str, checkpoint_seq: int
  ) -> Optional[tuple[Checkpoint, bytes]]:
    """Read a specific checkpoint.

    Args:
      session_id: The session to read.
      checkpoint_seq: The checkpoint sequence number.

    Returns:
      A tuple of (checkpoint, state_blob), or None if not found.
    """

  @abc.abstractmethod
  async def acquire_lease(
      self, *, session_id: str, lease_id: str, timeout_seconds: int
  ) -> bool:
    """Attempt to acquire a lease on a session.

    Leases prevent concurrent modifications to a session. Only the lease
    holder can write checkpoints or update session status.

    Args:
      session_id: The session to lease.
      lease_id: A unique identifier for this lease attempt.
      timeout_seconds: How long the lease should be valid.

    Returns:
      True if the lease was acquired, False if another lease is active.
    """

  @abc.abstractmethod
  async def release_lease(self, *, session_id: str, lease_id: str) -> None:
    """Release a lease on a session.

    Args:
      session_id: The session to release.
      lease_id: The lease ID to release (must match the active lease).
    """

  @abc.abstractmethod
  async def renew_lease(
      self, *, session_id: str, lease_id: str, timeout_seconds: int
  ) -> bool:
    """Renew an existing lease.

    Args:
      session_id: The session to renew.
      lease_id: The lease ID to renew (must match the active lease).
      timeout_seconds: New timeout for the lease.

    Returns:
      True if the lease was renewed, False if the lease is not active.
    """

  @abc.abstractmethod
  async def list_checkpoints(
      self, *, session_id: str, limit: int = 10
  ) -> list[Checkpoint]:
    """List checkpoints for a session.

    Args:
      session_id: The session to list checkpoints for.
      limit: Maximum number of checkpoints to return.

    Returns:
      List of checkpoints, ordered by checkpoint_seq descending.
    """

  @abc.abstractmethod
  async def delete_session(self, *, session_id: str) -> None:
    """Delete a session and all its checkpoints.

    Args:
      session_id: The session to delete.
    """
