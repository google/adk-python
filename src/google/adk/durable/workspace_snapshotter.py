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

"""Workspace snapshot handling for durable sessions."""

from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path
import tarfile
from typing import Any
from typing import Dict
from typing import Optional

logger = logging.getLogger("google_adk." + __name__)


class WorkspaceSnapshotter:
  """Handles workspace file snapshots for durable checkpoints.

  This class provides utilities for creating and restoring snapshots of
  workspace directories, enabling agents to persist and restore file-based
  state across checkpoint boundaries.

  Example:
    ```python
    snapshotter = WorkspaceSnapshotter(workspace_dir="/tmp/agent_workspace")

    # Create a snapshot
    blob, sha256, size = snapshotter.create_snapshot()

    # Later, restore from snapshot
    snapshotter.restore_snapshot(blob)
    ```
  """

  def __init__(
      self,
      workspace_dir: Optional[str] = None,
      exclude_patterns: Optional[list[str]] = None,
  ):
    """Initialize the workspace snapshotter.

    Args:
      workspace_dir: Path to the workspace directory to snapshot.
      exclude_patterns: List of glob patterns to exclude from snapshots.
    """
    self._workspace_dir = Path(workspace_dir) if workspace_dir else None
    self._exclude_patterns = exclude_patterns or [
        "__pycache__",
        "*.pyc",
        ".git",
        ".env",
        "node_modules",
        "*.log",
    ]

  @property
  def workspace_dir(self) -> Optional[Path]:
    """The workspace directory being snapshotted."""
    return self._workspace_dir

  def create_snapshot(self) -> tuple[bytes, str, int]:
    """Create a tarball snapshot of the workspace directory.

    Returns:
      A tuple of (blob_bytes, sha256_hash, size_bytes).

    Raises:
      ValueError: If no workspace directory is configured.
      FileNotFoundError: If the workspace directory doesn't exist.
    """
    if not self._workspace_dir:
      raise ValueError("No workspace directory configured")

    if not self._workspace_dir.exists():
      raise FileNotFoundError(
          f"Workspace directory not found: {self._workspace_dir}"
      )

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
      for path in self._workspace_dir.rglob("*"):
        if path.is_file() and not self._should_exclude(path):
          arcname = path.relative_to(self._workspace_dir)
          tar.add(path, arcname=str(arcname))

    blob = buffer.getvalue()
    sha256 = hashlib.sha256(blob).hexdigest()

    logger.debug(
        "Created workspace snapshot: %d bytes, sha256=%s", len(blob), sha256
    )

    return blob, sha256, len(blob)

  def restore_snapshot(self, blob: bytes) -> None:
    """Restore a workspace from a tarball snapshot.

    Args:
      blob: The snapshot blob previously created by create_snapshot().

    Raises:
      ValueError: If no workspace directory is configured.
    """
    if not self._workspace_dir:
      raise ValueError("No workspace directory configured")

    self._workspace_dir.mkdir(parents=True, exist_ok=True)

    buffer = io.BytesIO(blob)
    with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
      # Filter to prevent path traversal attacks
      safe_members = [
          m for m in tar.getmembers() if not m.name.startswith(("/", ".."))
      ]
      tar.extractall(path=self._workspace_dir, members=safe_members)

    logger.debug(
        "Restored workspace snapshot: %d bytes to %s",
        len(blob),
        self._workspace_dir,
    )

  def _should_exclude(self, path: Path) -> bool:
    """Check if a path should be excluded from snapshots."""
    path_str = str(path)
    for pattern in self._exclude_patterns:
      if pattern.startswith("*"):
        # Suffix match (e.g., *.pyc)
        if path_str.endswith(pattern[1:]):
          return True
      elif pattern in path_str:
        # Contains match (e.g., __pycache__)
        return True
    return False


def serialize_state_to_json(state: Dict[str, Any]) -> bytes:
  """Serialize state dictionary to JSON bytes.

  Args:
    state: The state dictionary to serialize.

  Returns:
    JSON-encoded bytes.
  """
  return json.dumps(state, sort_keys=True, default=str).encode("utf-8")


def deserialize_state_from_json(blob: bytes) -> Dict[str, Any]:
  """Deserialize state from JSON bytes.

  Args:
    blob: JSON-encoded bytes.

  Returns:
    The deserialized state dictionary.
  """
  return json.loads(blob.decode("utf-8"))


def compute_state_hash(state: Dict[str, Any]) -> str:
  """Compute a SHA-256 hash of the state dictionary.

  Args:
    state: The state dictionary to hash.

  Returns:
    The hex-encoded SHA-256 hash.
  """
  blob = serialize_state_to_json(state)
  return hashlib.sha256(blob).hexdigest()
