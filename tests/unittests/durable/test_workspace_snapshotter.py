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

"""Tests for WorkspaceSnapshotter and utilities."""

import os
import tempfile

from google.adk.durable.workspace_snapshotter import compute_state_hash
from google.adk.durable.workspace_snapshotter import deserialize_state_from_json
from google.adk.durable.workspace_snapshotter import serialize_state_to_json
from google.adk.durable.workspace_snapshotter import WorkspaceSnapshotter
import pytest


class TestSerializationUtilities:
  """Tests for serialization utility functions."""

  def test_serialize_simple_dict(self):
    """Test serialization of simple dictionary."""
    state = {"key": "value", "number": 42}
    blob = serialize_state_to_json(state)

    assert isinstance(blob, bytes)
    assert b"key" in blob
    assert b"value" in blob
    assert b"42" in blob

  def test_deserialize_simple_dict(self):
    """Test deserialization of simple dictionary."""
    blob = b'{"key": "value", "number": 42}'
    state = deserialize_state_from_json(blob)

    assert state == {"key": "value", "number": 42}

  def test_roundtrip_serialization(self):
    """Test roundtrip serialization/deserialization."""
    original = {
        "string": "hello",
        "number": 123,
        "float": 3.14,
        "bool": True,
        "null": None,
        "list": [1, 2, 3],
        "nested": {"a": {"b": "c"}},
    }

    blob = serialize_state_to_json(original)
    restored = deserialize_state_from_json(blob)

    assert restored == original

  def test_serialize_deterministic(self):
    """Test that serialization is deterministic (sorted keys)."""
    state1 = {"z": 1, "a": 2, "m": 3}
    state2 = {"a": 2, "m": 3, "z": 1}

    blob1 = serialize_state_to_json(state1)
    blob2 = serialize_state_to_json(state2)

    assert blob1 == blob2

  def test_compute_state_hash(self):
    """Test state hash computation."""
    state = {"key": "value"}
    hash1 = compute_state_hash(state)

    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256 produces 64 hex characters

  def test_hash_deterministic(self):
    """Test that hash is deterministic."""
    state1 = {"z": 1, "a": 2}
    state2 = {"a": 2, "z": 1}

    assert compute_state_hash(state1) == compute_state_hash(state2)

  def test_hash_changes_with_content(self):
    """Test that hash changes with content."""
    hash1 = compute_state_hash({"key": "value1"})
    hash2 = compute_state_hash({"key": "value2"})

    assert hash1 != hash2


class TestWorkspaceSnapshotter:
  """Tests for WorkspaceSnapshotter."""

  def test_init_default(self):
    """Test default initialization."""
    snapshotter = WorkspaceSnapshotter()

    assert snapshotter.workspace_dir is None
    assert "__pycache__" in snapshotter._exclude_patterns

  def test_init_with_workspace(self):
    """Test initialization with workspace directory."""
    snapshotter = WorkspaceSnapshotter(workspace_dir="/tmp/workspace")

    assert str(snapshotter.workspace_dir) == "/tmp/workspace"

  def test_init_with_custom_excludes(self):
    """Test initialization with custom exclude patterns."""
    snapshotter = WorkspaceSnapshotter(
        workspace_dir="/tmp/workspace",
        exclude_patterns=["*.log", "temp/"],
    )

    assert snapshotter._exclude_patterns == ["*.log", "temp/"]

  def test_should_exclude_pycache(self):
    """Test exclusion of __pycache__ directories."""
    snapshotter = WorkspaceSnapshotter()
    from pathlib import Path

    assert snapshotter._should_exclude(
        Path("/some/path/__pycache__/module.pyc")
    )

  def test_should_exclude_pyc_files(self):
    """Test exclusion of .pyc files."""
    snapshotter = WorkspaceSnapshotter()
    from pathlib import Path

    assert snapshotter._should_exclude(Path("/some/path/module.pyc"))

  def test_should_not_exclude_py_files(self):
    """Test that .py files are not excluded."""
    snapshotter = WorkspaceSnapshotter()
    from pathlib import Path

    assert not snapshotter._should_exclude(Path("/some/path/module.py"))

  def test_should_exclude_git(self):
    """Test exclusion of .git directories."""
    snapshotter = WorkspaceSnapshotter()
    from pathlib import Path

    assert snapshotter._should_exclude(Path("/some/path/.git/config"))

  def test_should_exclude_env(self):
    """Test exclusion of .env files."""
    snapshotter = WorkspaceSnapshotter()
    from pathlib import Path

    assert snapshotter._should_exclude(Path("/some/path/.env"))

  def test_create_snapshot_no_workspace(self):
    """Test that create_snapshot fails without workspace."""
    snapshotter = WorkspaceSnapshotter()

    with pytest.raises(ValueError, match="No workspace directory"):
      snapshotter.create_snapshot()

  def test_create_snapshot_missing_directory(self):
    """Test that create_snapshot fails with missing directory."""
    snapshotter = WorkspaceSnapshotter(workspace_dir="/nonexistent/path")

    with pytest.raises(FileNotFoundError):
      snapshotter.create_snapshot()

  def test_restore_snapshot_no_workspace(self):
    """Test that restore_snapshot fails without workspace."""
    snapshotter = WorkspaceSnapshotter()

    with pytest.raises(ValueError, match="No workspace directory"):
      snapshotter.restore_snapshot(b"data")

  def test_create_and_restore_snapshot(self):
    """Test creating and restoring a workspace snapshot."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create source workspace with files
      source_dir = os.path.join(tmpdir, "source")
      os.makedirs(source_dir)

      # Create test files
      with open(os.path.join(source_dir, "file1.txt"), "w") as f:
        f.write("content1")
      with open(os.path.join(source_dir, "file2.py"), "w") as f:
        f.write("print('hello')")

      # Create subdirectory
      subdir = os.path.join(source_dir, "subdir")
      os.makedirs(subdir)
      with open(os.path.join(subdir, "nested.txt"), "w") as f:
        f.write("nested content")

      # Create snapshot
      snapshotter = WorkspaceSnapshotter(workspace_dir=source_dir)
      blob, sha256, size = snapshotter.create_snapshot()

      assert isinstance(blob, bytes)
      assert len(sha256) == 64
      assert size > 0

      # Restore to different location
      dest_dir = os.path.join(tmpdir, "dest")
      restore_snapshotter = WorkspaceSnapshotter(workspace_dir=dest_dir)
      restore_snapshotter.restore_snapshot(blob)

      # Verify files were restored
      assert os.path.exists(os.path.join(dest_dir, "file1.txt"))
      assert os.path.exists(os.path.join(dest_dir, "file2.py"))
      assert os.path.exists(os.path.join(dest_dir, "subdir", "nested.txt"))

      # Verify content
      with open(os.path.join(dest_dir, "file1.txt")) as f:
        assert f.read() == "content1"

  def test_snapshot_excludes_pycache(self):
    """Test that snapshots exclude __pycache__ directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create workspace with __pycache__
      workspace = os.path.join(tmpdir, "workspace")
      os.makedirs(workspace)

      with open(os.path.join(workspace, "main.py"), "w") as f:
        f.write("print('main')")

      pycache = os.path.join(workspace, "__pycache__")
      os.makedirs(pycache)
      with open(os.path.join(pycache, "main.cpython-311.pyc"), "wb") as f:
        f.write(b"\x00\x00\x00\x00")

      # Create snapshot
      snapshotter = WorkspaceSnapshotter(workspace_dir=workspace)
      blob, _, _ = snapshotter.create_snapshot()

      # Restore and verify __pycache__ was excluded
      dest = os.path.join(tmpdir, "dest")
      restore_snapshotter = WorkspaceSnapshotter(workspace_dir=dest)
      restore_snapshotter.restore_snapshot(blob)

      assert os.path.exists(os.path.join(dest, "main.py"))
      assert not os.path.exists(os.path.join(dest, "__pycache__"))
