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

"""Unit tests for _BranchPath.

Verifies that _BranchPath correctly parses, serializes, and manipulates
hierarchical dynamic execution branch paths.
"""

from __future__ import annotations

from google.adk.events._branch_path import _BranchPath
import pytest


def test_from_string_with_empty_string_returns_empty_path():
  """Parsing an empty string returns a _BranchPath with no segments."""
  path = _BranchPath.from_string("")

  assert path.segments == []
  assert str(path) == ""


def test_from_string_with_single_segment_returns_path_with_one_segment():
  """Parsing a single name returns a _BranchPath with one segment."""
  path = _BranchPath.from_string("agent_0")

  assert path.segments == ["agent_0"]
  assert str(path) == "agent_0"


def test_from_string_with_multiple_segments_returns_path_with_all_segments():
  """Parsing a dot-separated string returns a _BranchPath with all segments."""
  path = _BranchPath.from_string("parent.child.node")

  assert path.segments == ["parent", "child", "node"]
  assert str(path) == "parent.child.node"


def test_equality_compares_path_segments():
  """Two _BranchPath objects are equal if and only if their segments match."""
  path1 = _BranchPath.from_string("parent.child")
  path2 = _BranchPath.from_string("parent.child")
  path3 = _BranchPath.from_string("parent.other")

  assert path1 == path2
  assert path1 != path3
  assert path1 != "parent.child"  # Different type


def test_run_ids_extracts_all_run_ids_from_path():
  """run_ids extracts all run IDs (the part after '@') from all segments."""
  # Given paths with various run ID patterns
  path_with_ids = _BranchPath.from_string("parent@1.child@2.node")
  path_no_ids = _BranchPath.from_string("parent.child")
  path_mixed = _BranchPath.from_string("parent@1.child.node@3")

  # Then the extracted run IDs match expectations
  assert path_with_ids.run_ids == {"1", "2"}
  assert path_no_ids.run_ids == set()
  assert path_mixed.run_ids == {"1", "3"}


def test_parent_returns_parent_path_or_none_for_root():
  """parent returns a new _BranchPath excluding the leaf segment, or None."""
  path = _BranchPath.from_string("parent.child.node")

  assert path.parent == _BranchPath.from_string("parent.child")
  assert path.parent.parent == _BranchPath.from_string("parent")
  assert path.parent.parent.parent is None


def test_is_descendant_of_verifies_path_hierarchy_safely():
  """is_descendant_of returns True if the path is a strict sub-path of ancestor."""
  # Given an ancestor and various comparison paths
  ancestor = _BranchPath.from_string("parent.child")
  descendant = _BranchPath.from_string("parent.child.node.leaf")
  not_descendant = _BranchPath.from_string("parent.other")
  same = _BranchPath.from_string("parent.child")

  # Then descendant checks match expectations
  assert descendant.is_descendant_of(ancestor)
  assert not ancestor.is_descendant_of(descendant)
  assert not not_descendant.is_descendant_of(ancestor)
  assert not same.is_descendant_of(ancestor)


def test_is_descendant_of_is_immune_to_partial_name_prefix_match():
  """is_descendant_of compares segments, avoiding partial string prefix bugs."""
  # Given an ancestor and a path that has a partial string prefix but different segment
  ancestor = _BranchPath.from_string("agent_0")
  descendant_with_prefix = _BranchPath.from_string("agent_00.child")

  # Then it is not recognized as a descendant because segments don't match
  assert not descendant_with_prefix.is_descendant_of(ancestor)


def test_common_prefix_finds_longest_shared_path():
  """common_prefix returns the longest common prefix of a list of paths."""
  # Given a list of paths sharing a common prefix
  paths = [
      _BranchPath.from_string("parent.child.node1"),
      _BranchPath.from_string("parent.child.node2.leaf"),
      _BranchPath.from_string("parent.child.node3"),
  ]

  # When finding the common prefix
  result = _BranchPath.common_prefix(paths)

  # Then the result matches the shared parent path
  assert result == _BranchPath.from_string("parent.child")


def test_common_prefix_with_no_shared_path_returns_empty():
  """common_prefix returns an empty path if there is no shared prefix."""
  paths = [
      _BranchPath.from_string("parent.child"),
      _BranchPath.from_string("other.child"),
  ]

  result = _BranchPath.common_prefix(paths)

  assert result == _BranchPath.from_string("")


def test_common_prefix_with_empty_list_returns_empty():
  """common_prefix returns an empty path if the input list is empty."""
  result = _BranchPath.common_prefix([])

  assert result == _BranchPath.from_string("")


def test_constructor_copies_segments_list():
  """_BranchPath copies the input segments list to ensure immutability."""
  segments = ["parent", "child"]
  path = _BranchPath(segments)

  # Mutate the original list
  segments.append("grandchild")

  # The path segments should remain unchanged
  assert path.segments == ["parent", "child"]


def test_run_ids_filters_out_empty_run_ids():
  """run_ids filters out segments with empty run IDs (e.g. ending with '@')."""
  path = _BranchPath.from_string("parent@.child@2.node@")

  assert path.run_ids == {"2"}
