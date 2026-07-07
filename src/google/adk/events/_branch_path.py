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

"""Hierarchical path representation for dynamic execution branches."""

from __future__ import annotations


class _BranchPath:
  """Represents a hierarchical path for execution branches.

  A path consists of dot-separated segments (e.g., 'segment1.segment2'),
  where each segment represents a node run and is typically formatted
  as 'node_name@run_id' or 'node_name'.

  Example:
    'parent_agent@1.collect_user_info_tool@2.sub_workflow'
  """

  def __init__(self, segments: list[str]):
    """Initializes a _BranchPath with a list of segments."""
    self._segments = list(segments)

  @classmethod
  def from_string(cls, path_str: str | None) -> _BranchPath:
    """Parses a _BranchPath from a dot-separated string representation."""
    if not path_str:
      return cls([])
    return cls(path_str.split("."))

  def __str__(self) -> str:
    """Returns the dot-separated string representation of the path."""
    return ".".join(self._segments)

  def __eq__(self, other: object) -> bool:
    """Returns True if segments are equal."""
    if not isinstance(other, _BranchPath):
      return NotImplemented
    return self._segments == other._segments

  @property
  def segments(self) -> list[str]:
    """Returns a copy of the path segments."""
    return list(self._segments)

  @property
  def run_ids(self) -> set[str]:
    """Extracts all run IDs (the part after '@') from all segments in the path.

    Example:
      - Path: 'parent@1.child@2.node'
      - Returns: {'1', '2'}
    """
    ids = set()
    for segment in self._segments:
      parts = segment.rsplit("@", 1)
      if len(parts) > 1 and parts[1]:
        ids.add(parts[1])
    return ids

  @property
  def parent(self) -> _BranchPath | None:
    """Returns the parent _BranchPath, or None if this is a root path."""
    if len(self._segments) <= 1:
      return None
    return _BranchPath(self._segments[:-1])

  def is_descendant_of(self, ancestor: _BranchPath) -> bool:
    """Checks if this path is a descendant of the ancestor path.

    A path is a descendant if it starts with all segments of the ancestor path
    and has additional segments.
    """
    if len(self._segments) <= len(ancestor._segments):
      return False
    return self._segments[: len(ancestor._segments)] == ancestor._segments

  @staticmethod
  def common_prefix(paths: list[_BranchPath]) -> _BranchPath:
    """Finds the common prefix of a list of _BranchPath objects."""
    if not paths:
      return _BranchPath([])

    common_segments = []
    for segments in zip(*[p.segments for p in paths]):
      if len(set(segments)) == 1:
        common_segments.append(segments[0])
      else:
        break
    return _BranchPath(common_segments)

  def append(
      self,
      segment_or_path: str | _BranchPath,
      run_id: str | None = None,
  ) -> _BranchPath:
    """Returns a new _BranchPath with segment(s) appended.

    Args:
      segment_or_path: A segment name (str), dot-separated path (str), or another
        _BranchPath instance to append.
      run_id: Optional run ID (or function_call_id) to format segment as
        'name@run_id'.
    """
    if isinstance(segment_or_path, _BranchPath):
      if run_id is not None:
        raise ValueError(
            "run_id cannot be provided when segment_or_path is a _BranchPath"
            " instance."
        )
      return _BranchPath(self._segments + segment_or_path.segments)

    if run_id is not None:
      if "." in segment_or_path:
        raise ValueError(
            "run_id cannot be provided when segment_or_path is a dot-separated"
            " path."
        )
      segment = f"{segment_or_path}@{run_id}"
      return _BranchPath(self._segments + [segment])

    new_segments = [s for s in segment_or_path.split(".") if s]
    return _BranchPath(self._segments + new_segments)

  @classmethod
  def create_sub_branch(
      cls,
      base_branch: str | None,
      *,
      name: str,
      run_id: str | None = None,
  ) -> str:
    """Creates a new dot-separated branch path string by appending a segment.

    Example:
      _BranchPath.create_sub_branch('parent', name='child', run_id='1') ->
      'parent.child@1'
      _BranchPath.create_sub_branch(None, name='agent') -> 'agent'
    """
    return str(cls.from_string(base_branch).append(name, run_id=run_id))
