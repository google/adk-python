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

"""Path builder for hierarchical node paths."""

from __future__ import annotations


class _NodePathBuilder:
  """Represents a path to a node in a hierarchical workflow.

  A node path is a sequence of segments, each identifying a node instance,
  typically in the form 'node_name@run_id' or just 'node_name'.
  """

  def __init__(self, segments: list[str]):
    """Initializes a _NodePathBuilder with a list of segments."""
    self._segments = segments

  @classmethod
  def from_string(cls, path_str: str) -> _NodePathBuilder:
    """Parses a _NodePathBuilder from a string representation.

    Example: 'wf@1/node@2'.
    """
    if not path_str:
      return cls([])
    return cls(path_str.split('/'))

  def __str__(self) -> str:
    """Returns the string representation of the path."""
    return '/'.join(self._segments)

  def __eq__(self, other: object) -> bool:
    """Returns True if segments are equal."""
    if not isinstance(other, _NodePathBuilder):
      return NotImplemented
    return self._segments == other._segments

  @property
  def node_name(self) -> str:
    """Returns the node name of the leaf segment."""
    if not self._segments:
      return ''
    return self._segments[-1].rsplit('@', 1)[0]

  @property
  def leaf_segment(self) -> str:
    """Returns the full leaf segment."""
    if not self._segments:
      return ''
    return self._segments[-1]

  @property
  def run_id(self) -> str | None:
    """Returns the run ID of the leaf segment, if any."""
    if not self._segments:
      return None
    parts = self._segments[-1].rsplit('@', 1)
    return parts[1] if len(parts) > 1 else None

  @property
  def parent(self) -> _NodePathBuilder | None:
    """Returns the parent _NodePathBuilder, or None if this is a root path."""
    if len(self._segments) <= 1:
      return None
    return _NodePathBuilder(self._segments[:-1])

  def append(
      self, node_name: str, run_id: str | None = None
  ) -> _NodePathBuilder:
    """Returns a new _NodePathBuilder with the child segment appended."""
    segment = node_name
    if run_id:
      segment = f'{node_name}@{run_id}'
    return _NodePathBuilder(self._segments + [segment])

  def is_descendant_of(self, ancestor: _NodePathBuilder) -> bool:  # pylint: disable=protected-access
    """Checks if this path is a descendant of the ancestor path."""
    if len(self._segments) <= len(ancestor._segments):
      return False
    return self._segments[: len(ancestor._segments)] == ancestor._segments

  def is_direct_child_of(self, parent: _NodePathBuilder) -> bool:  # pylint: disable=protected-access
    """Checks if this path is a direct child of the parent path."""
    if len(self._segments) != len(parent._segments) + 1:
      return False
    return self._segments[:-1] == parent._segments

  def get_direct_child(self, descendant: _NodePathBuilder) -> _NodePathBuilder:  # pylint: disable=protected-access
    """Returns a new _NodePathBuilder for the direct child towards the descendant."""
    if len(descendant._segments) <= len(self._segments):
      raise ValueError('Descendant path is not longer than self path')
    if descendant._segments[: len(self._segments)] != self._segments:
      raise ValueError('Descendant path does not start with self path')
    return _NodePathBuilder(descendant._segments[: len(self._segments) + 1])
