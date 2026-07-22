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

from google.adk.events._node_path_builder import _NodePathBuilder
import pytest


def test_from_string_returns_empty_path_when_string_is_empty():
  """Parsing an empty string returns a path with no segments."""
  path = _NodePathBuilder.from_string('')
  assert str(path) == ''


def test_from_string_parses_single_segment():
  """Parsing a single segment string returns a path with that segment."""
  path = _NodePathBuilder.from_string('wf@1')
  assert str(path) == 'wf@1'


def test_from_string_parses_multiple_segments():
  """Parsing a slash-separated string returns a path with all segments."""
  path = _NodePathBuilder.from_string('wf@1/node@2')
  assert str(path) == 'wf@1/node@2'


def test_str_joins_segments_with_slash():
  """String representation joins segments with slashes."""
  path = _NodePathBuilder(['wf@1', 'node@2'])
  assert str(path) == 'wf@1/node@2'


def test_eq_returns_true_for_same_segments():
  """Equality returns True if both paths have identical segments."""
  path1 = _NodePathBuilder(['wf@1', 'node@2'])
  path2 = _NodePathBuilder(['wf@1', 'node@2'])
  assert path1 == path2


def test_eq_returns_false_for_different_segments():
  """Equality returns False if segments differ."""
  path1 = _NodePathBuilder(['wf@1', 'node@1'])
  path2 = _NodePathBuilder(['wf@1', 'node@2'])
  assert path1 != path2


def test_eq_returns_not_implemented_for_other_types():
  """Equality returns False when comparing with non-_NodePathBuilder objects."""
  path = _NodePathBuilder(['wf@1'])
  assert path != 'wf@1'


def test_node_name_returns_name_without_run_id():
  """node_name returns the name part of the leaf segment, removing @id."""
  path = _NodePathBuilder.from_string('wf@1/node@2')
  assert path.node_name == 'node'


def test_node_name_returns_full_segment_when_no_id_present():
  """node_name returns the full segment if no @id is present."""
  path = _NodePathBuilder.from_string('wf@1/node')
  assert path.node_name == 'node'


def test_run_id_returns_id_when_present():
  """run_id returns the ID part after @ in the leaf segment."""
  path = _NodePathBuilder.from_string('wf@1/node@2')
  assert path.run_id == '2'


def test_run_id_returns_none_when_no_id_present():
  """run_id returns None if no @ is present in the leaf segment."""
  path = _NodePathBuilder.from_string('wf@1/node')
  assert path.run_id is None


def test_parent_returns_prefix_path():
  """parent returns a new path with the last segment removed."""
  path = _NodePathBuilder.from_string('wf@1/node@2')
  parent = path.parent
  assert parent is not None
  assert str(parent) == 'wf@1'


def test_parent_returns_none_for_root_path():
  """parent returns None for a path with a single segment."""
  path = _NodePathBuilder.from_string('wf@1')
  assert path.parent is None


def test_append_adds_segment_with_id():
  """append adds a new segment with the specified name and run ID."""
  path = _NodePathBuilder.from_string('wf@1')
  child = path.append('node', '2')
  assert str(child) == 'wf@1/node@2'


def test_append_adds_segment_without_id():
  """append adds a new segment without run ID if not provided."""
  path = _NodePathBuilder.from_string('wf@1')
  child = path.append('node')
  assert str(child) == 'wf@1/node'


def test_is_descendant_of_returns_true_for_deep_child():
  """is_descendant_of returns True if the path starts with the ancestor segments."""
  ancestor = _NodePathBuilder.from_string('wf@1')
  descendant = _NodePathBuilder.from_string('wf@1/node@2')
  assert descendant.is_descendant_of(ancestor)


def test_is_descendant_of_returns_false_for_unrelated_path():
  """is_descendant_of returns False if the path does not start with ancestor segments."""
  ancestor = _NodePathBuilder.from_string('wf@1')
  other = _NodePathBuilder.from_string('other@1/node@2')
  assert not other.is_descendant_of(ancestor)


def test_is_direct_child_of_returns_true_for_direct_child():
  """is_direct_child_of returns True if the path is exactly one segment longer than parent."""
  parent = _NodePathBuilder.from_string('wf@1')
  child = _NodePathBuilder.from_string('wf@1/node@2')
  assert child.is_direct_child_of(parent)


def test_is_direct_child_of_returns_false_for_grandchild():
  """is_direct_child_of returns False if the path is more than one segment longer."""
  parent = _NodePathBuilder.from_string('wf@1')
  descendant = _NodePathBuilder.from_string('wf@1/inner@1/node@2')
  assert not descendant.is_direct_child_of(parent)


def test_get_direct_child_returns_path_object():
  """get_direct_child returns a new path object for the direct child."""
  parent = _NodePathBuilder.from_string('wf@1')
  descendant = _NodePathBuilder.from_string('wf@1/inner@1/node@2')
  child = parent.get_direct_child(descendant)
  assert isinstance(child, _NodePathBuilder)
  assert str(child) == 'wf@1/inner@1'


def test_get_direct_child_raises_value_error_for_unrelated_path():
  """get_direct_child raises ValueError if descendant does not start with self."""
  parent = _NodePathBuilder.from_string('wf@1')
  other = _NodePathBuilder.from_string('other@1/node@2')
  with pytest.raises(ValueError):
    parent.get_direct_child(other)
