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

"""Unit tests for BaseNode helpers."""

from __future__ import annotations

from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._base_node import find_static_node_path
import pydantic
from pydantic import Field
import pytest


class _Parent(BaseNode):
  children: list[BaseNode] = Field(default_factory=list)
  parent_node: BaseNode | None = None


def _build_tree() -> tuple[BaseNode, BaseNode, BaseNode]:
  """root -> (team_a -> worker), (team_b -> worker): two nodes named 'worker'."""
  worker_a = _Parent(name='worker')
  worker_b = _Parent(name='worker')
  team_a = _Parent(name='team_a', children=[worker_a])
  team_b = _Parent(name='team_b', children=[worker_b])
  worker_a.parent_node = team_a
  worker_b.parent_node = team_b
  root = _Parent(name='root', children=[team_a, team_b])
  team_a.parent_node = root
  team_b.parent_node = root
  return root, worker_a, worker_b


def test_find_static_node_path_returns_root_path_for_root():
  """The root resolves to its own name."""
  root, _, _ = _build_tree()
  assert find_static_node_path(root, root) == 'root'


def test_find_static_node_path_disambiguates_same_name_nodes():
  """Nodes sharing a name resolve to distinct paths via their parents."""
  root, worker_a, worker_b = _build_tree()
  assert find_static_node_path(root, worker_a) == 'root/team_a/worker'
  assert find_static_node_path(root, worker_b) == 'root/team_b/worker'


def test_find_static_node_path_handles_cycles_and_back_references():
  """Cycles (e.g. parent back-references) do not cause infinite recursion."""
  root, worker_a, worker_b = _build_tree()
  # Introduce an explicit cycle where a leaf node points back to root.
  worker_a.children.append(root)

  assert find_static_node_path(root, worker_a) == 'root/team_a/worker'
  assert find_static_node_path(root, worker_b) == 'root/team_b/worker'
  assert find_static_node_path(root, BaseNode(name='orphan')) is None


def test_find_static_node_path_returns_none_for_unreachable_node():
  """A node outside the tree has no path."""
  root, _, _ = _build_tree()
  assert find_static_node_path(root, BaseNode(name='orphan')) is None


def test_find_static_node_path_traverses_dict_fields():
  """find_static_node_path discovers child nodes stored inside a dict field."""

  class _DictParent(BaseNode):
    branches: dict[str, BaseNode] = Field(default_factory=dict)

  leaf = BaseNode(name='leaf_worker')
  mid = _DictParent(name='router', branches={'primary': leaf})
  root = _DictParent(name='root', branches={'entry': mid})

  assert find_static_node_path(root, leaf) == 'root/router/leaf_worker'


def test_base_node_rejects_non_identifier_name():
  """BaseNode raises ValidationError when name is not a Python identifier."""
  with pytest.raises(
      pydantic.ValidationError, match='must be a valid Python identifier'
  ):
    BaseNode(name='invalid-node-name')
